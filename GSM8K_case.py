# 导入包
import time
import torch
import pandas as pd
import transformer_lens
import transformer_lens.utils as utils
import torch.nn.functional as F

from functools import partial
from rich import print as rprint
from eap.graph import Graph
from eap.dataset import EAPDataset
from eap.attribute import attribute
from eap.metrics import logit_diff, direct_logit
from eap.evaluate import evaluate_graph, evaluate_baseline,get_circuit_logits
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
from datetime import datetime

# 日志配置
log_path = 'GSM8K_case.log'
logging.basicConfig(
    filename=log_path,
    filemode='w',  # 覆盖写
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 函数定义
def pad_to_same_token_length(s1, s2, model):
    # 不断补空格，直到token长度一样
    while len(model.to_str_tokens(s1)) < len(model.to_str_tokens(s2)):
        s1 += ' '
    while len(model.to_str_tokens(s2)) < len(model.to_str_tokens(s1)):
        s2 += ' '
    return s1, s2

def get_component_logits(logits, model, answer_token, top_k=10):
    logits = utils.remove_batch_dim(logits)
    # print(heads_out[head_name].shape)
    probs = logits.softmax(dim=-1)
    token_probs = probs[-1]
    answer_str_token = model.to_string(answer_token)
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
    # Janky way to get the index of the token in the sorted list - I couldn't find a better way?
    correct_rank = torch.arange(len(sorted_token_values))[
        (sorted_token_values == answer_token).cpu()
    ].item()
    # answer_ranks = []
    # answer_ranks.append((answer_str_token, correct_rank))
    # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
    # rprint gives rich text printing
    logger.info(
        f"Performance on answer token: Rank: {correct_rank: <8} Logit: {logits[-1, answer_token].item():5.2f} Prob: {token_probs[answer_token].item():6.2%} Token: |{answer_str_token}|"
    )
    for i in range(top_k):
        logger.info(
            f"Top {i}th token. Logit: {logits[-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{model.to_string(sorted_token_values[i])}|"
        )
    # rprint(f"[b]Ranks of the answer tokens:[/b] {answer_ranks}")

def get_hf_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_ids, response

def main():
    # --------------------------- 导入模型 ---------------------------
    MODEL_PATH = "/hpc2hdd/home/hchen763/jhaidata/local_model/DeepSeek-R1-Distill-Qwen-1.5B"
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    model = transformer_lens.HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B", # 使用Qwen2风格
        hf_model=hf_model, 
        tokenizer=hf_tokenizer,
        device="cuda", 
        fold_ln=False, 
        center_writing_weights=False, 
        center_unembed=False)

    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # --------------------------- 测试数据 ---------------------------
    clean_subject = '48'
    corrupted_subject = 'many'
    
    clean = f"""
    <｜begin▁of▁sentence｜>You are a helpful assistant. You should generate answer as short as possible.
    <｜User｜>Natalia sold clips to {clean_subject} of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</think>
    """
    corrupted = f"""
    <｜begin▁of▁sentence｜>You are a helpful assistant. You should generate answer as short as possible.
    <｜User｜>Natalia sold clips to {corrupted_subject} of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</think>
    """

    # 补齐token长度
    clean, corrupted = pad_to_same_token_length(clean, corrupted, model)

    assert len(model.to_str_tokens(clean)) == len(model.to_str_tokens(corrupted))

    # 获得ground truth
    hf_model.to("cuda")
    clean_token_ids, clean_text = get_hf_response(hf_model, hf_tokenizer, clean)
    corrupted_token_ids, corrupted_text = get_hf_response(hf_model, hf_tokenizer, corrupted)

    logger.info(f"Clean text: {clean_text}")
    logger.info(f"Corrupted text: {corrupted_text}")

    generated_tokens = []
    case_time = time.time()

    for token_num in range(0, min(len(clean_token_ids), 256)):
        logger.info(f"---------------------------------- Step {token_num + 1} / {min(len(clean_token_ids), 256)} ----------------------------------")
        past_text_clean = hf_tokenizer.decode(clean_token_ids[:token_num], skip_special_tokens=False)
        past_text_corrupted = hf_tokenizer.decode(corrupted_token_ids[:token_num], skip_special_tokens=False)
        step_prompt_clean = clean + past_text_clean
        step_prompt_corrupted = corrupted + past_text_corrupted
        # 当前步的 label
        clean_label = clean_token_ids[token_num]
        corrupted_label = corrupted_token_ids[token_num]
        label = [[clean_label, corrupted_label]]
        label = torch.tensor(label)
        data = ([step_prompt_clean], [step_prompt_corrupted], label)

        g = Graph.from_model(model)
        # Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
        attribute(model, g, data, partial(logit_diff, loss=True, mean=True), method='EAP-IG-case', ig_steps=100)
        # attribute(model, g, data, partial(direct_logit, loss=True, mean=True), method='EAP-IG-case', ig_steps=30)
        # attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP-IG', ig_steps=30)

        g.apply_topn(3000, absolute=True)
        g.prune_dead_nodes()

        g.to_json(f'cot_graph/graph_{token_num}.json')

        # gz = g.to_graphviz()
        # gz.draw(f'cot_graph/graph_{token_num}.png', prog='dot')

        logits = get_circuit_logits(model, g, data)
        get_component_logits(logits, model, answer_token=clean_label, top_k=5)

        # 获取基于子图的回复
        logits = utils.remove_batch_dim(logits)  # shape: (seq_len, vocab)
        token_probs = logits[-1].softmax(dim=-1)
        # 取概率最大（top1）的token id
        next_token_id = token_probs.argmax(dim=-1).item()
        # 转成字符串
        next_token_str = model.to_string(next_token_id)
        generated_tokens.append(next_token_id)
        
        current_sentence = model.to_string(generated_tokens)
        logger.info(f"Current sentence: {current_sentence}")
    
    logger.info("Total Time: %.2fs", time.time() - case_time)

if __name__ == '__main__':
    main()