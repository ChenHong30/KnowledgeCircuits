# 导入包
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import time
import torch
import pandas as pd
import transformer_lens
import transformer_lens.utils as utils
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from rich import print as rprint
from eap.graph import Graph
from eap.dataset import COT_EAP_Dataset
from eap.attribute import attribute
from eap.metrics import logit_diff, direct_logit
from eap.evaluate import evaluate_graph, evaluate_baseline,get_circuit_logits
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
from datetime import datetime

# 函数定义
def pad_to_same_token_length(s1, s2, model, min_length=32):
    pad_token = model.tokenizer.pad_token
    # 先补齐到等长
    while len(model.to_str_tokens(s1)) < len(model.to_str_tokens(s2)):
        s1 = pad_token + s1
    while len(model.to_str_tokens(s2)) < len(model.to_str_tokens(s1)):
        s2 = pad_token + s2
    # 再统一补到min_length
    current_len = len(model.to_str_tokens(s1))  # 已保证等长
    while current_len < min_length:
        s1 = pad_token + s1
        s2 = pad_token + s2
        current_len += 1
    return s1, s2

def display_token(token_str):
    if token_str == '\n':
        return '\\n'
    elif token_str == '\t':
        return '\\t'
    elif token_str == ' ':
        return '[space]'
    else:
        return token_str

def get_component_logits(logits, model, answer_token, top_k=10, logger=None):
    logits = utils.remove_batch_dim(logits)
    probs = logits.softmax(dim=-1)
    token_probs = probs[-1]
    answer_str_token = model.to_string(answer_token)
    answer_str_token_disp = display_token(answer_str_token)
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
    correct_rank = torch.arange(len(sorted_token_values))[
        (sorted_token_values == answer_token).cpu()
    ].item()
    logger.info(
        f"Performance on answer token: Rank: {correct_rank: <8} Logit: {logits[-1, answer_token].item():5.2f} Prob: {token_probs[answer_token].item():6.2%} Token: |{answer_str_token_disp}|"
    )
    for i in range(top_k):
        token_str = model.to_string(sorted_token_values[i])
        token_str_disp = display_token(token_str)
        logger.info(
            f"Top {i}th token. Logit: {logits[-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{token_str_disp}|"
        )

def get_hf_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_ids, response

def main(is_baseline = False, max_samples=100):
    # 日志配置
    if not is_baseline:
        log_path = 'GSM8K_batch_EAP.log'
    else:
        log_path = 'GSM8K_batch_baseline.log'
    logging.basicConfig(
        filename=log_path,
        filemode='w',  # 覆盖写
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

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

    model.cfg.ungroup_grouped_query_attention = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    dataset = COT_EAP_Dataset("GSM8K_corrupted.csv", max_samples=max_samples)
    dataloader = dataset.to_dataloader(batch_size=1, shuffle=False)

    default_prompt = f"""
    <｜begin▁of▁sentence｜>You are a helpful assistant. You should generate answer as short as possible.
    <｜User｜>"""

    correct_sample = 0
    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        logger.info(f"---------------------------------- Case {idx + 1} / {len(dataloader)} ----------------------------------")
        # 获取数据与处理
        clean, corrupted, clean_subject, corrupted_subject, answer = batch
        clean = clean[0]
        corrupted = corrupted[0]
        clean_subject = clean_subject[0]
        corrupted_subject = corrupted_subject[0]
        # clean_subject = str(clean_subject.item()) if clean_subject.numel() == 1 else str(clean_subject[0].item())
        # corrupted_subject = str(corrupted_subject.item()) if corrupted_subject.numel() == 1 else str(corrupted_subject[0].item())
        clean = clean.format(clean_subject=clean_subject)
        corrupted = corrupted.format(corrupted_subject=corrupted_subject)
        answer = str(answer.item()) if answer.numel() == 1 else str(answer[0].item())

        # 构造prompt
        clean = default_prompt + clean + "</think>"
        corrupted = default_prompt + corrupted + "</think>"

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

        for token_num in range(0, min(len(clean_token_ids), 256, len(corrupted_token_ids))):
            logger.info(f"------------------ Step {token_num + 1} / {min(len(clean_token_ids), 256)} | Case {idx + 1} ------------------")
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

            if not is_baseline:
                # Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
                attribute(model, g, data, partial(logit_diff, loss=True, mean=True), method='EAP-IG-case', ig_steps=30)
                # attribute(model, g, data, partial(direct_logit, loss=True, mean=True), method='EAP-IG-case', ig_steps=30)
                # attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP-IG', ig_steps=30)

                g.apply_topn(30000, absolute=True)
                g.prune_dead_nodes()

                # 在这里添加检查
                if not g.nodes['logits'].in_graph:
                    logger.info(f"**** EMPTY CIRCUIT DETECTED at step {token_num + 1} ****")

                save_dir = f'cot_graph/gsm8k/sample_{idx}'
                os.makedirs(save_dir, exist_ok=True)
                g.to_json(f'{save_dir}/graph_{token_num}.json')

                # gz = g.to_graphviz()
                # gz.draw(f'{save_dir}/graph_{token_num}.png', prog='dot')
            # TODO: case 22 step 58 有错误 暂时跳过处理
            try:
                logits = get_circuit_logits(model, g, data)
            except AssertionError as e:
                logger.info(f"Shape mismatch, skipping.")
                continue
            get_component_logits(logits, model, answer_token=clean_label, top_k=5, logger=logger)

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
            # 检测是否生成结束
            if next_token_id == hf_tokenizer.eos_token_id:
                logger.info("EOS token generated. Stopping generation.")
                break
        
        if answer in current_sentence:
            logger.info(f"√ Correct Answer: {answer}")
            correct_sample += 1
        else:
            logger.info(f"× Incorrect Answer, the correct answer is {answer}")
        logger.info("Total Time: %.2fs", time.time() - case_time)

    logger.info(f"Total ACC: {correct_sample / len(dataloader):.2f}")

if __name__ == '__main__':
    # main(is_baseline=True, max_samples=100)
    main(is_baseline=False, max_samples=100)