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
    rprint(
        f"Performance on answer token:\n[b]Rank: {correct_rank: <8} Logit: {logits[-1, answer_token].item():5.2f} Prob: {token_probs[answer_token].item():6.2%} Token: |{answer_str_token}|[/b]"
    )
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{model.to_string(sorted_token_values[i])}|"
        )
    # rprint(f"[b]Ranks of the answer tokens:[/b] {answer_ranks}")

def main():
    # --------------------------- 导入模型 ---------------------------
    MODEL_PATH = "/hpc2hdd/home/hchen763/jhaidata/local_model/DeepSeek-R1-Distill-Qwen-1.5B"
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = transformer_lens.HookedTransformer.from_pretrained(
        "Qwen/Qwen2-1.5B", # 使用Qwen2风格
        hf_model=hf_model, 
        device="cuda", 
        fold_ln=False, 
        center_writing_weights=False, 
        center_unembed=False)

    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True

    # --------------------------- 测试数据 ---------------------------
    clean_subject = '48'
    corrupted_subject = '50'
    clean = f'Natalia sold clips to {clean_subject} of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'
    corrupted = f'Natalia sold clips to {corrupted_subject} of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'

    # 补齐token长度
    clean, corrupted = pad_to_same_token_length(clean, corrupted, model)

    assert len(model.to_str_tokens(clean)) == len(model.to_str_tokens(corrupted))

    labels = ['Euro','Chinese']
    country_idx = model.tokenizer(labels[0],add_special_tokens=False).input_ids[0]
    corrupted_country_idx = model.tokenizer(labels[1],add_special_tokens=False).input_ids[0]

    label = [[country_idx, corrupted_country_idx]]
    label = torch.tensor(label)

    data = ([clean], [corrupted], label)

    # --------------------------- 计算节点 ---------------------------
    g = Graph.from_model(model)
    start_time = time.time()
    # Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
    attribute(model, g, data, partial(logit_diff, loss=True, mean=True), method='EAP-IG-case', ig_steps=100)
    # attribute(model, g, data, partial(direct_logit, loss=True, mean=True), method='EAP-IG-case', ig_steps=30)
    # attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method='EAP-IG', ig_steps=30)
    g.apply_topn(5000, absolute=True)
    g.prune_dead_nodes()

    g.to_json('graph.json')

    gz = g.to_graphviz()
    gz.draw(f'graph.png', prog='dot')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间：{execution_time}秒")

    logits = get_circuit_logits(model, g, data)
    get_component_logits(logits, model, answer_token=model.to_tokens('Euro',prepend_bos=False)[0], top_k=5)

if __name__ == '__main__':
    main()