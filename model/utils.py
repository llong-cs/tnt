import os, re
import transformers
from typing import Dict
from config import INSERT_EMBS_TOKEN, INSERT_EMBS_TOKEN_ID, INSERT_START_TOKEN, INSERT_END_TOKEN, INSERT_END_TOKEN_LLAMA3, INSERT_START_TOKEN_LLAMA3
import torch
import pandas as pd
from tqdm import tqdm

def find_correct_case_file_name(path, name):
    ls = os.listdir(path)
    ls = [x.split('.')[0] for x in ls]
    for gt in ls:
        if gt.lower() == name.lower():
            return gt
    for gt in ls:
        if name.lower() in gt.lower():
            return gt
    raise ValueError(f'path {path}, name "{name}" not found')





def tokenize_insert_llama3(prompt: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    '''
    Tokenizes the input prompt by inserting a separator token between each chunk of text.

    Args:
        prompt (str): The input prompt to be tokenized. It contains one or more instances of the INSERT_EMBS_TOKEN.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer object used for tokenization.

    Returns:
        torch.Tensor: The tokenized input prompt as a tensor of input IDs. You need to move to the correct device before using it.

    '''
    # prompt = prompt.replace(INSERT_START_TOKEN, INSERT_START_TOKEN_LLAMA3).replace(INSERT_END_TOKEN, INSERT_END_TOKEN_LLAMA3)
    prompt = prompt.replace(INSERT_START_TOKEN, '').replace(INSERT_END_TOKEN, '')
    prompt_chunks = [tokenizer(e, padding="longest", max_length=tokenizer.model_max_length, truncation=True).input_ids for e in prompt.split(INSERT_EMBS_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id: 
        offset = 1
        input_ids.append(prompt_chunks[0][0])
       

    for x in insert_separator(prompt_chunks, [INSERT_EMBS_TOKEN_ID] * (offset + 1)): 
        input_ids.extend(x[offset:])
    assert max(input_ids) < 128256, max(input_ids)
    return torch.tensor(input_ids, dtype=torch.long)

def tokenize_insert(prompt: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    '''
    Tokenizes the input prompt by inserting a separator token between each chunk of text.

    Args:
        prompt (str): The input prompt to be tokenized. It contains one or more instances of the INSERT_EMBS_TOKEN.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer object used for tokenization.

    Returns:
        torch.Tensor: The tokenized input prompt as a tensor of input IDs. You need to move to the correct device before using it.

    '''
    
    prompt_chunks = [tokenizer(e, padding="longest", max_length=tokenizer.model_max_length, truncation=True).input_ids for e in prompt.split(INSERT_EMBS_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [INSERT_EMBS_TOKEN_ID] * (offset + 1)): 
        input_ids.extend(x[offset:])
    return torch.tensor(input_ids, dtype=torch.long)

def ray_work(func, data, num_gpus, num_gpus_per_worker, devices):
    import ray
    NUM_GPUS = num_gpus
    os.environ['CUDA_VISIBLE_DEVICES']=devices
    NUM_GPUS_PER_WORKER = num_gpus_per_worker
    NUM_PROCESSES = int(NUM_GPUS // NUM_GPUS_PER_WORKER)
    print(f'NUM_GPUS: {NUM_GPUS}, NUM_GPUS_PER_WORKER: {NUM_GPUS_PER_WORKER}, NUM_PROCESSES: {NUM_PROCESSES}')

    ray.shutdown()
    ray.init()
    CHUNK_SIZE = len(data) // NUM_PROCESSES + 1
    get_answers_func = ray.remote(num_gpus=NUM_GPUS_PER_WORKER)(func,).remote
    cur_data = [data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(NUM_PROCESSES)]
    print(len(cur_data))
    futures = [get_answers_func(tt_data) for tt_data in cur_data]
    ret = ray.get(futures)
    ray.shutdown()
    ret = [r for r in ret if r for r in r]
    return ret

def process_pool_work(func, data, num_workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    CHUNK_SIZE = len(data) // num_workers + 1
    data_split = [data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(num_workers)]

    ret = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(func, data) for data in data_split]
        for future in tqdm(as_completed(futures)):
            ret.extend(future.result())
    return ret

def process_pool_work_2(func, data, num_workers):
    import multiprocessing

    CHUNK_SIZE = len(data) // num_workers + 1
    data_split = [data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(num_workers)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(func, data_split)

    ret = []
    for result in results:
        ret.extend(result)
    return ret

def build_instruction(prompt, tokenizer):
    """
    Apply the chat template to the user prompt

    Args:
        prompt (str): The user prompt.
        tokenizer: The tokenizer object.

    Returns:
        str: The instruction text.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    decoder_input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return decoder_input_text