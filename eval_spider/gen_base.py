import sys
sys.path.append('../code')
import os
import re
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd 
from model.format import build_instruction
from config import EVALUATIION_FILE, SPIDER_DB_PATH, SPIDER_CSV_PATH
from tqdm import tqdm
from eval_spider.evaluate import process_command, parse_sql_from_output

def parse_sql(output_str):
    import re
    output_str = output_str.replace('\n', ' ')
    # the answer is wrapped in ```sql ... ```
    match = re.search(r'```sql(.*?)```', output_str)
    if match:
        return match.group(1).strip()
    match = re.search(r'```(.*?)```', output_str)
    if match:
        return match.group(1).strip()
    return output_str

    

import copy
@torch.inference_mode()
def process(
    model, 
    test_datas:list[dict], 
    max_new_tokens: int=1024,
    temperature: float = 0.01,
    model_type: str = "1",
    num_sample: int = None,
    batch_size: int = 16,
) -> list[dict]:
    test_datas = copy.deepcopy(test_datas)
    test_datas = test_datas[:num_sample]
    eval_answers = []

    
    
    # for instruction, path_csv in tqdm(zip(instructions, path_csvs), total=len(instructions)):
    batched = [test_datas[i:i+batch_size] for i in range(0, len(test_datas), batch_size)]
    
    for cur_batch in tqdm(batched):

        # for i in range(len(cur_batch)):
        #     cur_batch[i]['instruction'] = build_instruction(cur_batch[i]['instruction'].replace('<insert_sep><insert_embs><insert_sep>', ''), model.tokenizer)
        instruction = [build_instruction(cur_batch[i]['instruction'].replace('<insert><insert_embs></insert>|', '').replace('<insert_embs>|', ''), model.tokenizer) for i in range(len(cur_batch))]
        model.tokenizer.padding_side = 'left'
        model.tokenizer.model_max_length = 8192
        
        tokenized = model.tokenizer(instruction, return_tensors='pt', padding = 'longest').to(model.device)
        model_output = model.generate(
                            **tokenized,
                            max_new_tokens=max_new_tokens, 
                            eos_token_id = model.tokenizer.eos_token_id, 
                            pad_token_id = model.tokenizer.eos_token_id,
                            do_sample=False,
                            use_cache=True
                            )
        model_output = model.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        for cur_output, cur_data in zip(model_output, cur_batch):
            output = parse_sql_from_output(cur_output)
            output = str(copy.deepcopy(output))
            # print(output)
            output_dict = {}
            output_dict['model_output'] = cur_output
            output_dict['instruction'] = cur_data['instruction']
            output_dict['gold'] = cur_data['answer']
            output_dict["pred"] = output
            output_dict['db_id'] = cur_data['db_id']
            if 'question_id' in cur_data:
                output_dict['question_id'] = cur_data['question_id']
            eval_answers.append(output_dict)    

    return eval_answers

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype = torch.bfloat16).to(args.device, dtype=torch.bfloat16)
    model.tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=8192,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    #     model.tokenizer.pad_token_id = 128002
     
    test_datas = json.load(open(args.data_path))
    eval_outputs = process(model = model, test_datas = test_datas, batch_size=args.batch_size, num_sample=args.num_sample)
    output_path = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    for file_name in ['model_output.json', 'pred.sql', 'gold.sql', 'result']:
        with open(os.path.join(output_path, file_name), 'w') as f:
            f.write('')
            
    json.dump(eval_outputs, open(os.path.join(output_path, 'model_output.json'), 'w'), indent=2)
    pred = []
    gold = []
    for output in eval_outputs:
        pred.append(parse_sql(output['pred']))
        gold.append(output['gold'].strip().replace('\n', ' ').replace('\t', ' ') + '\t' + output['db_id'])
    pred_path, gold_path, result_path = [os.path.join(output_path, x) for x in ['pred.sql', 'gold.sql', 'result']]
    
    with open(pred_path, 'w') as f:
        f.write('\n'.join(pred))
    with open(gold_path, 'w') as f:
        f.write('\n'.join(gold))
        
    model_output_path = os.path.join(output_path, "model_output.json")
    if args.eval_type == 'r':
        result_path = os.path.join(output_path, 'result_ex')
        process_command(model_output_path, result_path, os.path.join(output_path, 'command_ex.sh'), 'r_ex')
        result_path = os.path.join(output_path, 'result_ts')
        process_command(model_output_path, result_path, os.path.join(output_path, 'command_ts.sh'), 'r_ts')
    else:
        process_command(model_output_path, result_path, os.path.join(output_path, 'command.sh'), args.eval_type)
    
        
    with open(result_path, 'r') as f:
        print(f.read())
    
    
    
if __name__ == "__main__":
    print('gen base!')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--data_path", type=str, default=None, help="Path of json file to be evaluated")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--save_path", type=str, default="output.json", help="Path to save the output")
    # num sample
    parser.add_argument("--num_sample", type=int, default=None, help="Number of samples to evaluate")
    # device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--eval_type", type=str, default='dev', help="Evaluation type")
    args = parser.parse_args()
    main(args)
