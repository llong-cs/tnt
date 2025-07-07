import sys
sys.path.append('../code')
from eval_spider.evaluate import process_command, replace_names_force
import os
import re
import torch
import json
import pandas as pd 
from model.format import build_instruction
from config import EVALUATIION_FILE
from config import INSERT_EMBS_TOKEN, INSERT_EMBS_TOKEN_ID, INSERT_START_TOKEN, INSERT_END_TOKEN, INSERT_END_TOKEN_LLAMA3, INSERT_START_TOKEN_LLAMA3
from tqdm import tqdm

def parse_sql(output_str):
    import re
    output_str = output_str.replace('\n', ' ')
    # the answer is wrapped in ```sql ... ```
    match = re.search(r'```sql(.*?)```', output_str)
    if match:
        return match.group(1).strip()
    # the answer is wrapped in ``` ... ```
    match = re.search(r'```(.*?)```', output_str)
    if match:
        return match.group(1).strip()
    return output_str
def get_sql(output_str, column_mapping):
    output_str = parse_sql(output_str)
    # reverse column mapping
    rev_col_map = {v: k for k, v in column_mapping.items()}
    output_str = replace_names_force(output_str, {}, column_mapping=rev_col_map)
    return output_str

    
import copy
@torch.inference_mode()
def process(
    model, 
    test_datas:list[dict], 
    max_new_tokens: int=2048,
    temperature: float = 0.01,
    model_type: str = "1",
    num_sample: int = None,
    batch_size: int = 2,
) -> list[dict]:
    test_datas = copy.deepcopy(test_datas)
    test_datas = test_datas[:num_sample]
    eval_answers = []

    
    batched_data = [test_datas[i:i+batch_size] for i in range(0, len(test_datas), batch_size)]
    
    # for instruction, path_csv in tqdm(zip(instructions, path_csvs), total=len(instructions)):
    for batch in tqdm(batched_data):
        instruction = [build_instruction(data['instruction'], model.tokenizer) for data in batch]
        path_csv = [data['path_csvs'] for data in batch]
        
        model_output = model.generate(
                            instruction,
                            max_new_tokens=max_new_tokens, 
                            # eos_token_id = model.tokenizer.eos_token_id, 
                            pad_token_id = model.tokenizer.eos_token_id,
                            path_csv=path_csv,
                            top_p = None,
                            temperature = None,
                            do_sample=False
                            )[1]
        # torch.cuda.empty_cache()
        for cur_output, cur_data in zip(model_output, batch):
            output = {}
            output['instruction'] = cur_data['instruction']
            output['gold'] = cur_data['answer_ori']
            output['pol_gold'] = cur_data['answer']
            output['pol_pred'] = parse_sql(cur_output)
            output["pred"] = get_sql(cur_output, cur_data['column_mapping'])
            output['model_output'] = cur_output
            output['db_id'] = cur_data['db_id']
            if 'question_id' in cur_data:
                output['question_id'] = cur_data['question_id']
            eval_answers.append(output)                

    return eval_answers

def main(args):
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    json.dump(vars(args), open(os.path.join(output_path, 'args.json'), 'w'), indent=2)
    
    model_output_path = os.path.join(output_path, 'model_output.json')
    if os.path.exists(model_output_path) == False:
        if args.model_type == 'llama3':
            from model.model_sft_qformer import Model
        elif args.model_type == 'mistral':
            from model.model_sft_qf_mistral import Model
        elif args.model_type == 'codellama':
            from model.model_sft_qf_codellama import Model
        model = Model.from_pretrained(args.model_path).to(args.device, dtype=torch.bfloat16)
        assert hasattr(model, 'max_length') == False
        
        test_datas = json.load(open(args.data_path))
        if args.data_num is not None:
            import random
            random.seed(42)
            test_datas = random.sample(test_datas, data_num)
        for test_data in test_datas:
            test_data['instruction'] = test_data['instruction'].replace(INSERT_START_TOKEN, '').replace(INSERT_END_TOKEN, '')
        data_num = args.data_num
        if data_num is not None:
            test_datas = test_datas[:data_num]
        eval_outputs = process(model = model, test_datas = test_datas, batch_size=args.batch_size, num_sample=args.num_sample)
        json.dump(eval_outputs, open(model_output_path, 'w'), indent=2)
    else:
        eval_outputs = json.load(open(model_output_path))
    # dumps args into output_path/args.json


    
    pred_path, gold_path, result_path = [os.path.join(output_path, x) for x in ['pred.sql', 'gold.sql', 'result']]
    
    if args.eval_type == 'r':
        result_path = os.path.join(output_path, 'result_ts')
        process_command(model_output_path, result_path, os.path.join(output_path, 'command_ts.sh'), 'r_ts')
    else:
        process_command(model_output_path, result_path, os.path.join(output_path, 'command.sh'), args.eval_type)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument('--model_type', type=str, default = 'llama3')
    parser.add_argument("--data_path", type=str, default=None, help="Path of json file to be evaluated")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    # num sample
    parser.add_argument("--num_sample", type=int, default=None, help="Number of samples to evaluate")
    # device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--data_num", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--eval_type", type=str, default='dev', help="Evaluation type")
    args = parser.parse_args()
    main(args)
    
