import os
import pandas as pd
from dotenv import load_dotenv
import shutil
from tqdm import tqdm
import sys
import re
import numpy as np


print(sys.executable)


load_dotenv()
DATASETS_PATH = os.environ["DATASETS_PATH"]

def filter_by_size(src_directory, tar_directory):
    # create the target directory if it does not exist
    if not os.path.exists(tar_directory):
        os.makedirs(tar_directory)
    
    # iterate over all files in the directory
    files_to_delete = []
    for filename in tqdm(os.listdir(src_directory)):
        for csv_files in os.listdir(f"{src_directory}/{filename}"):
            if csv_files.endswith('.csv'):
                file_path = f"{src_directory}/{filename}/{csv_files}"
                table_df = pd.read_csv(file_path, encoding='utf-8', low_memory=False, nrows=200)
                if table_df.shape[0] >= 10 and table_df.shape[1] >= 3:
                    shutil.copy(file_path, f"{tar_directory}/{filename}_{csv_files}")
                    
def filter_by_col_name(src_directory, tar_directory):
    # create the target directory if it does not exist
    tar_non_semantic = os.path.join(tar_directory, "non_semantic")
    tar_semantic = os.path.join(tar_directory, "semantic")
    if not os.path.exists(tar_non_semantic):
        os.makedirs(tar_non_semantic)
    if not os.path.exists(tar_semantic):
        os.makedirs(tar_semantic)
        
    # define the patterns of non_semantic column names
    patterns = [
        re.compile(r'^[A-Za-z]+\d+$'), 
        re.compile(r'^\d+$'), 
        re.compile(r'^[a-zA-Z]{1,2}$'), 
        re.compile(r'^[a-zA-Z]+_\d+$'), 
    ]
        
    # iterate over all files in the directory
    for csv_file in tqdm(os.listdir(src_directory)):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(src_directory, csv_file)
            table_df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            non_semantic_num = 0
            for col_name in table_df.columns:
                if any([pattern.match(col_name) for pattern in patterns]):
                    non_semantic_num += 1
            if non_semantic_num >= table_df.shape[1] / 2:
                shutil.copy(file_path, os.path.join(tar_non_semantic, csv_file))
            else:
                shutil.copy(file_path, os.path.join(tar_semantic, csv_file))
                
def prepare_data_file(src_directory_list, tar_file_name):
    semantic = []
    non_semantic = []
    
    # define the patterns of non_semantic column names
    patterns = [
        re.compile(r'^[A-Za-z]+\d+$'), 
        re.compile(r'^\d+$'), 
        re.compile(r'^[a-zA-Z]{1,2}$'), 
        re.compile(r'^[a-zA-Z]+_\d+$'), 
    ]
    
    for src_directory in src_directory_list:
        # iterate over all files in the directory
        for src_file in os.listdir(src_directory):
            if os.path.isdir(os.path.join(src_directory, src_file)):
                for csv_file in tqdm(os.listdir(os.path.join(src_directory, src_file))):
                    if csv_file.endswith('.csv'):
                        try:
                            file_path = os.path.join(src_directory, src_file, csv_file)
                            table_df = pd.read_csv(file_path, encoding='utf-8', low_memory=False, nrows=500)
                            if table_df.shape[0] < 10 or table_df.shape[1] < 3:
                                continue
                            non_semantic_num = 0
                            for col_name in table_df.columns:
                                if any([pattern.match(col_name) for pattern in patterns]):
                                    non_semantic_num += 1
                            if non_semantic_num >= table_df.shape[1] / 2:
                                non_semantic.append(file_path)
                            else:
                                semantic.append(file_path)
                        except:
                            print(f"Error: {file_path}")
    
    # save the file paths to the disk
    with open(f'data/{tar_file_name}.npz', 'wb') as f:
        np.savez(f, semantic=semantic, non_semantic=non_semantic)
                
        