import sqlite3
import csv
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
DATASETS_PATH = os.environ["DATASETS_PATH"]
MODELS_PATH = os.environ["MODELS_PATH"]

database_path = f'{DATASETS_PATH}/spider/database'
database_files = os.listdir(database_path)
database_csv_path = f'{DATASETS_PATH}/spider/database_csv'

for database_file in tqdm(database_files):
    database_file_path = os.path.join(database_path, database_file)
    try:
        sqlite_files = os.listdir(database_file_path)
        for sqlite_file in sqlite_files:
            if not sqlite_file.endswith('.sqlite'):
                continue
            sqlite_file_path = os.path.join(database_file_path, sqlite_file)
            conn = sqlite3.connect(sqlite_file_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM sqlite_master WHERE type="table"')
            tables = cursor.fetchall()
            if database_file not in os.listdir(database_csv_path):
                os.makedirs(os.path.join(database_csv_path, database_file))
            dir_path = os.path.join(database_csv_path, database_file)
            for table in tables:
                table_name = table[1]
                cursor.execute(f'SELECT * FROM {table_name}')
                results = cursor.fetchall()
                csv_file = os.path.join(dir_path, f'{table_name}.csv')
                with open(csv_file, 'w', encoding='utf-8', newline='') as file:
                    csv_writer = csv.writer(file)
                    column_names = [description[0] for description in cursor.description]
                    csv_writer.writerow(column_names)
                    csv_writer.writerows(results)
            cursor.close()
            conn.close()
    except Exception as e:
        print(f'Error: {database_file}, {e.with_traceback(None)}')
