import sys
sys.path.append('../code')
import os
import re
from config import EVALUATIION_FILE
def get_db(type):
    # You should fill in the correct database name for each type
    if type == 'test':
        return '...'
    if type == 'dev':
        return '...'
    if type == 'dk':
        return '...'
    if type == 'r_ex':
        return '...'
    if type == 'r_ts':
        return '...'
    

def get_table(type):
    # You should fill in the correct tables.json file path for each type
    if type == 'dk':
        return 'xxx/tables.json'
    if type == 'r_ex' or type == 'r_ts':
        return 'xxx/tables.json'
    if type == 'test':
        return 'xxx/tables.json'
    return 'xxx/tables.json'
    
def process_command(model_output_path, result_path, command_save_path, type):#

    model_output_path = os.path.abspath(model_output_path)
    result_path = os.path.abspath(result_path)

    COMMAND = f'''
    python {EVALUATIION_FILE} --path {model_output_path} --etype exec --db {get_db(type)} > {result_path}_exec
    '''.strip()
    COMMAND += f'''\npython {EVALUATIION_FILE} --path {model_output_path} --etype match --db {get_db(type)} --table {get_table(type)} > {result_path}_match'''

    print(COMMAND)
    with open(command_save_path, 'w') as f:
        f.write(COMMAND)
    MAX_RETRY = 80
    for i in range(MAX_RETRY):
        
        if os.system(COMMAND) == 0:
            break
    else:
        print(f"Failed to execute the command: {COMMAND}")
        return
    
    
import sqlparse
from sqlparse.sql import Identifier, IdentifierList
from sqlparse.tokens import Keyword, DML

def replace_names_in_query(query, table_mapping, column_mapping):
    
    parsed = sqlparse.parse(query)

    
    for statement in parsed:
        if statement.get_type() != 'UNKNOWN':  
            replace_tokens(statement.tokens, table_mapping, column_mapping)

    
    return str(parsed[0])



def replace_tokens(tokens, table_mapping, column_mapping):
    for token in tokens:
        if token.ttype is DML or token.ttype is Keyword:
            continue  

        if isinstance(token, Identifier):  
            replace_identifier(token, table_mapping, column_mapping)
        elif isinstance(token, IdentifierList): 
            for identifier in token.get_identifiers():
                replace_identifier(identifier, table_mapping, column_mapping)
        elif token.is_group: 
            replace_tokens(token.tokens, table_mapping, column_mapping)



def replace_identifier(identifier, table_mapping, column_mapping):
   
    try:
        column_name = identifier.get_real_name()
        table_name = identifier.get_parent_name() 
    except:
        # print('fail at token:', identifier)
        return
    
    if table_name and table_name in table_mapping:
        table_name = table_mapping[table_name]

    
    if column_name in column_mapping:
        
        new_name = f"{table_name}.{column_mapping[column_name]}" if table_name else column_mapping[column_name]
        identifier.tokens = [sqlparse.sql.Token(None, new_name)]
    elif table_name and table_name in table_mapping:
        
        new_name = f"{table_mapping[table_name]}.{column_name}"
        identifier.tokens = [sqlparse.sql.Token(None, new_name)]

def replace_names_force(query, table_mapping, column_mapping):
    for k, v in table_mapping.items():
        query = query.replace(k, v)
    for k, v in column_mapping.items():
        query = query.replace(k, v)
    return query
        
def parse_sql_from_output(cur_output):
    import re
    if '```sql' in cur_output:
        # use re to extract the SQL query and return it
        match = re.search(r'```sql(.*?)```', cur_output)
        if match:
            return match.group(1).replace('\n', ' ').replace('\t', ' ').strip()

    elif '```' in cur_output:
        match = re.search(r'```(.*?)```', cur_output)
        if match:
            return match.group(1).replace('\n', ' ').replace('\t', ' ').strip()
    
    if 'SELECT' in cur_output:
        ret = cur_output[cur_output.index('SELECT'):]
        if '```' in ret:
            ret = ret[:ret.index('```')]
        if ';' in ret:
            ret = ret[:ret.index(';')]
        return ret.replace('\n', ' ').replace('\t', ' ')
    
    return cur_output

