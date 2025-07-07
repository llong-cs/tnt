import pandas as pd
from config import INSERT_START_TOKEN, INSERT_END_TOKEN, INSERT_EMBS_TOKEN
import random


def get_df_category_desc(query, df):
    return {}

def dataframe_info_novalue(df:pd.DataFrame, df_name:str, primary_key = None, comments=None, selected_values = {}, lower = False):

    df_info_template_simple = "table {df_name}, columns = [{desc_info}]"
        
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    example_values = []
    for col in df.columns:
        col_value = df[col].dropna().unique().tolist()
        assert(type(col_value) == list)
        if col in selected_values.keys():
            col_value = [v for v in col_value if v != selected_values[col]]
            
        d_type = str(df[col].dtype)
        
        if len(col_value) > 3:
            if ("float" in d_type):
                col_value = col_value[0:2]
            else:
                col_value = col_value[0:3]
            col_value.append("...")
        
        col_value_limit = [s if not isinstance(s, str) or len(s) <= 80 else s[:80] + "...." for s in col_value]
        example_values.append(col_value_limit)
    info_df['Example Values'] = example_values

    if comments is not None:
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    desc_info_lines = []
    for key, value in desc_info_dict.items():
        if lower:
            key = key.lower()

        data_type = value["Data Type"]
        
        contains_nan = value["Contains NaN"]
        if contains_nan:
            contains_nan_str = "contains NaN, "
        else:
            contains_nan_str = ""
        
        is_unique = value["Is Unique"]
        if is_unique:
            unique_str = "is unique, "
        else:
            unique_str = ""
            # unique_str = "is not unique, "
        
        examples = str(value["Example Values"])
        if examples[0] == '[':
            examples = examples[1:-1]
        
        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        new_key = f'"{key}"' if ' ' in key else key
        dil = f"{df_name}.{new_key}({INSERT_EMBS_TOKEN}|{data_type}{'|primary key' if key == primary_key else ''})"
        desc_info_lines.append(dil)

    desc_info = ", ".join(desc_info_lines)

    desc_info = desc_info.replace(", '...')", ", ...)")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info

def dataframe_info(df:pd.DataFrame, df_name:str, primary_key = None, comments=None, selected_values = {}, lower = False):

    df_info_template_simple = "table {df_name}, columns = [{desc_info}]"
        
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    example_values = []
    for col in df.columns:
        col_value = df[col].dropna().unique().tolist()
        assert(type(col_value) == list)
        if col in selected_values.keys():
            col_value = [v for v in col_value if v != selected_values[col]]
            
        d_type = str(df[col].dtype)
        
        if len(col_value) > 3:
            if ("float" in d_type):
                col_value = col_value[0:2]
            else:
                col_value = col_value[0:3]
            col_value.append("...")
        
        col_value_limit = [s if not isinstance(s, str) or len(s) <= 80 else s[:80] + "...." for s in col_value]
        example_values.append(col_value_limit)
    info_df['Example Values'] = example_values

    if comments is not None:
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    desc_info_lines = []
    for key, value in desc_info_dict.items():
        if lower:
            key = key.lower()

        data_type = value["Data Type"]
        
        contains_nan = value["Contains NaN"]
        if contains_nan:
            contains_nan_str = "contains NaN, "
        else:
            contains_nan_str = ""
        
        is_unique = value["Is Unique"]
        if is_unique:
            unique_str = "is unique, "
        else:
            unique_str = ""
        
        examples = str(value["Example Values"])
        if examples[0] == '[':
            examples = examples[1:-1]
        
        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        new_key = f'"{key}"' if ' ' in key else key
        dil = f"{df_name}.{new_key}({INSERT_EMBS_TOKEN}|{data_type}|{'primary key|' if key == primary_key else ''}values: {examples})"
        desc_info_lines.append(dil)

    desc_info = ", ".join(desc_info_lines)

    desc_info = desc_info.replace(", '...')", ", ...)")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info

def dataframe_info_valuecount(df:pd.DataFrame, df_name:str, primary_key = None, comments=None, selected_values = {}, lower = False, valuecount = 3):
    df_info_template_simple = "table {df_name}, columns = [{desc_info}]"
        
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    example_values = []
    for col in df.columns:
        col_value = df[col].dropna().unique().tolist()
        assert(type(col_value) == list)
        if col in selected_values.keys():
            col_value = [v for v in col_value if v != selected_values[col]]
            
        d_type = str(df[col].dtype)
        
        if len(col_value) > valuecount:
            col_value = col_value[0:valuecount]
            col_value.append("...")
        
        col_value_limit = [s if not isinstance(s, str) or len(s) <= 80 else s[:80] + "...." for s in col_value]
        example_values.append(col_value_limit)
    info_df['Example Values'] = example_values

    if comments is not None:
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    desc_info_lines = []
    for key, value in desc_info_dict.items():
        if lower:
            key = key.lower()

        data_type = value["Data Type"]
        
        contains_nan = value["Contains NaN"]
        if contains_nan:
            contains_nan_str = "contains NaN, "
        else:
            contains_nan_str = ""
        
        is_unique = value["Is Unique"]
        if is_unique:
            unique_str = "is unique, "
        else:
            unique_str = ""
        
        examples = str(value["Example Values"])
        if examples[0] == '[':
            examples = examples[1:-1]
        
        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        new_key = f'"{key}"' if ' ' in key else key
        value_str = f'|values: {examples}' if valuecount > 0 else ''
        dil = f"{df_name}.{new_key}({INSERT_EMBS_TOKEN}|{data_type}{'|primary key' if key == primary_key else ''}{value_str})"
        desc_info_lines.append(dil)

    desc_info = ", ".join(desc_info_lines)

    desc_info = desc_info.replace(", '...')", ", ...)")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info

def find_primary_key(df:pd.DataFrame):
    try:
        for col in df.columns:
            if (df[col].nunique() == len(df)) and df[col].notnull().all():
                return col
    except:
        pass
    return ''

def primary_key_list(csv_paths: list):
    return [find_primary_key(pd.read_csv(csv_paths[i], encoding="utf-8", low_memory=False)) for i in range(len(csv_paths))]


def dataframe_info_list(*, csv_paths: list, df_names: list, primary_keys: list, comments=None, values = False):
    info_func = dataframe_info if values else dataframe_info_novalue
    if df_names is None:
        df_names = [[] for _ in range(len(csv_paths))]
    return '\n'.join([info_func(df = pd.read_csv(csv_paths[i], nrows=500), df_name = df_names[i], primary_key = primary_keys[i], comments = comments) for i in range(len(csv_paths))])
def dataframe_info_list_valuecount(*, csv_paths: list, df_names: list, primary_keys: list, comments=None, valuecount = 3):
    info_func = dataframe_info_valuecount
    if df_names is None:
        df_names = [[] for _ in range(len(csv_paths))]
    return '\n'.join([info_func(df = pd.read_csv(csv_paths[i], nrows=500), df_name = df_names[i], primary_key = primary_keys[i], comments = comments, valuecount=valuecount) for i in range(len(csv_paths))])

def get_text2sql_prompt(fixed = False):
# parameters: df_info, foreign_keys, question
    templates = [
        "Answer the following question with the corresponding sqlite SQL query only and with no explanation.",
        "Provide only the SQLite SQL query as a response to the following question, without any further explanation.",
        "For the next question, answer solely with the appropriate SQLite SQL query, without additional explanations.",
        "Respond to the following question exclusively the corresponding SQLite SQL query, omitting any explanation.",
        "Use only an SQLite SQL query to answer the upcoming question. Do not include any further details or explanations.",
        "Give the SQLite SQL query as the sole answer to the next question, without any explanation.",
        "Answer the next question by providing only the SQLite SQL query, and nothing else.",
        "Your response to the following question should consist solely of the corresponding SQLite SQL query, with no explanation.",
        "Please provide just the SQLite SQL query in response to the next question, without any further elaboration.",
        "Respond to the next question using only the relevant SQLite SQL query, with no additional comments.",
        "Provide only the necessary SQLite SQL query as your answer to the following question, with no further explanation.",
    ]
    choice = "Given the following database schema:\n{df_info}\n" + (random.choice(templates) if not fixed else templates[0]) + "\nQuestion: {question}"
    return choice.strip()


def build_text2sql_prompt(*, question:str, csv_paths: list, df_names: list, primary_keys = None, foreign_keys = None, answer = None, fixed = False):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    if foreign_keys != None and len(foreign_keys) > 0:
        df_info += "\nForeign keys: " + str(foreign_keys) + "\n"
    instruction = get_text2sql_prompt(fixed=fixed).format(df_info=df_info, question=question)
    if answer is None:
        return instruction
    # if answer is wrapped in ```sql, remove it
    if answer.startswith('```sql'):
        answer = answer[6:]
    if answer.endswith('```'):
        answer = answer[:-3]
    answer = answer.strip()
    return instruction, answer

def build_text2sql_prompt_valuecount(*, question:str, csv_paths: list, df_names: list, primary_keys = None, foreign_keys = None, answer = None, fixed = False, valuecount = 3):
    df_info = dataframe_info_list_valuecount(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, valuecount=valuecount)
    if foreign_keys != None and len(foreign_keys) > 0:
        df_info += "\nForeign keys: " + str(foreign_keys) + "\n"
    instruction = get_text2sql_prompt(fixed=fixed).format(df_info=df_info, question=question)
    if answer is None:
        return instruction
    # if answer is wrapped in ```sql, remove it
    if answer.startswith('```sql'):
        answer = answer[6:]
    if answer.endswith('```'):
        answer = answer[:-3]
    answer = answer.strip()
    return instruction, answer
    
def get_column_prediction_prompt(multi_column = False):
    templates = [
        "Suppose there is a row: {row}\nWhich column is the cell value '{cell_value}' most likely from? Answer with the corresponding column name only.",
        "Consider a row containing: {row}. Which column is the cell value '{cell_value}' most likely associated with? Respond with only the column name.",
        "Given a row with the following values: {row}, determine which column the cell value '{cell_value}' is most likely from. Answer with just the column name.",
        "Imagine a row formatted as: {row}. Identify the column where the cell value '{cell_value}' is most likely found. Provide only the column name.",
        "From the row {row}, which column is the cell value '{cell_value}' most likely located in? Please provide only the column name in your response.",
        "Given the data row: {row}, indicate which column the cell value '{cell_value}' most likely belongs to. Your answer should be the column name only.",
        "In the row {row}, to which column is the cell value '{cell_value}' most likely related? Respond solely with the column name.",
        "Consider a data row: {row}. Identify the column most likely containing the cell value '{cell_value}'. Provide only the column name.",
        "If you have a row structured as {row}, which column is the cell value '{cell_value}' most likely part of? Answer with the column name only.",
        "Looking at a row like this: {row}, which column is the cell value '{cell_value}' most likely associated with? Provide just the column name.",
        "For the row {row}, determine which column is the most likely source of the cell value '{cell_value}'. Your response should be the column name only.",
    ]
    templates_multi = [
        "Suppose there is a row: {row}\nWhich columns are the cell values {cell_value} most likely from? Answer with the corresponding column names only.",
        "Consider a row containing: {row}. Which columns are the cell values {cell_value} most likely associated with? Respond with only the column names.",
        "Given a row with the following values: {row}, determine which columns the cell values {cell_value} are most likely from. Answer with just the column names.",
        "Imagine a row formatted as: {row}. Identify the columns where the cell values {cell_value} are most likely found. Provide only the column names.",
        "From the row {row}, which columns are the cell values {cell_value} most likely located in? Please provide only the column names in your response.",
        "Given the data row: {row}, indicate which columns the cell values {cell_value} most likely belong to. Your answer should be the column names only.",
        "In the row {row}, to which columns are the cell values {cell_value} most likely related? Respond solely with the column names.",
        "Consider a data row: {row}. Identify the columns most likely containing the cell values {cell_value}. Provide only the column names.",
        "If you have a row structured as {row}, which columns are the cell values {cell_value} most likely part of? Answer with the column names only.",
        "Looking at a row like this: {row}, which columns are the cell values {cell_value} most likely associated with? Provide just the column names.",
        "For the row {row}, determine which columns are the most likely sources of the cell values {cell_value}. Your response should be the column names only.",
    ] # cell_value is a list, so there's no need to add quotes around it
    if not multi_column:
        ret = "Given the following database schema:\n{df_info}\n" + random.choice(templates)
    else:
        ret = "Given the following database schema:\n{df_info}\n" + random.choice(templates_multi) + '\nAnswer in the format: column1; column2; ...'
    return ret

def build_column_prediction_prompt(*, csv_paths: list, df_names: list, row:str, cell_value:str, primary_keys = None, multi_column = False):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    prompt = get_column_prediction_prompt(multi_column).format(df_info=df_info, row=row, cell_value=cell_value)
    return prompt


def get_cell_prediction_prompt():
    template = [
        "From the list {cell_value}, which value is most likely associated with column '{col_name}'? Provide only the cell value in your response.",
        "In the sequence {cell_value}, identify the value that most likely belongs to column '{col_name}'. Answer with just the cell value.",
        "Which value in {cell_value} is most likely from the '{col_name}' column? Respond with the specific cell value only.",
        "From the values {cell_value}, which one is most likely to be in column '{col_name}'? Please answer with only the cell value.",
        "Given the row {cell_value}, which cell value is most likely linked to column '{col_name}'? Provide only the corresponding value.",
        "Which of the following values {cell_value} is most likely to come from column '{col_name}'? Respond solely with the cell value.",
        "In the list {cell_value}, identify which value most likely pertains to the column '{col_name}'. Answer with just the value name.",
        "From {cell_value}, which value is most likely aligned with column '{col_name}'? Provide only the value in your response.",
        "Looking at the values {cell_value}, which one most likely corresponds to column '{col_name}'? Answer with the corresponding cell value only.",
        "Among the following cell values {cell_value}, which is most likely associated with column '{col_name}'? Respond with only the cell value.",
    ]
    return "Given the following database schema:\n{df_info}\n" + random.choice(template)
def build_cell_prediction_prompt(*, csv_paths: list, df_names: list, cell_value:str, col_name:str, primary_keys = None):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    prompt = "Given the following database schema:\n{df_info}\n" + get_cell_prediction_prompt().format(df_info=df_info, cell_value=cell_value, col_name=col_name)
    return prompt


def get_recall_prompt():
    templates = [
        "What are the relevant tables and columns related to the question: '{question}'?",
        "Which tables and columns are associated with the question: '{question}'?",
        "Identify the tables and columns relevant to the query: '{question}'.",
        "For the question: '{question}', which tables and columns are involved?",
        "Determine the tables and columns related to the question: '{question}'.",
        "What are the pertinent tables and columns for the question: '{question}'?",
        "List the tables and columns connected to the question: '{question}'.",
        "Which database tables and columns pertain to the question: '{question}'?",
        "Find the tables and columns that are linked to the question: '{question}'.",
        "What tables and columns should be referenced for the question: '{question}'?",
        "Identify the specific tables and columns associated with the query: '{question}'.",
    ]
    choice = "Given the following database schema:\n{df_info}\n" + random.choice(templates) + '\nAnswer in the format: table1.column1, table1.column2, table2.column1, ...'
    return choice.strip()

def build_recall_prompt(*, question:str, csv_paths: list, df_names: list, primary_keys = None, foreign_keys = None):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    if foreign_keys is not None and len(foreign_keys) > 0:
        df_info += "\nForeign keys: " + str(foreign_keys) + "\n"
    instruction = get_recall_prompt().format(df_info=df_info, question=question)
    if len(csv_paths) == 1:
        instruction = instruction.replace('tables and columns', 'columns')
        instruction = instruction.replace('table1.column1, table1.column2, table2.column1, ...', 'table1.column1, table1.column2, ...')
        
    return instruction

def get_question_raise_prompt():
    templates = [
        "Please propose a relevant question which can be answered with '{answer}'.",
        "Suggest a suitable question that could be answered with '{answer}'.",
        "Formulate a question that would result in the answer '{answer}'.",
        "Provide a question that matches the answer '{answer}'.",
        "Come up with a question that can be answered by '{answer}'.",
        "Create a question that has '{answer}' as the answer.",
        "Propose a question whose answer is '{answer}'.",
        "What question could be appropriately answered with '{answer}'?",
        "Draft a relevant question that would lead to the answer '{answer}'.",
        "Can you propose a question that would produce '{answer}' as its answer?",
        "Develop a question that would logically have '{answer}' as the response.",
    ]
    choice = "Given the following database schema:\n{df_info}\n" + random.choice(templates)
    return choice.strip()
def build_question_raise_prompt(*, answer:str, csv_paths: list, df_names: list, primary_keys = None, foreign_keys = None):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    if foreign_keys is not None and len(foreign_keys) > 0:
        df_info += "\nForeign keys: " + str(foreign_keys) + "\n"
    instruction = get_question_raise_prompt().format(df_info=df_info, answer=answer)
    return instruction

def get_description_prompt():
    template = [
        "Suppose the table above is displayed on a webpage. Please create a suitable introductory title for it.",
        "Given that the table above is part of a webpage, write an appropriate title for it.",
        "Assume the table above appears on a web page. Please provide a descriptive title for it.",
        "If the table above were on a webpage, what would be a fitting introductory title?",
        "Consider the table above as being on a webpage. Write an introductory title for this table.",
        "Think of the table above as part of a webpage. Create a relevant title for it.",
        "Suppose the table above is included in a webpage. Please suggest an introductory title for it.",
        "If the table above were published on a webpage, what would be an appropriate title?",
        "Imagine the table above appears on a webpage. Please craft a suitable introductory title.",
        "Assuming the table above is featured on a webpage, please write a relevant title for it.",
    ]
    choice = "Given the following database schema:\n{df_info}\n" + random.choice(template)
    return choice.strip()

def build_description_prompt(*, csv_paths: list, df_names: list, primary_keys = None, foreign_keys = None):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    if foreign_keys is not None and len(foreign_keys) > 0:
        df_info += "\nForeign keys: " + str(foreign_keys) + "\n"
    instruction = get_description_prompt().format(df_info=df_info)
    return instruction

def get_pythongen_prompt():
    template = [
        "Suppose there is a corresponding pandas DataFrame <df>, write the Python code to address the following question.",
        "Assume there is a corresponding pandas DataFrame <df>. Write the Python code to solve the following question.",
        "Given a corresponding pandas DataFrame <df>, provide the Python code to answer the question below.",
        "Imagine a corresponding pandas DataFrame <df> exists. Write the Python code needed to address the following question.",
        "Suppose you have a corresponding pandas DataFrame <df>. Write the Python code that answers the question provided.",
        "Consider a corresponding pandas DataFrame <df>. Please write the Python code to tackle the question below.",
        "Assume a corresponding pandas DataFrame <df> is available. Write the Python code to handle the following query.",
        "If there is a corresponding pandas DataFrame <df>, write the necessary Python code to address the question below.",
        "Envision a corresponding pandas DataFrame <df>. Write the Python code to resolve the following question.",
        "Assume the existence of a corresponding pandas DataFrame <df>. Write the Python code to respond to the question below.",
        "Suppose you have access to a pandas DataFrame <df>. Provide the Python code that addresses the following question.",
    ]
    return "Given the following database schema:\n{table_infos}\n" + random.choice(template) + "\nWrap your answer in ```python ```.\nQuestion: {question}"


def build_pythongen_prompt(*, question:str, csv_paths: list, df_names: list, primary_keys = None):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    instruction = get_pythongen_prompt().format(table_infos=df_info, question=question)
    return instruction

def get_rowsentence_prompt():
    template = [
        "Please summarize the following rows in one sentence (the columns may be shuffled). {rows}",
        "Summarize the information in the following rows into a single sentence (note that the columns may be rearranged). {rows}",
        "Condense the data in the following rows into one concise sentence (columns might be in a different order). {rows}",
        "Create a one-sentence summary of the following rows (columns may appear in different sequences). {rows}",
        "Please provide a single-sentence summary of the following rows (columns could be shuffled). {rows}",
        "Summarize the content of the following rows in one sentence, keeping in mind that the columns may be shuffled. {rows}",
        "Write a one-sentence summary for the rows below (be aware that columns might be in a different order). {rows}",
        "Reduce the information in the following rows to one sentence (columns may not be in the original order). {rows}",
        "Please distill the following rows into one sentence, noting that columns may be in various orders. {rows}",
        "In one sentence, summarize the information from the following rows (columns might be arranged differently). {rows}",
        "Provide a single-sentence summary of the following rows (columns may be shuffled). {rows}",
    ]
    return "Given the following database schema:\n{df_info}\n" + random.choice(template)
def build_rowsentence_prompt(*, csv_paths: list, df_names: list, rows:str, primary_keys = None):
    df_info = dataframe_info_list(csv_paths = csv_paths, df_names = df_names, primary_keys = primary_keys, values = False)
    prompt = get_rowsentence_prompt().format(df_info=df_info, rows=rows)
    return prompt

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