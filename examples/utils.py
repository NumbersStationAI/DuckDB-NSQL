import sys
import re
from typing import Any
import subprocess
from wurlitzer import pipes
from duckdb import DuckDBPyConnection

PROMPT_TEMPLATE = """### Instruction:\n{instruction}\n\n### Input:\n{input}\n### Question:\n{question}\n\n### Response (use duckdb shorthand if possible):\n"""
INSTRUCTION_TEMPLATE = """Your task is to generate valid duckdb SQL to answer the following question{has_schema}"""  # noqa: E501
ERROR_MESSAGE = "Quack! Much to our regret, SQL generation has gone a tad duck-side-down.\nThe model is currently not capable of crafting the desired SQL. \nSorry my duck friend.\n\nIf the question is about your own database, make sure to set the correct schema.\n\n```sql\n{sql_query}\n```\n\n```sql\n{error_msg}\n```"


def get_schema(connection: DuckDBPyConnection) -> str:
    """Get schema from DuckDB connection."""
    tables = []
    information_schema = connection.execute(
        "SELECT * FROM information_schema.tables"
    ).fetchdf()
    for table_name in information_schema["table_name"].unique():
        table_df = connection.execute(
            f"SELECT * FROM information_schema.columns WHERE table_name = '{table_name}'"
        ).fetchdf()
        columns = []
        for _, row in table_df.iterrows():
            col_name = row["column_name"]
            col_dtype = row["data_type"]
            columns.append(f"{col_name} {col_dtype}")
        column_str = ",\n    ".join(columns)
        table = f"CREATE TABLE {table_name} (\n    {column_str}\n);"
        tables.append(table)
    return "\n\n".join(tables)


def generate_prompt(question: str, schema: str) -> str:
    """Generate prompt."""
    input = ""
    if schema:
        # Lowercase types inside each CREATE TABLE (...) statement
        for create_table in re.findall(
            r"CREATE TABLE [^(]+\((.*?)\);", schema, flags=re.DOTALL | re.MULTILINE
        ):
            for create_col in re.findall(r"(\w+) (\w+)", create_table):
                schema = schema.replace(
                    f"{create_col[0]} {create_col[1]}",
                    f"{create_col[0]} {create_col[1].lower()}",
                )
        input = """Here is the database schema that the SQL query will run on:\n{schema}\n""".format(  # noqa: E501
            schema=schema
        )
    prompt = PROMPT_TEMPLATE.format(
        instruction=INSTRUCTION_TEMPLATE.format(
            has_schema="." if schema == "" else ", given a duckdb database schema."
        ),
        input=input,
        question=question,
    )
    return prompt


def generate_sql(
    question: str,
    connection: DuckDBPyConnection,
    llama: Any,
    max_tokens: int = 300,
) -> [str, bool, str]:
    schema = get_schema(connection)
    prompt = generate_prompt(question, schema)

    with pipes() as (out, err):
        res = llama(prompt, temperature=0.1, max_tokens=max_tokens)
    sql_query = res["choices"][0]["text"]
    
    is_valid, error_msg = validate_sql(sql_query, schema)

    if is_valid:
        print(sql_query)
    else:
        print("!!!Invalid SQL detected!!!")
        print(sql_query)
        print(error_msg)
    
    return sql_query


def validate_sql(query, schema):
    try:
        # Define subprocess
        process = subprocess.Popen(
            [sys.executable, './validate_sql.py', query, schema],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Get output and potential parser, and binder error message
        stdout, stderr = process.communicate(timeout=0.5)
        if stderr:
            error_message = stderr.decode('utf8').split("\n")
            # skip traceback
            if len(error_message) > 3:
                error_message = "\n".join(error_message[3:])
            return False, error_message
        return True, ""
    except subprocess.TimeoutExpired:
        process.kill()
        # timeout reached, so parsing and binding was very likely successful
        return True, ""
