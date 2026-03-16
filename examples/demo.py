# Import necessary modules
from llama_cpp import Llama
from wurlitzer import pipes
import gradio as gr
import duckdb
from utils import generate_sql

# Set up client with model path and context size
with pipes() as (out, err):
    client = Llama(
        model_path="/data/workspace/DuckDB-NSQL-7B-v0.1/DuckDB-NSQL-7B-v0.1-q8_0.gguf",
        n_ctx=2048,
    )


# Connect to DuckDB database
con = duckdb.connect("nyc.duckdb")

# This function will be the interface for Gradio
def generate_sql_from_natural_language(question):
    # Sample question for SQL generation
    #question = "alter taxi table and add struct column with name test and keys a:int, b:double"

    # Generate SQL, check validity, and print
    sql = generate_sql(question, con, client)
    print(sql)
    return sql

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_sql_from_natural_language,
    inputs=[gr.Textbox(lines=2, placeholder="Enter your SQL request in natural language here..."),],
    outputs=["text"],
    title="Natural Language to SQL",
    description="Enter a SQL request in natural language to generate the corresponding SQL query."
)

# Launch the application
iface.launch(server_name="0.0.0.0", server_port=7860)

