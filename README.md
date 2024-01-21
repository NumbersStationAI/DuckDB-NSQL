# DuckDB-NSQL
Numbers Station Text to SQL model for DuckDB.

NSQL is a family of autoregressive open-source foundational models (FMs) that are particularly designed for SQL generation tasks. We are thrilled to introduce DuckDB-NSQL in this repository, an FM tailored for local DuckDB SQL analytics tasks. All model weights can be found on HuggingFace.

| Model Name                            | Size | Link                                                           |
| --------------------------------------| ---- | -------------------------------------------------------------- |
| motherduckdb/DuckDB-NSQL-7B-v0.1      | 7B   | [link](https://huggingface.co/motherduckdb/DuckDB-NSQL-7B-v0.1) |
| motherduckdb/DuckDB-NSQL-7B-v0.1-GGUF | 7B   | [link](https://huggingface.co/motherduckdb/DuckDB-NSQL-7B-v0.1-GGUF)|

## Setup
To install all the necessary dependencies, please run
```
pip install -r requirements.txt
```

## Usage
Please refer to the examples in the `examples/` folder to learn how to connect to a local DuckDB and directly query your data. A simple notebook is provided in the `examples/` directory for reference.

To host the model with llama.cpp, please execute the following:

```python
# Import necessary modules
from llama_cpp import Llama
from wurlitzer import pipes

# Set up client with model path and context size
with pipes() as (out, err):
    client = Llama(
        model_path="DuckDB-NSQL-7B-v0.1-q8_0.gguf",
        n_ctx=2048,
    )
```

To load the DuckDB database and query against it, please execute the following:

```python
# Import necessary modules
import duckdb
from utils import generate_sql

# Connect to DuckDB database
con = duckdb.connect("nyc.duckdb")

# Sample question for SQL generation
question = "alter taxi table and add struct column with name test and keys a:int, b:double"

# Generate SQL, check validity, and print
sql = generate_sql(question, con, client)
print(sql)
```

## Training Data

The training data for this model consists of two parts: 1) 200k synthetically generated DuckDB SQL queries, based on the DuckDB v.0.9.2 documentation, and 2) labeled text-to-SQL pairs from [NSText2SQL](https://huggingface.co/datasets/NumbersStation/NSText2SQL) transpiled to DuckDB SQL using [sqlglot](https://github.com/tobymao/sqlglot).

## Evaluate the benchmark

Please refer to the `eval/` folder to check the details for evaluating the model against our proposed DuckDB benchmark.

## Acknowledgement

We would like to express our appreciation to all authors of the evaluation scripts. Their work made this project possible.
