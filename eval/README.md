This folder contains the suite for evaluating the DuckDB-Text2SQL model.

Please install the dependencies listed in the requirements.txt file located in the parent folder.

## Setup
To evaluate against the benchmark dataset, you need to prepare the evaluation script using this benchmark.

```
mkdir metrics
cd metrics
git clone git@github.com:ElementAI/test-suite-sql-eval.git test_suite_sql_eval
cd ..
```

You need to add a new remote to evaluate against duckdb in the test-suite-sql-eval folder. And check the latest duckdb-only branch (640a12975abf75a94e917caca149d56dbc6bcdd7).

```
git remote add till https://github.com/tdoehmen/test-suite-sql-eval.git
git fetch till
git checkout till/duckdb-only
```

Next, prepare the docs for retrieval.
```
mkdir docs
cd docs
git clone https://github.com/duckdb/duckdb-web.git
cd ..
```

#### Dataset
The benchmark dataset is located in the `data/` folder and includes all databases (`data/databases`), table schemas (`data/tables.json`), and examples (`data/dev.json`).

#### Eval
Start a manifest session with the model you want to evaluate.

```bash
python -m manifest.api.app \
    --model_type huggingface \
    --model_generation_type text-generation \
    --model_name_or_path motherduckdb/DuckDB-NSQL-7B-v0.1 \
    --fp16 \
    --device 0
```

Then, from the `DuckDB-NSQL` main folder, run:

```bash
python eval/predict.py \
    predict \
    eval/data/dev.json \
    eval/data/tables.json \
    --output-dir output/ \
    --stop-tokens ';' \
    --stop-tokens '--' \
    --stop-tokens '```' \
    --stop-tokens '###' \
    --overwrite-manifest \
    --manifest-client huggingface \
    --manifest-connection http://localhost:5000 \
    --prompt-format duckdbinst
```
This will format the prompt using the duckdbinst style.

To evaluate the prediction, first run the following in a Python shell:

```python
try:
    import duckdb

    con = duckdb.connect()
    con.install_extension("httpfs")
    con.load_extension("httpfs")
except Exception as e:
    print(f"Error loading duckdb extensions: {e}")
```

Then, run the evaluation script:

```bash
python eval/evaluate.py \
    evaluate \
    --gold eval/data/dev.json \
    --db eval/data/databases/ \
    --tables eval/data/tables.json \
    --output-dir output/ \
    --pred [PREDICITON_FILE]
```

To view the output, all the information is located in the prediction file in the [output-dir]. Here, `query` is gold and `pred` is predicted.
