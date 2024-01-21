"""Run dataset on text2sql zazu experiment.

See README.md for more details.
"""
import datetime
import json
import multiprocessing
import random
import re
from pathlib import Path

import click
import numpy as np
from constants import PROMPT_FORMATTERS
from loaders import DefaultLoader
from get_manifest import get_manifest
from manifest import Manifest
from prompt_formatters import RajkumarFormatter
from rich.console import Console
from schema import Table, TextToSQLModelResponse, TextToSQLParams
from text_to_sql import instruction_to_sql, instruction_to_sql_list
from doc_retriever import (
    load_documentation,
    split_documents,
    embed_documents,
    query_docs,
)
from tqdm import tqdm
from transformers import AutoTokenizer

console = Console(soft_wrap=True)


def generate_sql(
    manifest: Manifest,
    text_to_sql_in: list[TextToSQLParams],
    retrieved_docs: list[list[str]],
    prompt_formatter: RajkumarFormatter,
    stop_tokens: list[str] | None = None,
    overwrite_manifest: bool = False,
    max_tokens: int = 300,
    temperature: float = 0.01,
    num_beams: int = 2,
    parallel: bool = False,
) -> list[tuple[str, TextToSQLModelResponse]]:
    """Call our text2sql function with manifest of our choice."""
    if parallel:
        instruction_to_sql_resps: list[
            TextToSQLModelResponse
        ] = instruction_to_sql_list(
            params=text_to_sql_in,
            extra_context=retrieved_docs,
            manifest=manifest,
            prompt_formatter=prompt_formatter,
            overwrite_manifest=overwrite_manifest,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_tokens,
            num_beams=num_beams,
        )
    else:
        instruction_to_sql_resps = [
            instruction_to_sql(
                params=_text_to_sql_in,
                extra_context=_retrieved_docs,
                manifest=manifest,
                prompt_formatter=prompt_formatter,
                overwrite_manifest=overwrite_manifest,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_tokens,
                num_beams=num_beams,
            )
            for _retrieved_docs, _text_to_sql_in in tqdm(
                zip(retrieved_docs, text_to_sql_in),
                desc="Generating SQL",
                total=len(text_to_sql_in),
                disable=(len(text_to_sql_in) <= 1),
            )
        ]
    assert len(instruction_to_sql_resps) == len(text_to_sql_in)

    sql_statements = []
    for i in range(len(instruction_to_sql_resps)):
        sql_statement = instruction_to_sql_resps[i].output.strip()
        if "<>" in sql_statement:
            sql_statement.replace("<>", "!=")
        # Models sometime train to predict <databasename/schema> | <sql>
        sql_statement = sql_statement.split("|")[-1].strip()
        sql_statements.append(sql_statement)
    return list(zip(sql_statements, instruction_to_sql_resps))


def get_text_to_sql_in(
    input_question: dict, db_to_tables: dict[str, dict[str, Table]]
) -> TextToSQLParams:
    """Format input question for text2sql function."""
    question = input_question["question"]
    db_id = input_question.get("db_id", None)
    if db_id != "none":
        table_params = list(db_to_tables.get(db_id, {}).values())
    else:
        table_params = []
    if len(table_params) == 0:
        console.print(f"[red] WARNING: No tables found for {db_id} [/red]")
    text_to_sql_in = TextToSQLParams(
        instruction=question,
        database=db_id,
        tables=table_params,
    )
    return text_to_sql_in


@click.group()
def cli() -> None:
    """Entrypoint."""
    pass


@cli.command()
@click.argument("dataset-path")
@click.argument("table-meta-path")
@click.option("--output-dir", type=str, default="")
@click.option("--run-name", type=str, default="")
@click.option("--num-run", type=int, default=-1)
@click.option("--num-print", type=int, default=20)
# Format options
@click.option("--prompt-format", type=str, default="spider")
# Prompt options
@click.option("--stop-tokens", type=str, default=[], multiple=True)
@click.option("--max-tokens", type=int, default=200)
@click.option("--temperature", type=float, default=0)
@click.option("--num-beams", type=int, default=-1)  # use whatever is in manifest
@click.option("--max-context-length", type=int, default=-1)
# Docs options
@click.option(
    "--markdown-docs-path",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=True, readable=True, path_type=Path
    ),
    default="eval/docs/duckdb-web/docs/archive/0.9.2/sql",
)
@click.option("--num-retrieved-docs", type=int, default=0)
# Manifest options
@click.option("--manifest-client", type=str, default="openai")
@click.option("--manifest-engine", type=str, default="gpt-4-1106-preview")
@click.option("--manifest-connection", type=str, default="http://localhost:5005")
@click.option("--overwrite-manifest", is_flag=True, default=False)
@click.option("--parallel", is_flag=True, default=False)
def predict(
    dataset_path: str,
    table_meta_path: str,
    output_dir: str,
    run_name: str,
    num_run: int,
    num_print: int,
    prompt_format: str,
    stop_tokens: list[str],
    max_tokens: int,
    temperature: float,
    num_beams: int,
    max_context_length: int,
    markdown_docs_path: Path,
    num_retrieved_docs: int,
    manifest_client: str,
    manifest_engine: str,
    manifest_connection: str,
    overwrite_manifest: bool,
    parallel: bool,
) -> None:
    """Predict SQL.

    Args:
        dataset_path: the dataset path.
        table_meta_path: the json path of the table metadata.
        database_path: the database path for sqlite.
        output_dir: the prediction output directory
        run_name: special prefix to add to filename
        num_run: the number of examples to run
        num_print: the number of examples to print
        prompt_format: the format of the prompt. E.g., "rajkumar"
        stop_tokens: the stop tokens to try
        max_tokens: the max tokens
        temperature: the temperature
        num_beams: the number of beams
        max_context_length: max context length for demonstration truncation (-1 means None)
        markdown_docs_path: path to duckdb sql docs
        num_retrieved_docs: number of docs to retrieve
        manifest_client: the manifest client
        manifest_engine: the manifest engine
        manifest_connection: the manifest connection
    """
    multiprocessing.set_start_method("spawn", force=True)
    random.seed(0)
    np.random.seed(0)
    locals_dict = locals()
    locals_dict["markdown_docs_path"] = str(markdown_docs_path)
    console.print(json.dumps(locals_dict, indent=2))

    data_formatter = DefaultLoader()

    if prompt_format not in PROMPT_FORMATTERS:
        raise ValueError(f"Unknown prompt format {prompt_format}")
    prompt_formatter = PROMPT_FORMATTERS[prompt_format]()

    # load manifest
    manifest = get_manifest(
        manifest_client=manifest_client,
        manifest_connection=manifest_connection,
        manifest_engine=manifest_engine,
    )
    manifest_params = manifest.client_pool.get_current_client().get_model_params()
    console.print(f"Running with {manifest_params} manifest.")
    model_name = manifest_params.get("engine", manifest_params["model_name"])

    if "openai" in manifest_client:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if stop_tokens:
        stop_tokens = [st.strip("'") for st in stop_tokens]
    console.print(f"Stop tokens: {stop_tokens}")

    # Get output filename
    full_dataset_path = Path(dataset_path)
    # Get todays date
    date_today = datetime.datetime.now().strftime("%y-%m-%d")
    if run_name:
        run_name = f"{run_name}_"
    suffix = f"{run_name}{full_dataset_path.stem}_{date_today}.json"  # noqa: E501
    prefix = f"{prompt_format}_{num_retrieved_docs}docs"
    if manifest_client in {"openai", "openaichat", "openaiazure"}:
        middleix = manifest_engine
    elif manifest_client in {"huggingface", "ray"}:
        middleix = Path(manifest_params.get("model_path", "")).name.replace("/", "-")
    elif manifest_client == "toma":
        middleix = manifest_engine.split("/")[-1]
    else:
        raise ValueError(f"Unknown manifest client {manifest_client}")
    output_filename = f"{prefix}_{middleix}_{suffix}"
    console.print(f"Saving to {Path(output_dir) / output_filename}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    console.print("Loading metadata...")
    db_to_tables = data_formatter.load_table_metadata(table_meta_path)

    console.print("Loading data...")
    data = data_formatter.load_data(dataset_path)
    if num_run > 0:
        console.print(f"Running on {min(len(data), num_run)} examples")
        data = data[:num_run]
    original_data = data

    # load the examples
    console.print("Formatting data...")
    num_print = min(num_print, len(data))
    token_lengths = []
    text_to_sql_in = [
        get_text_to_sql_in(input_question, db_to_tables) for input_question in data
    ]

    if num_retrieved_docs > 0:
        console.print("Loading documenration and indexing...")
        retrieved_docs = []
        doc_contents = load_documentation(markdown_docs_path)
        chunked_docs = split_documents(doc_contents)
        embedded_docs, full_embedding_mat = embed_documents(chunked_docs)
        for i in tqdm(range(len(text_to_sql_in)), desc="Retrieving docs"):
            _, retrieved_docs_strings = query_docs(
                text_to_sql_in[i].instruction,
                embedded_docs,
                full_embedding_mat,
                top_n=num_retrieved_docs,
            )
            retrieved_docs.append(retrieved_docs_strings)
    else:
        retrieved_docs = [[] for _ in range(len(text_to_sql_in))]

    for i in range(num_print):
        # Run a few to get some examples to print
        generated_responses = generate_sql(
            manifest=manifest,
            text_to_sql_in=[text_to_sql_in[i]],
            retrieved_docs=[retrieved_docs[i]],
            stop_tokens=stop_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            num_beams=num_beams,
            prompt_formatter=prompt_formatter,
            overwrite_manifest=overwrite_manifest,
            parallel=parallel,
        )
        for prediction, model_response in generated_responses:
            prediction = re.sub(r"[\s\t\n]+", " ", prediction)
            token_lengths.append(len(tokenizer(prediction).input_ids))
            console.print(f"[blue]Prompt:[/blue] {model_response.final_prompt}")
            console.print(f"[red]Prediction:[/red] {prediction}")
            if data[i].get("query") or data[i].get("sql"):
                console.print(
                    "[purple]Gold:[/purple] "
                    f"{data[i].get('query') or data[i].get('sql')}"
                )
            console.print("\n****\n")

    # Run the entire thing now - the to_print results will be in cache and fast
    generated_sqls = generate_sql(
        manifest=manifest,
        text_to_sql_in=text_to_sql_in,
        retrieved_docs=retrieved_docs,
        stop_tokens=stop_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        num_beams=num_beams,
        prompt_formatter=prompt_formatter,
        overwrite_manifest=overwrite_manifest,
        parallel=parallel,
    )

    with open(Path(output_dir) / output_filename, "w") as fout:
        for i, (prediction, model_response) in enumerate(generated_sqls):
            if isinstance(model_response.final_prompt, str):
                token_lengths.append(
                    len(tokenizer(model_response.final_prompt).input_ids)
                )
            else:
                for prompt in model_response.final_prompt:
                    token_lengths.append(len(tokenizer(prompt["content"]).input_ids))
            entry = {
                **original_data[i],
                "pred": prediction,
                "raw_pred": model_response.output,
                "raw_output": model_response.raw_output,
                "prompt": model_response.final_prompt,
                "tables": [tbl.model_dump() for tbl in text_to_sql_in[i].tables or []],
            }
            formatted_entry = data_formatter.format_output(entry)
            print(json.dumps(formatted_entry), file=fout)
    overflow = len([tl for tl in token_lengths if tl > 2048]) / len(token_lengths)
    console.print(f"Overflow 2048 prompt {100*overflow:.2f}%")
    console.print(f"Saved to {Path(output_dir) / output_filename}")


if __name__ == "__main__":
    cli()
