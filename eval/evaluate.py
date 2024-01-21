"""Evaluate text2sql spider model predictions."""
import json
import os
import re
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.console import Console
from tqdm.auto import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
# from metrics.spider import evaluation as spider_evaluation  # type: ignore # noqa: E402
from metrics.test_suite_sql_eval import (  # type: ignore # noqa: E402
    evaluation as test_suite_evaluation,
)
from data_utils import read_tables_json  # type: ignore  # noqa: E402
from metric_utils import (  # type: ignore  # noqa: E402
    correct_casing,
    edit_distance,
)

console = Console(soft_wrap=True)

LEVELS = ["easy", "medium", "hard", "duckdb", "ddl", "all"]
PARTIAL_TYPES = [
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
]
TIMEOUT_SECONDS = 30


def timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Function execution timed out.")


def print_scores(scores: dict, model_name: str, metric_type: str = "exec") -> None:
    """Print scores."""

    def print_formated_s(
        row_name: str, l: list[str], element_format: str = "{}", sep: str = "\t"
    ) -> None:
        template = "{}" + sep + sep.join([element_format] * len(l))
        console.print(template.format(row_name, *l))

    # Add empty scores for each level if not present
    for level in LEVELS:
        if level not in scores:
            scores[level] = {}
            scores[level]["count"] = 0
            scores[level]["exec"] = 0
            scores[level]["exact"] = 0

    print_formated_s("", LEVELS)
    counts = [scores[level]["count"] for level in LEVELS]
    print_formated_s("count", counts)
    console.print(f">======================   {model_name}     =====================")
    if metric_type == "exec":
        console.print(
            ">=====================   EXECUTION ACCURACY     ====================="
        )
        exec_scores = [scores[level]["exec"] for level in LEVELS]
        print_formated_s("execution", exec_scores, element_format="{:.3f}")

    elif metric_type == "exact":
        console.print(
            "\n>====================== EXACT MATCHING ACCURACY ====================="
        )
        exact_scores = [scores[level]["exact"] for level in LEVELS]
        print_formated_s("exact match", exact_scores, element_format="{:.3f}")


def compute_exact_match_metric(
    predictions: list,
    references: list,
    gold_dbs: list,
    kmaps: dict,
    db_dir: str,
    categories,
) -> dict:
    """Compute exact match metric."""
    exact_match = {}
    exact_match["all"] = {}
    exact_match["all"]["count"] = 0
    exact_match["all"]["exact"] = 0
    for prediction, reference, gold_db, category in tqdm(
        zip(predictions, references, gold_dbs, categories), total=len(predictions)
    ):
        if category not in exact_match:
            exact_match[category] = {}
            exact_match[category]["count"] = 0
            exact_match[category]["exact"] = 0
        exact_match["all"]["count"] += 1
        exact_match[category]["count"] += 1
        try:
            match = int(prediction.trim() == reference.trim())
            exact_match[category]["exact"] += match
            exact_match["all"]["exact"] += match
        except Exception:
            pass
    return exact_match


def compute_test_suite_metric(
    predictions: list,
    references: list,
    gold_dbs: list,
    setup_sqls: list,
    validate_sqls: list,
    kmaps: dict,
    db_dir: str,
    categories: list[str] = None,
) -> tuple[Any, list[int | None]]:
    """Compute test suite execution metric."""
    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=kmaps,
        etype="exec",
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False,
    )
    # Only used for Sparc/CoSQL
    turn_scores: dict[str, list] = {"exec": [], "exact": []}
    by_row_metrics: list[int | None] = []
    for prediction, reference, gold_db, setup_sql, validate_sql, category in tqdm(
        zip(predictions, references, gold_dbs, setup_sqls, validate_sqls, categories),
        total=len(predictions),
    ):
        turn_idx = 0
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue

        # Register the timeout handler function
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        try:
            ex_metrics = evaluator.evaluate_one(
                gold_db,
                reference,
                prediction,
                setup_sql,
                validate_sql,
                turn_scores,
                idx=turn_idx,
                category=category,
            )
            signal.alarm(0)

            by_row_metrics.append(int(ex_metrics["exec"]))
        except Exception as e:
            raise e
            by_row_metrics.append(None)
            pass
    evaluator.finalize()
    return evaluator.scores, by_row_metrics


def compute_metrics(
    gold_sqls: list[str],
    pred_sqls: list[str],
    gold_dbs: list[str],
    setup_sqls: list[str],
    validate_sqls: list[str],
    kmaps: dict,
    db_schemas: dict,
    database_dir: str,
    lowercase_schema_match: bool,
    model_name: str,
    categories: list[str] = None,
) -> dict[str, str]:
    """Compute all metrics for data slice."""
    if len(gold_sqls) != len(pred_sqls):
        raise ValueError(
            f"Gold {len(gold_sqls)} and pred {len(pred_sqls)} have different number of lines!"
        )
    all_metrics: dict[str, Any] = {}

    # Execution Accuracy
    metrics, by_row_metrics = compute_test_suite_metric(
        pred_sqls,
        gold_sqls,
        gold_dbs,
        setup_sqls,
        validate_sqls,
        kmaps,
        database_dir,
        categories,
    )
    all_metrics["exec"] = metrics
    all_metrics["by_row_exec"] = by_row_metrics
    print_scores(metrics, model_name, "exec")

    # Exact Match Accuracy
    metrics = compute_exact_match_metric(
        pred_sqls, gold_sqls, gold_dbs, kmaps, database_dir, categories
    )
    all_metrics["exact"] = metrics
    print_scores(metrics, model_name, "exact")

    # Equality Accuracy
    per_row_match = [
        int(gold.lower() == pred.lower()) for gold, pred in zip(gold_sqls, pred_sqls)
    ]
    all_metrics["equality"] = {"equality": sum(per_row_match) / len(gold_sqls)}
    all_metrics["by_row_equality"] = per_row_match

    # Edit Distance
    per_row_edit_dist = [
        edit_distance(gold, pred) for gold, pred in zip(gold_sqls, pred_sqls)
    ]
    edit_dist = sum(per_row_edit_dist) / len(gold_sqls)
    all_metrics["edit_distance"] = {"edit_distance": edit_dist}
    all_metrics["by_row_edit_distance"] = per_row_edit_dist

    return all_metrics


def get_to_print(metrics: dict, key: str, model_name: str, num_rows: int) -> dict:
    """Get pretty print dictionary of metrics."""
    return {
        "slice": key,
        "model": model_name,
        "support": num_rows,
        "exec": f"{metrics[key]['exec']['all']['exec']:.3f}",
        "exact": f"{metrics[key]['exact']['all']['exact']:.3f}",
        "equality": f"{metrics[key]['equality']['equality']:.3f}",
        "edit_distance": f"{metrics[key]['edit_distance']['edit_distance']:.3f}",
    }


@click.group()
def cli() -> None:
    """Entrypoint."""
    pass


@cli.command()
@click.option("--gold", type=str, required=True)
@click.option("--pred", type=str, required=True)
@click.option("--tables", type=str, required=True)
@click.option("--db", type=str, default="")
@click.option("--slice-attribute", type=str, default=None)
@click.option("--output-dir", type=str, default="")
@click.option("--output-filename", type=str, default="")
@click.option(
    "--correct-sql-casing", type=bool, is_flag=True, default=False, required=False
)
@click.option(
    "--lowercase-schema-match", type=bool, is_flag=True, default=False, required=False
)
def evaluate(
    gold: str,
    pred: str,
    tables: str,
    db: str,
    slice_attribute: str,
    output_dir: str,
    output_filename: str,
    correct_sql_casing: bool,
    lowercase_schema_match: bool,
) -> None:
    """Evaluate SQL.

    Args:
        gold: path to gold sql file.
        pred: path to predicted json lines file.
        tables: the json path of the table metadata.
        db: path to database dir.
        slice_attribute: json attribute in gold data to slice on.
        output_dir: the prediction output directory
        output_filename: the prediction output filename
        correct_sql_casing: whether to correct casing of SQL keywords
        lowercase_schema_match: whether to lowercase schema match
    """
    gold_path = Path(gold)
    pred_path = Path(pred)
    model_name = pred_path.stem
    if not output_filename:
        output_filename = pred_path.stem + "_eval.json"
    console.print(f"Saving to {Path(output_dir) / output_filename}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    kmaps = test_suite_evaluation.build_foreign_key_map_from_json(tables)
    db_schemas = read_tables_json(tables)

    gold_sqls_dict = json.load(gold_path.open("r", encoding="utf-8"))
    pred_sqls_dict = [json.loads(l) for l in pred_path.open("r").readlines()]

    # Data validation
    assert len(gold_sqls_dict) == len(
        pred_sqls_dict
    ), "Sample size doesn't match between pred and gold file"

    # Keep track of everything
    full_results = []
    for gold_sql, pred_sql in zip(gold_sqls_dict, pred_sqls_dict):
        merged_res = {**pred_sql, **gold_sql}
        full_results.append(merged_res)

    gold_sqls = [
        re.sub(r"[\s\t\n]+", " ", p.get("gold", p.get("query", p.get("sql", ""))))
        for p in gold_sqls_dict
    ]
    setup_sqls = [re.sub(r"[\s\t\n]+", " ", p["setup_sql"]) for p in gold_sqls_dict]
    validate_sqls = [
        re.sub(r"[\s\t\n]+", " ", p["validation_sql"]) for p in gold_sqls_dict
    ]
    gold_dbs = [p.get("db_id", p.get("db", "")) for p in gold_sqls_dict]
    pred_sqls = [re.sub(r"[\s\t\n]+", " ", p["pred"]) for p in pred_sqls_dict]
    categories = [p.get("category", "") for p in gold_sqls_dict]
    if correct_sql_casing:
        # One line to correct casing of SQL keywords using correct_casing(sql)
        gold_sqls = [correct_casing(sql) for sql in gold_sqls]
        pred_sqls = [correct_casing(sql) for sql in pred_sqls]

    final_metrics: dict[str, dict[str, Any]] = {}
    to_print = []
    final_metrics["all"] = compute_metrics(
        gold_sqls=gold_sqls,
        pred_sqls=pred_sqls,
        gold_dbs=gold_dbs,
        setup_sqls=setup_sqls,
        validate_sqls=validate_sqls,
        kmaps=kmaps,
        db_schemas=db_schemas,
        database_dir=db,
        lowercase_schema_match=lowercase_schema_match,
        model_name=model_name + "(all)",
        categories=categories,
    )

    for k, v in final_metrics["all"].items():
        if k.startswith("by_row"):
            assert len(v) == len(gold_sqls)
            for dct, val in zip(full_results, v):
                dct[k[len("by_row_") :]] = val
    to_print.append(get_to_print(final_metrics, "all", model_name, len(gold_sqls)))
    # TODO: could be way more efficient if we subsliced the results but...whatever
    if slice_attribute:
        for unq_value in sorted(set([g[slice_attribute] for g in gold_sqls_dict])):
            idx_set = [
                i
                for i, g in enumerate(gold_sqls_dict)
                if g[slice_attribute] == unq_value
            ]
            print(f"Processing {unq_value} with {len(idx_set)} samples")
            final_metrics[unq_value] = compute_metrics(
                gold_sqls=[gold_sqls[i] for i in idx_set],
                pred_sqls=[pred_sqls[i] for i in idx_set],
                gold_dbs=[gold_dbs[i] for i in idx_set],
                setup_sqls=[setup_sqls[i] for i in idx_set],
                validate_sqls=[validate_sqls[i] for i in idx_set],
                kmaps=kmaps,
                db_schemas=db_schemas,
                database_dir=db,
                lowercase_schema_match=lowercase_schema_match,
                model_name=model_name + f"({unq_value})",
                categories=[categories[i] for i in idx_set],
            )
            to_print.append(
                get_to_print(final_metrics, unq_value, model_name, len(idx_set))
            )

    df = pd.DataFrame(to_print)
    console.print(df.to_csv(sep=",", index=False))
    console.print("******")
    console.print(f"Saved metrics to {Path(output_dir) / output_filename}")
    json.dump(final_metrics, open(Path(output_dir) / output_filename, "w"), indent=4)
    output_filename = str(output_filename).replace("_eval.json", "_fd.jsonl")
    console.print(f"Saved dump to {Path(output_dir) / output_filename}")
    with open(Path(output_dir) / output_filename, "w") as f:
        for dct in full_results:
            f.write(json.dumps(dct) + "\n")


if __name__ == "__main__":
    cli()
