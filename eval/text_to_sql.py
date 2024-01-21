"""Text-to-SQL running."""
import asyncio
import json
import re
import time
from typing import cast

import structlog
from manifest import Manifest
from manifest.response import Response, Usage
from prompt_formatters import RajkumarFormatter
from schema import DEFAULT_TABLE_NAME, TextToSQLModelResponse, TextToSQLParams
from tqdm.auto import tqdm

logger = structlog.get_logger()


def clean_whitespace(sql: str) -> str:
    """Clean whitespace."""
    return re.sub(r"[\t\n\s]+", " ", sql)


def instruction_to_sql(
    params: TextToSQLParams,
    extra_context: list[str],
    manifest: Manifest,
    prompt_formatter: RajkumarFormatter = None,
    overwrite_manifest: bool = False,
    max_tokens: int = 300,
    temperature: float = 0.0,
    stop_sequences: list[str] | None = None,
    num_beams: int = 1,
) -> TextToSQLModelResponse:
    """Parse the instruction to a sql command."""
    return instruction_to_sql_list(
        params=[params],
        extra_context=[extra_context],
        manifest=manifest,
        prompt_formatter=prompt_formatter,
        overwrite_manifest=overwrite_manifest,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
        num_beams=num_beams,
    )[0]


def instruction_to_sql_list(
    params: list[TextToSQLParams],
    extra_context: list[list[str]],
    manifest: Manifest,
    prompt_formatter: RajkumarFormatter = None,
    overwrite_manifest: bool = False,
    max_tokens: int = 300,
    temperature: float = 0.0,
    stop_sequences: list[str] | None = None,
    num_beams: int = 1,
    verbose: bool = False,
) -> list[TextToSQLModelResponse]:
    """Parse the list of instructions to sql commands.

    Connector is used for default retry handlers only.
    """
    if prompt_formatter is None:
        raise ValueError("Prompt formatter is required.")

    def construct_params(
        params: TextToSQLParams,
        context: list[str],
    ) -> str | list[dict]:
        """Turn params into prompt."""
        if prompt_formatter.clean_whitespace:
            instruction = clean_whitespace(params.instruction)
        else:
            instruction = params.instruction

        table_texts = prompt_formatter.format_all_tables(
            params.tables, instruction=instruction
        )
        # table_texts can be list of chat messages. Only join list of str.
        if table_texts:
            if isinstance(table_texts[0], str):
                table_text = prompt_formatter.table_sep.join(table_texts)
            else:
                table_text = table_texts
        else:
            table_text = ""

        if context:
            context_text = prompt_formatter.format_retrieved_context(context)
        else:
            context_text = "" if isinstance(table_text, str) else []
        prompt = prompt_formatter.format_prompt(
            instruction,
            table_text,
            context_text,
        )
        return prompt

    # If no inputs, return nothing
    if not params:
        return []

    # Stitch together demonstrations and params
    prompts: list[str | list[dict]] = []
    for i, param in tqdm(
        enumerate(params),
        total=len(params),
        desc="Constructing prompts",
        disable=not verbose,
    ):
        predict_str = construct_params(param, extra_context[i] if extra_context else [])
        if isinstance(predict_str, str):
            prompt = predict_str.lstrip()
        else:
            prompt = predict_str
        prompts.append(prompt)

    manifest_params = dict(
        max_tokens=max_tokens,
        overwrite_cache=overwrite_manifest,
        num_beams=num_beams,
        logprobs=5,
        temperature=temperature,
        do_sample=False if temperature <= 0 else True,
        stop_sequences=stop_sequences or prompt_formatter.stop_sequences,
    )

    ret: list[TextToSQLModelResponse] = []
    if len(params) == 1:
        prompt = prompts[0]
        model_response = _run_manifest(
            prompt,
            manifest_params,
            prompt_formatter,
            manifest,
            stop_sequences=stop_sequences,
        )
        usage = model_response.usage
        model_response.usage = usage
        ret.append(model_response)
    else:
        # We do not handle retry logic on parallel requests right now
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = cast(
            Response,
            loop.run_until_complete(
                manifest.arun_batch(
                    prompts,
                    **manifest_params,  # type: ignore
                ),
            ),
        )
        loop.close()

        response_usage = response.get_usage()
        response_text = response.get_parsed_response()
        for prompt, resp in zip(prompts, response_text):
            # This will restitch the query in the case we force it to start with SELECT
            sql_query = prompt_formatter.format_model_output(cast(str, resp), prompt)
            for token in stop_sequences:
                sql_query = sql_query.split(token)[0]
            logger.info(f"FINAL OUTPUT: {sql_query}")
            ret.append(
                TextToSQLModelResponse(
                    output=sql_query,
                    raw_output=cast(str, resp),
                    final_prompt=prompt,
                    usage=response_usage,
                )
            )

    return ret


def _run_manifest(
    prompt: str | list[str],
    manifest_params: dict,
    prompt_formatter: RajkumarFormatter,
    manifest: Manifest,
    stop_sequences: list[str] | None = None,
) -> TextToSQLModelResponse:
    """Run manifest for prompt format."""
    logger.info(f"PARAMS: {manifest_params}")
    if isinstance(prompt, list):
        for p in prompt:
            logger.info(f"PROMPT: {p['role']}: {p['content']}")
    else:
        logger.info(f"PROMPT: {prompt}")
    start_time = time.time()
    # Run result
    response = cast(
        Response,
        manifest.run(
            prompt,
            return_response=True,
            client_timeout=1800,
            **manifest_params,  # type: ignore
        ),
    )
    logger.info(f"TIME: {time.time() - start_time: .2f}")

    response_usage = response.get_usage_obj()
    summed_usage = Usage()
    for usage in response_usage.usages:
        summed_usage.completion_tokens += usage.completion_tokens
        summed_usage.prompt_tokens += usage.prompt_tokens
        summed_usage.total_tokens += usage.total_tokens
    # This will restitch the query in the case we force it to start with SELECT
    sql_query = prompt_formatter.format_model_output(
        cast(str, response.get_response()), prompt
    )

    for token in stop_sequences:
        sql_query = sql_query.split(token)[0]
    logger.info(f"OUTPUT: {sql_query}")
    model_response = TextToSQLModelResponse(
        output=sql_query,
        raw_output=cast(str, response.get_response()),
        final_prompt=prompt,
        usage=summed_usage,
    )
    return model_response
