"""Rajkumar prompt formatter."""

from random import shuffle
from manifest import Manifest
from schema import Table


class RajkumarFormatter:
    """RajkumarFormatter class.

    From https://arxiv.org/pdf/2204.00498.pdf.
    """

    table_sep: str = "\n\n"
    shuffle_table_order: bool = True
    _cache: dict[tuple[str, str, str], list[str]] = {}
    clean_whitespace = False

    @classmethod
    def format_table(cls, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table.name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table.name}"
        return create_tbl

    @classmethod
    def format_all_tables(cls, tables: list[Table], instruction: str) -> list[str]:
        """Get all tables format."""
        table_texts = [cls.format_table(table) for table in tables]
        key = ("tables", instruction, str(tables))
        if key not in cls._cache:
            shuffle(table_texts)
            cls._cache[key] = table_texts
        else:
            table_texts = cls._cache[key]
        return table_texts

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n\n/*\nHere is additional documentation about DuckDB that could be useful.\n--------\n{context_str}\n--------\n*/"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        return f"""{table_text}\n\n\n-- Using valid DuckDB SQL, answer the following question for the tables provided above.{context_text}\n\n-- {instruction}\n"""  # noqa: E501

    @classmethod
    def format_model_output(cls, output_sql: str, prompt: str) -> str:
        """Format model output."""
        return output_sql

    @classmethod
    def format_gold_output(cls, output_sql: str) -> str:
        """Format gold output for demonstration."""
        return output_sql


class DuckDBFormatter(RajkumarFormatter):
    """DuckDB class."""

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        return f"""{table_text}\n\n\n-- Using valid DuckDB SQL, answer the following question for the tables provided above.{context_text}\n\n-- {instruction}\n```sql\n"""  # noqa: E501


class DuckDBInstFormatter(RajkumarFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """### Instruction:\n{instruction}\n\n### Input:\n{input}{context}\n### Question:\n{question}\n\n### Response (use duckdb shorthand if possible):\n"""
    INSTRUCTION_TEMPLATE = """Your task is to generate valid duckdb SQL to answer the following question{has_schema}"""  # noqa: E501

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n### Documentation:\n{context_str}\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: str,
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        input = ""
        if table_text:
            input = """Here is the database schema that the SQL query will run on:\n{schema}\n""".format(  # noqa: E501
                schema=table_text
            )
        instruction = cls.PROMPT_TEMPLATE.format(
            instruction=cls.INSTRUCTION_TEMPLATE.format(
                has_schema="."
                if table_text == ""
                else ", given a duckdb database schema."
            ),
            context=context_text,
            input=input,
            question=instruction,
        )
        return instruction


class DuckDBInstNoShorthandFormatter(DuckDBInstFormatter):
    """DuckDB Inst class."""

    PROMPT_TEMPLATE = """### Instruction:\n{instruction}\n\n### Input:\n{input}{context}\n### Question:\n{question}\n\n### Response:\n"""
    INSTRUCTION_TEMPLATE = """Your task is to generate valid duckdb SQL to answer the following question{has_schema}"""  # noqa: E501


class DuckDBChat:
    """DuckDB Inst class."""

    table_sep: str = "\n\n"
    shuffle_table_order: bool = True
    _cache: dict[tuple[str, str, str], list[str]] = {}
    clean_whitespace = False
    model = None

    @classmethod
    def format_table(cls, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table.name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table.name}"
        return create_tbl

    @classmethod
    def format_all_tables(cls, tables: list[Table], instruction: str) -> list[dict]:
        """Get all tables format."""
        if not cls.model:
            cls.model = Manifest(
                engine="gpt-3.5-turbo",
                client_name="openaichat",
                cache_name="sqlite",
                cache_connection=".manifest.sqlite",
            )
        table_texts = [cls.format_table(table) for table in tables]
        full_schema = cls.table_sep.join(table_texts)
        prompt = f"""SQL schema of my database:
{full_schema}
Explain in a few sentences what the data is about:
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can generate an human redable summary of database content based on the schema.",
            },
            {"role": "user", "content": prompt},
        ]
        explanation = cls.model.run(messages, temperature=0)
        messages.append({"role": "assistant", "content": explanation})
        return messages[1:]

    @classmethod
    def format_retrieved_context(
        cls,
        context: list[str],
    ) -> str:
        """Format retrieved context."""
        context_str = "\n--------\n".join(context)
        return f"\n\nHere is additional documentation about DuckDB that could be useful.\n--------\n{context_str}\n--------\n"

    @classmethod
    def format_prompt(
        cls,
        instruction: str,
        table_text: list[dict],
        context_text: str,
    ) -> str | list[str]:
        """Get prompt format."""
        prompt = f"""Now output a single SQL query without any explanation and do not add anything 
to the query that was not part of the question, also do not use markdown. Make sure to only 
use information provided in the prompt, or tables and columns from the schema above and write a query to answer the question.{context_text}\n\nMy quesiton is \n`{instruction}`\n\nGenerate the DuckDB specific SQL query:"""  # noqa: E501
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can generate DuckDB sql queries, which is a superset of Postgresql, based on the user input. You do not respond with any human readable text, only SQL code.",
            },
            *table_text,
            {"role": "user", "content": prompt},
        ]
        return messages

    @classmethod
    def format_model_output(cls, output_sql: str, prompt: str) -> str:
        """Format model output."""
        return output_sql

    @classmethod
    def format_gold_output(cls, output_sql: str) -> str:
        """Format gold output for demonstration."""
        return output_sql
