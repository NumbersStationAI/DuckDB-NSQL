"""Constants."""

from prompt_formatters import (
    DuckDBFormatter,
    DuckDBInstFormatter,
    DuckDBInstNoShorthandFormatter,
    RajkumarFormatter,
    DuckDBChat,
)

PROMPT_FORMATTERS = {
    "rajkumar": RajkumarFormatter,
    "duckdb": DuckDBFormatter,
    "duckdbinst": DuckDBInstFormatter,
    "duckdbinstnoshort": DuckDBInstNoShorthandFormatter,
    "duckdbchat": DuckDBChat,
}
