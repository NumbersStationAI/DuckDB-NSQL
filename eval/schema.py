"""Text2SQL schemas."""
import enum

from manifest.response import Usage
from pydantic import BaseModel

DEFAULT_TABLE_NAME: str = "db_table"


class Dialect(str, enum.Enum):
    """SQGFluff and SQLGlot dialects.

    Lucky for us, the dialects match both parsers.

    Ref: https://github.com/sqlfluff/sqlfluff/blob/main/src/sqlfluff/core/dialects/__init__.py  # noqa: E501
    Ref: https://github.com/tobymao/sqlglot/blob/main/sqlglot/dialects/__init__.py  # noqa: E501
    """

    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    POSTGRES = "postgres"
    UNKNOWN = "unknown"

    @property
    def dialect_str(self) -> str | None:
        """Get the dialect string for validation.

        We need to pass in dialect = None for UNKNOWN dialects.
        """
        if self != Dialect.UNKNOWN:
            return self.value
        else:
            return None

    @property
    def quote_str(self) -> str:
        """Get the quote string for the dialect."""
        if self == Dialect.SNOWFLAKE:
            return '"'
        elif self == Dialect.BIGQUERY:
            return "`"
        elif self == Dialect.REDSHIFT:
            return '"'
        elif self == Dialect.POSTGRES:
            return '"'
        elif self == Dialect.UNKNOWN:
            return '"'
        raise NotImplementedError(f"Quote string not implemented for dialect {self}")

    def quote(self, string: str) -> str:
        """Quote a string."""
        return f"{self.quote_str}{string}{self.quote_str}"


class ColumnOrLiteral(BaseModel):
    """Column that may or may not be a literal."""

    name: str | None = None
    literal: bool = False

    def __hash__(self) -> int:
        """Hash."""
        return hash((self.name, self.literal))


class TableColumn(BaseModel):
    """Table column."""

    name: str
    dtype: str | None


class ForeignKey(BaseModel):
    """Foreign key."""

    # Referenced column
    column: TableColumn
    # References table name
    references_name: str
    # References column
    references_column: TableColumn


class Table(BaseModel):
    """Table."""

    name: str | None
    columns: list[TableColumn] | None
    pks: list[TableColumn] | None
    # FK from this table to another column in another table
    fks: list[ForeignKey] | None
    examples: list[dict] | None
    # Is the table a source or intermediate reference table
    is_reference_table: bool = False


class TextToSQLParams(BaseModel):
    """A text to sql request."""

    instruction: str
    database: str | None
    # Default to unknown
    dialect: Dialect = Dialect.UNKNOWN
    tables: list[Table] | None


class TextToSQLModelResponse(BaseModel):
    """Model for Autocomplete Responses."""

    output: str
    final_prompt: str | list[dict]
    raw_output: str
    usage: Usage
    metadata: str | None = None
