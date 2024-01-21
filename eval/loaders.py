"""Data loaders."""
import json
import re
import string
from abc import ABC, abstractmethod

from rich.console import Console
from data_utils import read_tables_json
from schema import Table

RE_COLUMN = re.compile(r"^select (.+?) from")
RE_CONDS = re.compile(r"where (.+?)$")
RE_COND = re.compile(r"^(.+?)\s*([=><])\s*(.+?)$")

translator = str.maketrans(
    string.punctuation, " " * len(string.punctuation)
)  # map punctuation to space

console = Console(soft_wrap=True)


def standardize_column(col: str) -> str:
    """Standardize the column name to SQL compatible."""
    col_name = col.replace("#", "num").replace("%", "perc")
    col_name = col_name.strip().lower().translate(translator)
    col_name = re.sub("[^0-9a-z ]", " ", col_name).strip()
    col_name = re.sub(" +", "_", col_name)
    if not col_name:
        console.print(f"original {col}, new {col_name}")
    return col_name


def clean_col(col: str) -> str:
    """Remove table name and standardize column name."""
    if "." in col and not col.endswith("."):
        col = col.split(".")[-1]
    return standardize_column(col)


class Loader(ABC):
    """Loader abstract class."""

    @classmethod
    @abstractmethod
    def load_data(cls, path: str) -> list[dict]:
        """Load data from path."""

    @classmethod
    @abstractmethod
    def load_table_metadata(cls, path: str) -> dict[str, dict[str, Table]]:
        """Extract table metadata from table-metadata-path."""

    @classmethod
    def format_output(cls, prediction: dict) -> dict:
        """Parse for spider format."""
        return prediction


class DefaultLoader(Loader):
    """Spider loader and writer."""

    @classmethod
    def load_data(cls, path: str) -> list[dict]:
        """Load data from path."""
        try:
            with open(path) as f:
                data = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            # Try with jsonl
            data = [json.loads(line) for line in open(path)]
        return data

    @classmethod
    def load_table_metadata(cls, path: str) -> dict[str, dict[str, Table]]:
        """Extract table metadata from table-metadata-path."""
        # load the tables
        db_to_tables = read_tables_json(path, lowercase=True)
        return db_to_tables
