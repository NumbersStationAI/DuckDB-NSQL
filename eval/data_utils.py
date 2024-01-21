"""Training data prep utils."""
import json
import re
from collections import defaultdict
from schema import ForeignKey, Table, TableColumn


def read_tables_json(
    schema_file: str,
    lowercase: bool = False,
) -> dict[str, dict[str, Table]]:
    """Read tables json."""
    data = json.load(open(schema_file))
    db_to_tables = {}
    for db in data:
        db_name = db["db_id"]
        table_names = db["table_names_original"]
        db["column_names_original"] = [
            [x[0], x[1]] for x in db["column_names_original"]
        ]
        db["column_types"] = db["column_types"]
        if lowercase:
            table_names = [tn.lower() for tn in table_names]
        pks = db["primary_keys"]
        fks = db["foreign_keys"]
        tables = defaultdict(list)
        tables_pks = defaultdict(list)
        tables_fks = defaultdict(list)
        for idx, ((ti, col_name), col_type) in enumerate(
            zip(db["column_names_original"], db["column_types"])
        ):
            if ti == -1:
                continue
            if lowercase:
                col_name = col_name.lower()
                col_type = col_type.lower()
            if idx in pks:
                tables_pks[table_names[ti]].append(
                    TableColumn(name=col_name, dtype=col_type)
                )
            for fk in fks:
                if idx == fk[0]:
                    other_column = db["column_names_original"][fk[1]]
                    other_column_type = db["column_types"][fk[1]]
                    other_table = table_names[other_column[0]]
                    tables_fks[table_names[ti]].append(
                        ForeignKey(
                            column=TableColumn(name=col_name, dtype=col_type),
                            references_name=other_table,
                            references_column=TableColumn(
                                name=other_column[1], dtype=other_column_type
                            ),
                        )
                    )
            tables[table_names[ti]].append(TableColumn(name=col_name, dtype=col_type))
        db_to_tables[db_name] = {
            table_name: Table(
                name=table_name,
                columns=tables[table_name],
                pks=tables_pks[table_name],
                fks=tables_fks[table_name],
                examples=None,
            )
            for table_name in tables
        }
    return db_to_tables


def clean_str(target: str) -> str:
    """Clean string for question."""
    if not target:
        return target

    target = re.sub(r"[^\x00-\x7f]", r" ", target)
    line = re.sub(r"''", r" ", target)
    line = re.sub(r"``", r" ", line)
    line = re.sub(r"\"", r"'", line)
    line = re.sub(r"[\t ]+", " ", line)
    return line.strip()
