import sys
import duckdb
from duckdb import ParserException, SyntaxException, BinderException, CatalogException


def validate_query(query, schemas):
    try:
        with duckdb.connect(
                ":memory:", config={"enable_external_access": False}
        ) as duckdb_conn:
            # register schemas
            for schema in schemas.split(";"):
                duckdb_conn.execute(schema)
            cursor = duckdb_conn.cursor()
            cursor.execute(query)
    except ParserException as e:
        return str(e)
    except SyntaxException as e:
        return str(e)
    except BinderException as e:
        return str(e)
    except CatalogException as e:
        if not ("but it exists" in str(e) and "extension" in str(e)):
            return str(e)
    except Exception as e:
        return None
    return None


if __name__ == "__main__":
    if len(sys.argv) > 2:
        error = validate_query(sys.argv[1], sys.argv[2])
        if error:
            raise Exception(error)
    else:
        print("No query provided.")
