"""SQL guards: allow a single read-only SELECT (including WITH/CTE)."""

from __future__ import annotations

import re

# Statement keywords that must never appear as SQL tokens in agent queries.
_FORBIDDEN = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "DROP",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "ATTACH",
        "DETACH",
        "COPY",
        "EXPORT",
        "IMPORT",
        "INSTALL",
        "LOAD",
        "PRAGMA",
        "CALL",
        "EXECUTE",
        "SET",
        "RESET",
        "GRANT",
        "REVOKE",
        "VACUUM",
        "CHECKPOINT",
        "FORCE",
        "EXPLAIN",
        "DESCRIBE",
        "SHOW",
        "SUMMARIZE",
        "PIVOT",
        "UNPIVOT",
        "PREPARE",
        "DEALLOCATE",
        "TRANSACTION",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "ABORT",
    }
)

_COMMENT_LINE = re.compile(r"--.*?$", re.MULTILINE)
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING = re.compile(
    r"('(?:''|[^'])*')|(\"(?:\"\"|[^\"])*\")",
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SqlGuardError(ValueError):
    """Raised when a query fails the read-only single-SELECT guard."""


def _strip_comments_and_strings(sql: str) -> str:
    """Remove comments and replace string literals with placeholders for keyword scans."""
    no_comments = _COMMENT_BLOCK.sub(" ", _COMMENT_LINE.sub(" ", sql))
    return _STRING.sub(" '' ", no_comments)


def validate_readonly_select(query: str) -> str:
    """Validate and normalize a single read-only SELECT / WITH query.

    Returns the stripped SQL (trailing semicolons removed).
    Raises SqlGuardError on anything else.
    """
    if query is None or not str(query).strip():
        raise SqlGuardError("Query must be a non-empty SELECT statement")

    raw = str(query).strip()
    # Disallow null bytes / obvious injection junk.
    if "\x00" in raw:
        raise SqlGuardError("Query contains invalid characters")

    # Allow a single trailing semicolon only.
    body = raw.rstrip().rstrip(";").strip()
    if not body:
        raise SqlGuardError("Query must be a non-empty SELECT statement")
    if ";" in body:
        raise SqlGuardError("Only a single SQL statement is allowed")

    scanned = _strip_comments_and_strings(body)
    if not scanned.strip():
        raise SqlGuardError("Query must be a non-empty SELECT statement")

    tokens = [m.group(0).upper() for m in _IDENT.finditer(scanned)]
    if not tokens:
        raise SqlGuardError("Query must be a SELECT or WITH ... SELECT statement")

    if tokens[0] not in {"SELECT", "WITH"}:
        raise SqlGuardError(
            "Only a single read-only SELECT (or WITH ... SELECT) is allowed"
        )

    forbidden_hit = next((t for t in tokens if t in _FORBIDDEN), None)
    if forbidden_hit is not None:
        raise SqlGuardError(
            f"Forbidden SQL keyword for read-only tool: {forbidden_hit}"
        )

    # WITH queries must eventually select; reject WITH-only nonsense lightly.
    if tokens[0] == "WITH" and "SELECT" not in tokens:
        raise SqlGuardError("WITH query must include a SELECT")

    return body
