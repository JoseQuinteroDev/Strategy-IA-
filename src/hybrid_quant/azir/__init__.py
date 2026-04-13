"""AzirIA MT5 audit helpers.

The MQL5 expert advisor is the operational source of truth. This package only
contains schemas and utilities used to validate/export Azir logs before a
faithful Python replica is implemented.
"""

from .event_log import AZIR_EVENT_COLUMNS, REQUIRED_AZIR_EVENT_COLUMNS, validate_event_row, write_event_log
from .inspection import CsvInspection, classify_columns, inspect_csv

__all__ = [
    "AZIR_EVENT_COLUMNS",
    "REQUIRED_AZIR_EVENT_COLUMNS",
    "CsvInspection",
    "classify_columns",
    "inspect_csv",
    "validate_event_row",
    "write_event_log",
]
