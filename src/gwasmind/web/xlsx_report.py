"""Small, dependency-free XLSX writer for the local legacy UI.

The module intentionally implements only the subset needed for tabular result
downloads: one worksheet, a header row, and scalar cell values.  Text is stored
as explicit inline strings, and formula-looking text is prefixed with an
apostrophe so spreadsheet applications cannot reinterpret untrusted output as
a formula.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape, quoteattr
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

__all__ = ["build_xlsx_report"]


_MAX_ROWS = 1_048_576
_MAX_COLUMNS = 16_384
_MAX_CELL_CHARACTERS = 32_767
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_FORMULA_PREFIXES = frozenset("=+-@")
_INVALID_SHEET_CHARACTERS = re.compile(r"[\\/*?:\[\]]")


def build_xlsx_report(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    sheet_name: str = "Results",
) -> bytes:
    """Return a valid, single-sheet XLSX workbook.

    ``columns`` controls both output order and dictionary lookup. Missing keys
    and explicit ``None`` values produce blank cells. Strings that could be
    interpreted as formulas are made inert before being written.

    Args:
        rows: Row dictionaries to export.
        columns: Ordered dictionary keys and worksheet headings.
        sheet_name: Optional worksheet name; invalid Excel characters are
            replaced and the result is limited to 31 characters.

    Raises:
        TypeError: If rows/columns do not have the expected shape.
        ValueError: If the table exceeds XLSX limits or columns are ambiguous.
    """

    column_names = _validate_columns(columns)
    row_values = _validate_rows(rows)
    if len(row_values) + 1 > _MAX_ROWS:
        raise ValueError(f"XLSX supports at most {_MAX_ROWS - 1} data rows")

    safe_sheet_name = _safe_sheet_name(sheet_name)
    last_column = _column_letter(len(column_names))
    last_row = len(row_values) + 1
    table_range = f"A1:{last_column}{last_row}"

    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        _write_entry(archive, "[Content_Types].xml", _CONTENT_TYPES)
        _write_entry(archive, "_rels/.rels", _ROOT_RELATIONSHIPS)
        _write_entry(archive, "xl/workbook.xml", _workbook_xml(safe_sheet_name))
        _write_entry(archive, "xl/_rels/workbook.xml.rels", _WORKBOOK_RELATIONSHIPS)
        _write_entry(archive, "xl/styles.xml", _STYLES)
        _write_worksheet(
            archive,
            rows=row_values,
            columns=column_names,
            table_range=table_range,
        )
    return output.getvalue()


def _validate_columns(columns: Sequence[str]) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise TypeError("columns must be a sequence of strings")
    result = tuple(columns)
    if not result:
        raise ValueError("at least one column is required")
    if len(result) > _MAX_COLUMNS:
        raise ValueError(f"XLSX supports at most {_MAX_COLUMNS} columns")
    if any(not isinstance(column, str) for column in result):
        raise TypeError("every column name must be a string")
    if len(set(result)) != len(result):
        raise ValueError("column names must be unique")
    return result


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mappings")
    result = tuple(rows)
    for index, row in enumerate(result):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {index} is not a mapping")
    return result


def _safe_sheet_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("sheet_name must be a string")
    cleaned = _clean_xml_text(_INVALID_SHEET_CHARACTERS.sub("_", value)).strip()
    cleaned = cleaned.strip("'")[:31].strip("'")
    return cleaned or "Results"


def _write_entry(archive: ZipFile, name: str, content: str) -> None:
    info = _zip_info(name)
    archive.writestr(info, content.encode("utf-8"))


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 0
    return info


def _write_worksheet(
    archive: ZipFile,
    *,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    table_range: str,
) -> None:
    info = _zip_info("xl/worksheets/sheet1.xml")
    with archive.open(info, mode="w") as worksheet:
        _write_xml(
            worksheet,
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="{table_range}"/>'
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '</sheetView></sheetViews>'
            '<sheetFormatPr defaultRowHeight="15"/>'
            '<sheetData>',
        )

        # Headings are the caller's fixed column names, not untrusted output;
        # guarding them would show a literal apostrophe, e.g. "'-log10(P)".
        header_cells = "".join(
            _text_cell(f"{_column_letter(index)}1", column, style=1, guard=False)
            for index, column in enumerate(columns, start=1)
        )
        _write_xml(worksheet, f'<row r="1">{header_cells}</row>')

        for row_number, row in enumerate(rows, start=2):
            _write_xml(worksheet, f'<row r="{row_number}">')
            for column_number, column in enumerate(columns, start=1):
                reference = f"{_column_letter(column_number)}{row_number}"
                _write_xml(worksheet, _cell_xml(reference, row.get(column)))
            _write_xml(worksheet, "</row>")

        _write_xml(
            worksheet,
            '</sheetData>'
            f'<autoFilter ref="{table_range}"/>'
            '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" '
            'header="0.3" footer="0.3"/>'
            '</worksheet>',
        )


def _write_xml(stream: Any, value: str) -> None:
    stream.write(value.encode("utf-8"))


def _cell_xml(reference: str, value: Any) -> str:
    if value is None:
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, int):
        # Excel preserves only 15 significant digits in numeric cells.
        if abs(value) <= 999_999_999_999_999:
            return f'<c r="{reference}" t="n"><v>{value}</v></c>'
        return _text_cell(reference, str(value))
    if isinstance(value, float):
        if math.isfinite(value):
            return f'<c r="{reference}" t="n"><v>{repr(value)}</v></c>'
        return _text_cell(reference, str(value))
    if isinstance(value, Decimal):
        # Decimal inputs can exceed Excel's precision or floating-point range.
        # Preserve those measurements as exact text rather than silently rounding
        # identifiers or turning an extremely small P value into numeric zero.
        if value.is_finite() and (
            value.is_zero()
            or (len(value.as_tuple().digits) <= 15 and -307 <= value.adjusted() <= 307)
        ):
            return f'<c r="{reference}" t="n"><v>{value}</v></c>'
        return _text_cell(reference, str(value))
    if isinstance(value, (datetime, date)):
        return _text_cell(reference, value.isoformat())
    if isinstance(value, (dict, list, tuple)):
        return _text_cell(
            reference,
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
    return _text_cell(reference, str(value))


def _text_cell(reference: str, value: str, *, style: int | None = None, guard: bool = True) -> str:
    safe_value = _safe_cell_text(value) if guard else _clean_xml_text(value)[:_MAX_CELL_CHARACTERS]
    style_attribute = f' s="{style}"' if style is not None else ""
    return (
        f'<c r="{reference}"{style_attribute} t="inlineStr">'
        f'<is><t xml:space="preserve">{escape(safe_value)}</t></is></c>'
    )


def _safe_cell_text(value: str) -> str:
    cleaned = _clean_xml_text(value)
    if cleaned.lstrip()[:1] in _FORMULA_PREFIXES:
        cleaned = "'" + cleaned
    return cleaned[:_MAX_CELL_CHARACTERS]


def _clean_xml_text(value: str) -> str:
    return "".join(character if _is_xml_character(character) else "\ufffd" for character in value)


def _is_xml_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        codepoint in {0x09, 0x0A, 0x0D}
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _column_letter(number: int) -> str:
    if number < 1 or number > _MAX_COLUMNS:
        raise ValueError(f"column number must be between 1 and {_MAX_COLUMNS}")
    letters: list[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _workbook_xml(sheet_name: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>'
        f'<sheet name={quoteattr(sheet_name)} sheetId="1" r:id="rId1"/>'
        '</sheets>'
        '</workbook>'
    )


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.'
    'spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/worksheets/sheet1.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.'
    'spreadsheetml.worksheet+xml"/>'
    '<Override PartName="/xl/styles.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.'
    'spreadsheetml.styles+xml"/>'
    '</Types>'
)

_ROOT_RELATIONSHIPS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships '
    'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/officeDocument" Target="xl/workbook.xml"/>'
    '</Relationships>'
)

_WORKBOOK_RELATIONSHIPS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships '
    'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    '<Relationship Id="rId2" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/styles" Target="styles.xml"/>'
    '</Relationships>'
)

_STYLES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<styleSheet '
    'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<fonts count="2">'
    '<font><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
    '<font><b/><sz val="11"/><name val="Calibri"/><family val="2"/></font>'
    '</fonts>'
    '<fills count="2">'
    '<fill><patternFill patternType="none"/></fill>'
    '<fill><patternFill patternType="gray125"/></fill>'
    '</fills>'
    '<borders count="1">'
    '<border><left/><right/><top/><bottom/><diagonal/></border>'
    '</borders>'
    '<cellStyleXfs count="1">'
    '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
    '</cellStyleXfs>'
    '<cellXfs count="2">'
    '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
    '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
    '</cellXfs>'
    '<cellStyles count="1">'
    '<cellStyle name="Normal" xfId="0" builtinId="0"/>'
    '</cellStyles>'
    '</styleSheet>'
)
