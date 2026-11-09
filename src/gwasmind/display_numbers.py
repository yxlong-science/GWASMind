"""Reader-facing precision; scientific values and admission checks stay unchanged."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Any


def three_significant_figures(value: Any) -> str:
    """Round a finite decimal value half up, retaining three significant figures."""
    if isinstance(value, bool) or value is None:
        return "unavailable"
    try:
        number = Decimal(str(value))
        if not number.is_finite() or abs(number.adjusted()) > 10000:
            return "unavailable"
        if number.is_zero():
            return "0"
        with localcontext() as context:
            context.prec = 40
            rounded = number.quantize(Decimal(1).scaleb(number.adjusted() - 2), rounding=ROUND_HALF_UP)
            # A carry across a power of ten must not introduce a fourth digit.
            rounded = rounded.quantize(Decimal(1).scaleb(rounded.adjusted() - 2), rounding=ROUND_HALF_UP)
        return (format(rounded, 'f') if -3 <= rounded.adjusted() < 3
                else format(rounded, '.2E').replace('E', 'e'))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return "unavailable"
