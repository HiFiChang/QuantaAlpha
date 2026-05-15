from __future__ import annotations

from dataclasses import dataclass, field

from pyparsing import Forward, Optional, ParseException, ParserElement, Regex, Suppress, Word
from pyparsing import alphanums, alphas, delimitedList, infixNotation, opAssoc, oneOf


ParserElement.enablePackrat()


class IntradaySqlExpressionError(ValueError):
    """Raised when an intraday DSL expression cannot be compiled."""


@dataclass(frozen=True)
class SqlExpr:
    sql: str
    fields: frozenset[str] = field(default_factory=frozenset)
    has_window: bool = False


FIELD_SQL = {
    "$open": "open",
    "$close": "close",
    "$high": "high",
    "$low": "low",
    "$volume": "volume",
    "$money": "money",
    "$vwap": "vwap",
    "$return": "return_value",
    "$bid1": "bid1",
    "$ask1": "ask1",
    "$bidv1": "bidv1",
    "$askv1": "askv1",
    "$total_volume": "total_volume",
    "$total_value": "total_value",
    "$spread": "spread",
    "$relative_spread": "relative_spread",
    "$depth_imbalance_1": "depth_imbalance_1",
}
for _level in range(2, 11):
    FIELD_SQL[f"$bid{_level}"] = f"bid{_level}"
    FIELD_SQL[f"$ask{_level}"] = f"ask{_level}"
    FIELD_SQL[f"$bidv{_level}"] = f"bidv{_level}"
    FIELD_SQL[f"$askv{_level}"] = f"askv{_level}"
FIELD_SQL.update(
    {
        "$last": "last",
        "$ztprice": "ztprice",
        "$dtprice": "dtprice",
        "$num_trades": "num_trades",
    }
)

M1_FIELDS = {"$open", "$close", "$high", "$low", "$volume", "$money", "$vwap", "$return"}
TK_DIRECT_FIELDS = (
    [f"$bid{level}" for level in range(1, 11)]
    + [f"$ask{level}" for level in range(1, 11)]
    + [f"$bidv{level}" for level in range(1, 11)]
    + [f"$askv{level}" for level in range(1, 11)]
    + ["$total_volume", "$total_value", "$last", "$ztprice", "$dtprice", "$num_trades"]
)
TK_FIELDS = {
    *TK_DIRECT_FIELDS,
    "$spread",
    "$relative_spread",
    "$depth_imbalance_1",
}
def _window(rows: int) -> str:
    if rows < 1:
        raise IntradaySqlExpressionError(f"Window must be positive, got {rows}")
    return f"PARTITION BY code, date ORDER BY date_time ROWS BETWEEN {rows - 1} PRECEDING AND CURRENT ROW"


def _to_int_arg(arg: SqlExpr, func_name: str) -> int:
    try:
        value = int(float(arg.sql))
    except ValueError as exc:
        raise IntradaySqlExpressionError(f"{func_name} window argument must be a number") from exc
    if value < 1:
        raise IntradaySqlExpressionError(f"{func_name} window argument must be >= 1")
    return value


def _unary_func(func_name: str, arg: SqlExpr) -> SqlExpr:
    if func_name == "ABS":
        return SqlExpr(f"abs({arg.sql})", arg.fields, arg.has_window)
    if func_name == "LOG":
        return SqlExpr(f"log(abs({arg.sql}) + 1)", arg.fields, arg.has_window)
    if func_name == "SQRT":
        return SqlExpr(f"sqrt(greatest({arg.sql}, 0))", arg.fields, arg.has_window)
    if func_name == "SIGN":
        return SqlExpr(f"sign({arg.sql})", arg.fields, arg.has_window)
    if func_name == "EXP":
        return SqlExpr(f"exp({arg.sql})", arg.fields, arg.has_window)
    if func_name == "INV":
        return SqlExpr(f"(1 / nullIf({arg.sql}, 0))", arg.fields, arg.has_window)
    if func_name == "FLOOR":
        return SqlExpr(f"floor({arg.sql})", arg.fields, arg.has_window)
    raise IntradaySqlExpressionError(f"Unsupported unary function: {func_name}")


def _binary_math_func(func_name: str, args: list[SqlExpr]) -> SqlExpr:
    if len(args) != 2:
        raise IntradaySqlExpressionError(f"{func_name} expects exactly 2 arguments")
    fields = args[0].fields | args[1].fields
    has_window = args[0].has_window or args[1].has_window
    if func_name == "POW":
        return SqlExpr(f"pow({args[0].sql}, {args[1].sql})", fields, has_window)
    if func_name == "MAX":
        return SqlExpr(f"greatest({args[0].sql}, {args[1].sql})", fields, has_window)
    if func_name == "MIN":
        return SqlExpr(f"least({args[0].sql}, {args[1].sql})", fields, has_window)
    raise IntradaySqlExpressionError(f"Unsupported binary math function: {func_name}")


def _conditional_func(func_name: str, args: list[SqlExpr]) -> SqlExpr:
    if func_name == "WHERE":
        if len(args) != 3:
            raise IntradaySqlExpressionError("WHERE expects exactly 3 arguments: condition, true_value, false_value")
        return SqlExpr(
            f"if({args[0].sql}, {args[1].sql}, {args[2].sql})",
            args[0].fields | args[1].fields | args[2].fields,
            args[0].has_window or args[1].has_window or args[2].has_window,
        )
    if func_name == "FILTER":
        if len(args) != 2:
            raise IntradaySqlExpressionError("FILTER expects exactly 2 arguments: value, condition")
        return SqlExpr(
            f"if({args[1].sql}, {args[0].sql}, 0)",
            args[0].fields | args[1].fields,
            args[0].has_window or args[1].has_window,
        )
    raise IntradaySqlExpressionError(f"Unsupported conditional function: {func_name}")


def _window_func(func_name: str, args: list[SqlExpr]) -> SqlExpr:
    if len(args) != 2:
        raise IntradaySqlExpressionError(f"{func_name} expects exactly 2 arguments")
    source = args[0]
    rows = _to_int_arg(args[1], func_name)
    if func_name == "DELAY":
        return SqlExpr(
            f"lagInFrame({source.sql}, {rows}) OVER (PARTITION BY code, date ORDER BY date_time)",
            source.fields,
            True,
        )
    if func_name == "DELTA":
        return SqlExpr(
            f"(({source.sql}) - lagInFrame({source.sql}, {rows}) OVER (PARTITION BY code, date ORDER BY date_time))",
            source.fields,
            True,
        )
    if func_name == "TS_MEAN":
        return SqlExpr(f"avg({source.sql}) OVER ({_window(rows)})", source.fields, True)
    if func_name == "TS_SUM":
        return SqlExpr(f"sum({source.sql}) OVER ({_window(rows)})", source.fields, True)
    if func_name == "TS_STD":
        return SqlExpr(f"stddevSamp({source.sql}) OVER ({_window(rows)})", source.fields, True)
    if func_name == "TS_VAR":
        return SqlExpr(f"varSamp({source.sql}) OVER ({_window(rows)})", source.fields, True)
    if func_name == "TS_MIN":
        return SqlExpr(f"min({source.sql}) OVER ({_window(rows)})", source.fields, True)
    if func_name == "TS_MAX":
        return SqlExpr(f"max({source.sql}) OVER ({_window(rows)})", source.fields, True)
    if func_name == "TS_PCTCHANGE":
        lag_sql = f"lagInFrame({source.sql}, {rows}) OVER (PARTITION BY code, date ORDER BY date_time)"
        return SqlExpr(f"(({source.sql}) / nullIf({lag_sql}, 0) - 1)", source.fields, True)
    if func_name == "TS_ZSCORE":
        mean_sql = f"avg({source.sql}) OVER ({_window(rows)})"
        std_sql = f"stddevSamp({source.sql}) OVER ({_window(rows)})"
        return SqlExpr(f"(({source.sql}) - ({mean_sql})) / nullIf(({std_sql}), 0)", source.fields, True)
    raise IntradaySqlExpressionError(f"Unsupported time-series function: {func_name}")


def _conditional_window_func(func_name: str, args: list[SqlExpr]) -> SqlExpr:
    if func_name == "COUNT":
        if len(args) != 2:
            raise IntradaySqlExpressionError("COUNT expects exactly 2 arguments: condition, window")
        if args[0].has_window:
            raise IntradaySqlExpressionError("COUNT condition cannot contain another window function")
        rows = _to_int_arg(args[1], func_name)
        return SqlExpr(f"sum(if({args[0].sql}, 1, 0)) OVER ({_window(rows)})", args[0].fields, True)
    if func_name == "SUMIF":
        if len(args) != 3:
            raise IntradaySqlExpressionError("SUMIF expects exactly 3 arguments: value, window, condition")
        if args[0].has_window or args[2].has_window:
            raise IntradaySqlExpressionError("SUMIF value and condition cannot contain another window function")
        rows = _to_int_arg(args[1], func_name)
        return SqlExpr(
            f"sum(if({args[2].sql}, {args[0].sql}, 0)) OVER ({_window(rows)})",
            args[0].fields | args[2].fields,
            True,
        )
    raise IntradaySqlExpressionError(f"Unsupported conditional window function: {func_name}")


def _compile_function(tokens) -> SqlExpr:
    func_name = str(tokens[0]).upper()
    args = [arg for arg in tokens[1:] if isinstance(arg, SqlExpr)]

    if func_name in {"ABS", "LOG", "SQRT", "SIGN", "EXP", "INV", "FLOOR"}:
        if len(args) != 1:
            raise IntradaySqlExpressionError(f"{func_name} expects exactly 1 argument")
        return _unary_func(func_name, args[0])
    if func_name in {"POW", "MAX", "MIN"}:
        return _binary_math_func(func_name, args)
    if func_name in {"WHERE", "FILTER"}:
        return _conditional_func(func_name, args)
    if func_name in {"COUNT", "SUMIF"}:
        return _conditional_window_func(func_name, args)
    if func_name in {
        "DELAY",
        "DELTA",
        "TS_MEAN",
        "TS_SUM",
        "TS_STD",
        "TS_VAR",
        "TS_MIN",
        "TS_MAX",
        "TS_ZSCORE",
        "TS_PCTCHANGE",
    }:
        return _window_func(func_name, args)
    if func_name in {"RANK", "ZSCORE", "MEAN", "STD"}:
        if len(args) != 1:
            raise IntradaySqlExpressionError(f"{func_name} expects exactly 1 argument")
        return SqlExpr(f"{func_name}({args[0].sql})", args[0].fields, True)
    raise IntradaySqlExpressionError(f"Unsupported function: {func_name}")


def _compile_binary(tokens) -> SqlExpr:
    values = tokens[0].asList()
    expr = values[0]
    for op, rhs in zip(values[1::2], values[2::2], strict=False):
        if op == "+":
            expr = SqlExpr(f"(({expr.sql}) + ({rhs.sql}))", expr.fields | rhs.fields, expr.has_window or rhs.has_window)
        elif op == "-":
            expr = SqlExpr(f"(({expr.sql}) - ({rhs.sql}))", expr.fields | rhs.fields, expr.has_window or rhs.has_window)
        elif op == "*":
            expr = SqlExpr(f"(({expr.sql}) * ({rhs.sql}))", expr.fields | rhs.fields, expr.has_window or rhs.has_window)
        elif op == "/":
            expr = SqlExpr(f"(({expr.sql}) / nullIf(({rhs.sql}), 0))", expr.fields | rhs.fields, expr.has_window or rhs.has_window)
        else:
            raise IntradaySqlExpressionError(f"Unsupported operator: {op}")
    return expr


def _compile_comparison(tokens) -> SqlExpr:
    values = tokens[0].asList()
    expr = values[0]
    op_map = {"==": "=", "!=": "!="}
    for op, rhs in zip(values[1::2], values[2::2], strict=False):
        expr = SqlExpr(
            f"(({expr.sql}) {op_map.get(op, op)} ({rhs.sql}))",
            expr.fields | rhs.fields,
            expr.has_window or rhs.has_window,
        )
    return expr


def _compile_logical(tokens) -> SqlExpr:
    values = tokens[0].asList()
    expr = values[0]
    for op, rhs in zip(values[1::2], values[2::2], strict=False):
        sql_op = "AND" if op in {"&&", "&"} else "OR"
        expr = SqlExpr(f"(({expr.sql}) {sql_op} ({rhs.sql}))", expr.fields | rhs.fields, expr.has_window or rhs.has_window)
    return expr


def _compile_field(tokens) -> SqlExpr:
    field_name = tokens[0]
    if field_name not in FIELD_SQL:
        raise IntradaySqlExpressionError(f"Unsupported field: {field_name}")
    return SqlExpr(FIELD_SQL[field_name], frozenset({field_name}))


def _build_parser():
    expr = Forward()
    number = Regex(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?").setParseAction(lambda t: SqlExpr(t[0]))
    field_expr = Regex(r"\$[A-Za-z_][A-Za-z0-9_]*").setParseAction(_compile_field)
    func_name = Word(alphas + "_", alphanums + "_")
    function_call = (
        func_name
        + Suppress("(")
        + Optional(delimitedList(expr), default=[])
        + Suppress(")")
    ).setParseAction(_compile_function)
    parens = Suppress("(") + expr + Suppress(")")
    atom = function_call | field_expr | number | parens
    expr <<= infixNotation(
        atom,
        [
            (
                oneOf("+ -"),
                1,
                opAssoc.RIGHT,
                lambda t: SqlExpr(f"(-1 * ({t[0][1].sql}))", t[0][1].fields, t[0][1].has_window) if t[0][0] == "-" else t[0][1],
            ),
            (oneOf("* /"), 2, opAssoc.LEFT, _compile_binary),
            (oneOf("+ -"), 2, opAssoc.LEFT, _compile_binary),
            (oneOf("> < >= <= == !="), 2, opAssoc.LEFT, _compile_comparison),
            (oneOf("&& &"), 2, opAssoc.LEFT, _compile_logical),
            (oneOf("|| |"), 2, opAssoc.LEFT, _compile_logical),
        ],
    )
    return expr


_PARSER = _build_parser()


def compile_factor_sql_expr(expression: str) -> str:
    return compile_factor_sql(expression).sql


def compile_factor_sql(expression: str) -> SqlExpr:
    expression = (expression or "").strip()
    if not expression:
        raise IntradaySqlExpressionError("Empty expression")
    try:
        parsed = _PARSER.parseString(expression, parseAll=True)[0]
    except ParseException as exc:
        raise IntradaySqlExpressionError(f"Failed to parse expression: {exc}") from exc
    if not isinstance(parsed, SqlExpr):
        raise IntradaySqlExpressionError(f"Failed to compile expression: {expression}")
    return parsed


def build_m1_factor_sql(
    expression: str,
    calc_start: str,
    analysis_start: str,
    analysis_end: str,
    calc_time_start: str = "09:30:00",
    calc_time_end: str = "15:00:00",
    output_time_start: str = "09:30:00",
    output_time_end: str = "15:00:00",
) -> str:
    compiled = compile_factor_sql(expression)
    factor_sql = compiled.sql
    required_fields = compiled.fields
    final_sql = "factor_raw"

    cross_section_func = None
    for func_name in ("RANK", "ZSCORE", "MEAN", "STD"):
        prefix = f"{func_name}("
        if factor_sql.startswith(prefix) and factor_sql.endswith(")"):
            cross_section_func = func_name
            factor_sql = factor_sql[len(prefix):-1]
            break
    if cross_section_func == "RANK":
        final_sql = "percent_rank() OVER (PARTITION BY date_time ORDER BY factor_raw)"
    elif cross_section_func == "ZSCORE":
        final_sql = (
            "(factor_raw - avg(factor_raw) OVER (PARTITION BY date_time)) / "
            "nullIf(stddevSamp(factor_raw) OVER (PARTITION BY date_time), 0)"
        )
    elif cross_section_func == "MEAN":
        final_sql = "avg(factor_raw) OVER (PARTITION BY date_time)"
    elif cross_section_func == "STD":
        final_sql = "stddevSamp(factor_raw) OVER (PARTITION BY date_time)"
    if any(marker in factor_sql for marker in ("RANK(", "ZSCORE(", "MEAN(", "STD(")):
        raise IntradaySqlExpressionError("Cross-sectional functions are only supported as the outermost expression function")

    needs_tk = bool(required_fields & TK_FIELDS)
    extra_selects: list[str] = []
    tk_cte = ""
    tk_join = ""
    if needs_tk:
        tk_direct_columns = [FIELD_SQL[field] for field in TK_DIRECT_FIELDS]
        extra_selects.extend([f"tk_bar.{column} AS {column}" for column in tk_direct_columns])
        extra_selects.extend(
            [
                "tk_bar.spread AS spread",
                "tk_bar.relative_spread AS relative_spread",
                "tk_bar.depth_imbalance_1 AS depth_imbalance_1",
            ]
        )
        tk_argmax_select_sql = ",\n        ".join(
            f"argMax({column}, event_time) AS {column}" for column in tk_direct_columns
        )
        tk_inner_select_sql = ",\n            ".join(tk_direct_columns)
        tk_bar_select_sql = ",\n        ".join(tk_direct_columns)
        tk_cte = f""",
tk_last AS (
    SELECT
        bar_time AS date_time,
        date,
        code,
        {tk_argmax_select_sql}
    FROM (
        SELECT
            toStartOfMinute(date_time) AS bar_time,
            date_time AS event_time,
            date,
            code,
            {tk_inner_select_sql}
        FROM stock_base.tk
        WHERE date >= toDate('{calc_start}')
          AND date <= toDate('{analysis_end}')
          AND time_int >= Tit('{calc_time_start}')
          AND time_int <= Tit('{calc_time_end}')
    )
    GROUP BY bar_time, date, code
),
tk_bar AS (
    SELECT
        date_time,
        date,
        code,
        {tk_bar_select_sql},
        ask1 - bid1 AS spread,
        (ask1 - bid1) / nullIf((ask1 + bid1) / 2, 0) AS relative_spread,
        (bidv1 - askv1) / nullIf(bidv1 + askv1, 0) AS depth_imbalance_1
    FROM tk_last
)"""
        tk_join = """
    LEFT JOIN tk_bar USING (date_time, date, code)"""

    extra_select_sql = ""
    if extra_selects:
        extra_select_sql = ",\n        " + ",\n        ".join(extra_selects)

    return f"""
WITH m1_base AS (
    SELECT
        date_time,
        time_int,
        date,
        code,
        open,
        close,
        high,
        low,
        volume,
        amount AS money,
        amount / nullIf(volume, 0) AS vwap,
        (close / nullIf(lagInFrame(close, 1) OVER (PARTITION BY code, date ORDER BY date_time), 0) - 1) AS return_value
    FROM stock_base.m1
    WHERE date >= toDate('{calc_start}')
      AND date <= toDate('{analysis_end}')
      AND time_int >= Tit('{calc_time_start}')
      AND time_int <= Tit('{calc_time_end}')
){tk_cte},
base AS (
    SELECT
        m1_base.date_time AS date_time,
        m1_base.time_int AS time_int,
        m1_base.date AS date,
        m1_base.code AS code,
        m1_base.open AS open,
        m1_base.close AS close,
        m1_base.high AS high,
        m1_base.low AS low,
        m1_base.volume AS volume,
        m1_base.money AS money,
        m1_base.vwap AS vwap,
        m1_base.return_value AS return_value{extra_select_sql}
    FROM m1_base{tk_join}
),
calc AS (
    SELECT
        date_time,
        time_int,
        date,
        code,
        {factor_sql} AS factor_raw
    FROM base
),
factor_output AS (
    SELECT
        date_time,
        code,
        {final_sql} AS factor_value
    FROM calc
    WHERE date >= toDate('{calc_start}')
      AND date <= toDate('{analysis_end}')
      AND time_int >= Tit('{output_time_start}')
      AND time_int <= Tit('{output_time_end}')
)
SELECT
    toString(date_time) AS datetime,
    code AS instrument,
    factor_value
FROM factor_output
ORDER BY datetime, instrument
"""
