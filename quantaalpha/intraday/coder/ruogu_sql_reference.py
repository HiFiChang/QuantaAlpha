import os

import pandas as pd
import ruogu as rg
from ruogu import trade_cal


def fetch_factor_values() -> pd.DataFrame:
    token = os.environ.get("RUOGU_TOKEN")
    if token:
        rg.set_token(token)

    analysis_start = os.environ["INTRADAY_ANALYSIS_START"]
    analysis_end = os.environ["INTRADAY_ANALYSIS_END"]
    calc_start = trade_cal.pre_date(analysis_start)
    calc_dates = trade_cal.range_date_list(calc_start, analysis_end, check_open=False)

    frames = []
    for date in calc_dates:
        date_str = str(date).split()[0]
        sql = f"""
        with base as (
            select
                date_time,
                code,
                close,
                volume,
                amount,
                amount / nullIf(volume, 0) as vwap
            from stock_base.m1
            where date = toDate('{date_str}')
              and time_int >= Tit('09:30:00')
              and time_int <= Tit('15:00:00')
        )
        select
            toString(date_time) as datetime,
            code as instrument,
            (vwap - close) / (close + 1e-8) as factor_value
        from base
        order by date_time, code
        """
        day_df = rg.command_get_df(sql)
        if day_df.empty:
            continue
        frames.append(day_df)

    if not frames:
        raise ValueError("No remote intraday rows were fetched.")

    df = pd.concat(frames, axis=0, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str).str.zfill(6)
    df["factor_value"] = pd.to_numeric(df["factor_value"], errors="coerce")
    return df


def calculate_factor() -> pd.Series:
    df = fetch_factor_values()
    factor_value = (
        df.set_index(["datetime", "instrument"])
        .sort_index()["factor_value"]
        .astype(float)
    )
    return factor_value


if __name__ == "__main__":
    result = calculate_factor()
    result.to_hdf("result.h5", key="data")
