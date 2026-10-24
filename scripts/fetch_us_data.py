"""Build a Qlib-format S&P 500 daily dataset from Yahoo Finance.

Membership history comes from github.com/fja05680/sp500 (per-ticker start/end
dates), so `sp500` in the resulting instruments file is point-in-time rather
than today's list only.

Prices follow the Qlib Yahoo collector convention: OHLC are split/dividend
adjusted (raw * factor, factor = adjclose / close), volume is divided by
factor, and `$factor` is stored so raw prices can be recovered. Yahoo has no
VWAP, so `$vwap` is approximated by the adjusted typical price (H + L + C) / 3.

Usage:
    .venv/Scripts/python.exe scripts/fetch_us_data.py --target_dir C:/Users/Administrator/.qlib/qlib_data/us_data_qlib
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from yahooquery import Ticker

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
FAR_FUTURE = pd.Timestamp("2099-12-31")


def to_yahoo(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def fetch_membership(since: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(requests.get(MEMBERSHIP_URL, timeout=60).text))
    df["symbol"] = df["ticker"].map(to_yahoo)
    df["start"] = pd.to_datetime(df["start_date"]).clip(lower=since)
    df["end"] = pd.to_datetime(df["end_date"]).fillna(FAR_FUTURE)
    df = df[df["end"] >= since]
    return df[["symbol", "start", "end"]].sort_values(["symbol", "start"]).reset_index(drop=True)


def download(symbols: list[str], start: str, end: str, csv_dir: Path, batch: int = 50) -> list[str]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in symbols if not (csv_dir / f"{s}.csv").exists()]
    print(f"{len(symbols)} symbols, {len(symbols) - len(todo)} cached, downloading {len(todo)}")
    failed: list[str] = []
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for attempt in range(3):
            try:
                hist = Ticker(chunk, asynchronous=True, max_workers=8).history(
                    start=start, end=end, interval="1d", adj_ohlc=False)
                break
            except Exception as exc:  # network hiccups / rate limits
                print(f"  batch {i // batch} attempt {attempt + 1} failed: {exc}")
                time.sleep(10 * (attempt + 1))
        else:
            failed.extend(chunk)
            continue
        if not isinstance(hist, pd.DataFrame):
            hist = pd.DataFrame()
        got = set()
        if not hist.empty:
            hist = hist.reset_index()
            for sym, df in hist.groupby("symbol"):
                df = df.dropna(subset=["close", "adjclose"])
                df = df[df["close"] > 0]
                if df.empty:
                    continue
                df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])
                df = df.drop_duplicates("date").sort_values("date")
                factor = df["adjclose"] / df["close"]
                out = pd.DataFrame({
                    "date": df["date"].dt.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "open": df["open"] * factor,
                    "high": df["high"] * factor,
                    "low": df["low"] * factor,
                    "close": df["adjclose"],
                    "volume": df["volume"] / factor,
                    "factor": factor,
                })
                out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3
                out["change"] = out["close"].pct_change()
                out.to_csv(csv_dir / f"{sym}.csv", index=False)
                got.add(sym)
        failed.extend(s for s in chunk if s not in got)
        print(f"  {min(i + batch, len(todo))}/{len(todo)} done, {len(failed)} missing so far")
    return failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", default=str(Path.home() / ".qlib/qlib_data/us_data_qlib"))
    parser.add_argument("--csv_dir", default=str(ROOT / "data/us_yahoo_csv"))
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    target, csv_dir = Path(args.target_dir), Path(args.csv_dir)
    membership = fetch_membership(pd.Timestamp(args.start))
    print(f"S&P 500 membership: {membership['symbol'].nunique()} symbols, {len(membership)} intervals")

    symbols = sorted(set(membership["symbol"]) | {"^GSPC"})
    failed = download(symbols, args.start, args.end, csv_dir)
    if failed:
        print(f"No Yahoo data for {len(failed)} symbols (mostly delisted): {' '.join(failed[:40])}"
              + (" ..." if len(failed) > 40 else ""))

    subprocess.run([sys.executable, str(ROOT / "src/data_collection/qlib_dump_bin.py"), "dump_all",
                    "--csv_path", str(csv_dir), "--qlib_dir", str(target),
                    "--symbol_field_name", "symbol", "--date_field_name", "date",
                    "--exclude_fields", "symbol,date"], check=True)

    # Point-in-time sp500 universe, clipped to the data each symbol actually has.
    inst_dir = target / "instruments"
    have = pd.read_csv(inst_dir / "all.txt", sep="\t", header=None, names=["symbol", "first", "last"])
    have = have.set_index("symbol")
    rows = []
    for sym, start, end in membership.itertuples(index=False):
        if sym not in have.index:
            continue
        s = max(start, pd.Timestamp(have.at[sym, "first"]))
        e = min(end, pd.Timestamp(have.at[sym, "last"]))
        if s <= e:
            rows.append((sym, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    pd.DataFrame(rows).to_csv(inst_dir / "sp500.txt", sep="\t", header=False, index=False)
    print(f"Wrote {inst_dir / 'sp500.txt'} with {len({r[0] for r in rows})} symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
