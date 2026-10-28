"""Evaluate the paper's appendix factors and my CSI300 pool on the same data, without qlib.
Reads qlib .bin files directly and mimics StockData (members-only spans, 100 backtrack / 30 future days)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path(sys.argv[1]); Q = Path(sys.argv[2]); OUT = sys.argv[3]
sys.path.insert(0, str(REPO / "src"))
from alphagen_qlib.stock_data import StockData, FeatureType
from alphagen.data.expression import Feature, Ref
from alphagen.data.tree import ExpressionParser
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr

PAPER = [
    "Div(Sub(Less($open,$close),$low),$open)",
    "TsCorr(Sub($close,$open),Sub($low,TsMin($low,5)),5)",
    "TsCorr(Sub($close,$open),TsRank(Sub($low,TsMin($low,5)),5),5)",
    "TsCorr(TsDelta($close,1),TsMinDiff($low,5),5)",
    "TsCorr(Sub($close,$open),Div(Sub($low,TsMin($low,5)),TsStd($low,5)),5)",
    "TsCorr(TsPctChange($close,5),TsMinDiff($low,5),5)",
    "TsCorr(Sub($close,$open),TsRank(Div(Sub($low,TsMin($low,5)),TsStd($low,5)),5),5)",
    "TsCorr(TsDelta($close,5),TsMaxDiff($low,5),5)",
    "TsCorr(TsDelta($close,5),TsDelta($low,5),5)",
    "TsCorr(TsPctChange($close,5),TsPctChange($low,5),5)",
    "TsCorr(TsDelta($close,5),Sub(TsMax($high,5),$close),5)",
    "TsCorr(TsDelta($close,5),TsDelta($vwap,5),5)",
    "TsCorr(TsPctChange($close,5),TsPctChange($vwap,5),5)",
    "TsCorr(TsDelta($close,5),TsDelta(Sub($high,$low),5),5)",
    "TsCorr(TsDelta($close,5),TsDelta($volume,5),5)",
    "TsCorr(TsDelta($close,5),TsPctChange($low,5),5)",
    "TsCorr(TsPctChange($close,5),Sub($high,$low),5)",
    "TsCorr(Sub($close,Ref($close,1)),Sub($vwap,Ref($vwap,1)),5)",
    "Rank(TsCorr(TsDelta($close,5),TsDelta($vwap,5),5))",
    "TsCorr(TsSkew($close,5),TsKurt($vwap,5),5)",
    "TsCorr(TsPctChange($close,5),TsDelta($low,5),5)",
    "TsCorr(TsDelta($close,5),TsPctChange($volume,5),5)",
    "TsCorr(TsPctChange($close,5),TsPctChange($volume,5),5)",
    "Mul(TsCorr(TsDelta($close,5),TsDelta($vwap,5),5),TsSkew(TsDelta($close,5),5))",
    "Rank(TsCorr(TsPctChange($close,5),TsPctChange($vwap,5),5))",
    "Rank(TsCorr(TsPctChange($close,5),TsPctChange($low,5),5))",
    "Rank(TsCorr(TsPctChange($close,5),TsDelta($volume,5),5))",
    "Sub(Rank(TsCorr(TsPctChange($close,5),TsPctChange($vwap,5),5)),Rank(TsCorr(TsPctChange($close,5),TsDelta($volume,5),5)))",
]
PAPER_ICIR = [.2391,.2652,.0947,.0611,.1179,.1319,.1614,.1507,.1998,.1632,.2639,.1210,.0719,.2813,.2639,.0935,.2746,
              .1252,.1169,.1218,.1452,.1760,.2681,.1015,.0974,.1731,.2765,.2281]

cal = pd.to_datetime(pd.read_csv(Q / "calendars/day.txt", header=None)[0]).values
spans = pd.read_csv(Q / "instruments/csi300.txt", sep="\t", header=None, names=["code", "s", "e"])
spans["s"] = pd.to_datetime(spans["s"]); spans["e"] = pd.to_datetime(spans["e"])
FIELDS = ["open", "close", "high", "low", "volume", "vwap"]  # FeatureType order


def load(start, end, back=100, fut=30):
    si = cal.searchsorted(np.datetime64(start)); ei = cal.searchsorted(np.datetime64(end))
    if cal[ei] != np.datetime64(end): ei -= 1
    lo, hi = si - back, ei + fut
    dates = cal[lo:hi + 1]
    sp = spans[(spans.s <= dates[-1]) & (spans.e >= dates[0])]
    codes = sorted(sp.code.unique())
    arr = np.full((len(dates), len(FIELDS), len(codes)), np.nan, np.float32)
    for j, c in enumerate(codes):
        mask = np.zeros(len(dates), bool)
        for _, r in sp[sp.code == c].iterrows():
            mask |= (dates >= np.datetime64(r.s)) & (dates <= np.datetime64(r.e))
        for k, f in enumerate(FIELDS):
            p = Q / "features" / c.lower() / f"{f}.day.bin"
            if not p.exists(): continue
            raw = np.fromfile(p, "<f4"); s0 = int(raw[0]); v = raw[1:]
            idx = np.arange(lo, hi + 1) - s0
            ok = (idx >= 0) & (idx < len(v))
            col = np.full(len(dates), np.nan, np.float32); col[ok] = v[idx[ok]]
            col[~mask] = np.nan
            arr[:, k, j] = col
    d = object.__new__(StockData)
    d.data = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32).reshape(arr.shape); d.max_backtrack_days = back; d.max_future_days = fut
    d.device = torch.device("cpu"); d._features = list(FeatureType)
    return d


mine = json.load(open(REPO / "analysis/results/csi300_factor_metrics.json", encoding="utf-8"))["factors"]
MINE = [f["expr"] for f in mine]
parser = ExpressionParser()
close = Feature(FeatureType.CLOSE); target = Ref(close, -20) / close - 1
periods = {"train": ("2010-01-01", "2020-12-31"), "valid": ("2021-01-01", "2022-06-30"), "test": ("2022-07-01", "2025-06-30")}

res = {"paper": [dict(id=i + 1, expr=e, paper_icir=PAPER_ICIR[i]) for i, e in enumerate(PAPER)],
       "mine": [dict(id=i + 1, expr=e, stored_test_ic=f["test_ic"]) for i, (e, f) in enumerate(zip(MINE, mine))]}
test_vals = {}
for name, (a, b) in periods.items():
    data = load(a, b); tgt = target.evaluate(data)
    print("loaded", name, tuple(data.data.shape), flush=True)
    for grp, exprs in (("paper", PAPER), ("mine", MINE)):
        for r, e in zip(res[grp], exprs):
            v = parser.parse(e).evaluate(data)
            v = torch.where(torch.isfinite(v), v, torch.nan)
            ic = batch_pearsonr(v, tgt); ric = batch_spearmanr(v, tgt)
            ici = ic[~torch.isnan(ic)]; rici = ric[~torch.isnan(ric)]
            r[f"{name}_ic"] = float(ici.mean()); r[f"{name}_icir"] = float(ici.mean() / (ici.std() + 1e-9))
            r[f"{name}_ric"] = float(rici.mean()); r[f"{name}_ricir"] = float(rici.mean() / (rici.std() + 1e-9))
            if name == "test": test_vals[(grp, r["id"])] = v
            print(name, grp, r["id"], round(r[f"{name}_ic"], 4), flush=True)
    print("done", name, flush=True)

# test-period cross-sectional correlation of each paper factor with my pool and with other paper factors
for r in res["paper"]:
    v = test_vals[("paper", r["id"])]
    cm = [abs(float(torch.nanmean(batch_pearsonr(v, test_vals[("mine", m["id"])])))) for m in res["mine"]]
    r["max_corr_mine"] = max(cm); r["most_corr_mine"] = int(np.argmax(cm)) + 1
    cp = [abs(float(torch.nanmean(batch_pearsonr(v, test_vals[("paper", o["id"])])))) for o in res["paper"] if o["id"] != r["id"]]
    r["max_corr_paper"] = max(cp)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT)
