"""Approximate WorldQuant BRAIN-style simulation on our S&P 500 data (CPU).

Per alpha: daily cross-sectional values -> market neutralisation (demean) -> scale so
sum|w| = 1 -> truncate each stock at 8% -> re-scale. Delay 1: weights built from data
up to day d-1 earn the close-to-close return of day d+... (signal day d-1, held over day d).
Reports annualised Sharpe, turnover (sum|dw| per day), annual return on half the book
(long side), fitness = Sharpe * sqrt(|returns| / max(turnover, 0.125)).
These are approximations of BRAIN's definitions, not its exact engine.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))


def simulate(alpha: np.ndarray, ret: np.ndarray, sign: float = 1.0):
    a = sign * alpha
    T = a.shape[0]
    w = np.zeros_like(a)
    for t in range(T):
        x = a[t]
        m = np.isfinite(x)
        if m.sum() < 20:
            continue
        v = x[m] - x[m].mean()
        s = np.abs(v).sum()
        if s == 0:
            continue
        v = v / s
        for _ in range(5):  # truncation at 8% of book, re-scaled
            v = np.clip(v, -0.08, 0.08)
            v = v - v.mean()
            v = v / np.abs(v).sum()
        w[t, np.where(m)[0]] = v
    # delay 1: weight from day t-1's data applies to day t's return
    pnl = np.nansum(w[:-1] * np.nan_to_num(ret[1:]), axis=1)
    live = np.abs(w[:-1]).sum(axis=1) > 0
    pnl = pnl[live]
    turnover = np.abs(np.diff(w, axis=0)).sum(axis=1)[live].mean()
    sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)
    returns = pnl.mean() * 252 * 2  # on half the book (BRAIN-style)
    fitness = sharpe * np.sqrt(abs(returns) / max(turnover, 0.125))
    return dict(sharpe=float(sharpe), turnover=float(turnover), returns=float(returns), fitness=float(fitness),
                days=int(live.sum()))


if __name__ == "__main__":
    from alphagen_qlib.stock_data import StockData, FeatureType
    from alphagen.data.expression import Feature
    from alphagen.data.tree import ExpressionParser

    metrics = json.load(open(sys.argv[1], encoding="utf-8"))["factors"]
    out = sys.argv[2]
    start, end = sys.argv[3], sys.argv[4]
    qlib = "C:/Users/Administrator/.qlib/qlib_data/us_data_qlib"
    data = StockData(instrument="sp500", start_time=start, end_time=end, qlib_path=qlib, device=torch.device("cpu"))
    close = Feature(FeatureType.CLOSE).evaluate(data).numpy()
    ret = np.full_like(close, np.nan); ret[1:] = close[1:] / close[:-1] - 1
    ret = np.clip(ret, -0.5, 0.5)
    parser = ExpressionParser()
    jobs = [dict(name="user_brain", expr="Rank(TsCorr(Abs(Sub(Div($close,Ref($close,1)),1.0)),Div($volume,TsMean($volume,20)),20))", sign=1.0)]
    jobs += [dict(name=f"pool_{i}", expr=f["expr"], sign=float(np.sign(f["train_ic"])) or 1.0) for i, f in enumerate(metrics)]
    res = []
    for j in jobs:
        a = parser.parse(j["expr"]).evaluate(data).numpy()
        r = simulate(a, ret, j["sign"])
        r.update(j)
        res.append(r)
        print(f"{j['name']:<12} sharpe {r['sharpe']:+.2f} turnover {r['turnover']*100:5.1f}% returns {r['returns']*100:+5.1f}% fitness {r['fitness']:+.2f}", flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
