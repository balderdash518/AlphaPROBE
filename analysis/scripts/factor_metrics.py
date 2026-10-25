"""Per-factor metrics for the final sp500 pool (CPU only, so the running GPU job is not disturbed)."""
import json, re, sys
import numpy as np
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

if __name__ == "__main__":
    from alphagen_qlib.stock_data import StockData, FeatureType
    from alphagen.data.expression import Feature, Ref
    from alphagen.data.tree import ExpressionParser
    from alphagen.data.dimension import DimensionCalculator, _is_close
    from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr

    run_dir = sys.argv[1]; out = sys.argv[2]
    qlib = sys.argv[4]; inst = sys.argv[3]; dev = torch.device(sys.argv[5])
    ck = torch.load(run_dir + "/checkpoint.pt", map_location=sys.argv[5], weights_only=False)
    pool, graph = ck["pool"], ck["graph"]
    size = pool["size"]
    exprs = [re.sub(r"\d+(\.\d+)?e-\d+", lambda m: format(float(m.group()), "f"), str(e)) for e in pool["exprs"][:size]]
    nodes = pool["expr2node"][:size]
    records = graph._node_records

    def root_of(n):
        while n is not None and n.parent is not None:
            n = n.parent
        return n

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    parser, dim = ExpressionParser(), DimensionCalculator()
    periods = {"train": ("2010-01-01", "2020-12-31"), "valid": ("2021-01-01", "2022-06-30"), "test": ("2022-07-01", "2025-06-30")}
    vals = {}
    res = [dict(idx=i, expr=e) for i, e in enumerate(exprs)]
    for name, (a, b) in periods.items():
        data = StockData(instrument=inst, start_time=a, end_time=b, qlib_path=qlib, device=dev)
        tgt = target.evaluate(data)
        vals[name] = []
        for r in res:
            v = parser.parse(r["expr"]).evaluate(data)
            vals[name].append(v)
            ic = batch_pearsonr(v, tgt); ric = batch_spearmanr(v, tgt)
            r[f"{name}_ic"] = float(torch.nanmean(ic)); r[f"{name}_icir"] = float(torch.nanmean(ic) / (ic[~torch.isnan(ic)].std() + 1e-9))
            r[f"{name}_ric"] = float(torch.nanmean(ric)); r[f"{name}_ricir"] = float(torch.nanmean(ric) / (ric[~torch.isnan(ric)].std() + 1e-9))
            r[f"{name}_cov"] = float((~torch.isnan(v)).float().mean())
        print("done", name, flush=True)
    # mutual correlation on the test period (mean daily cross-sectional Pearson)
    n = len(res); M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = float(torch.nanmean(batch_pearsonr(vals["test"][i], vals["test"][j])))
    for i, r in enumerate(res):
        node = nodes[i]
        d = dim.dimension(r["expr"]).dimension
        r.update(dimension=d, dimensionless=bool(_is_close(d, 0)),
                 train_ic_pool=float(pool["single_ics"][i]), weight=float(pool["weights"][i]),
                 depth=getattr(node, "depth", None), times=getattr(node, "times", None),
                 topic=getattr(node, "topic", None), description=getattr(node, "description", None),
                 parent=(node.parent.payload.source if node is not None and node.parent is not None else None),
                 root=(root_of(node).payload.source if node is not None else None),
                 n_children=len(getattr(node, "children", []) or []),
                 max_corr_test=float(np.max(np.abs(np.delete(M[i], i)))),
                 most_corr_with=int(np.argmax(np.abs(np.where(np.arange(n) == i, 0, M[i])))))
    json.dump({"factors": res, "corr_test": M.tolist(), "graph_nodes": len(records),
               "graph_with_parent": sum(1 for x in records.values() if x.parent is not None),
               "graph_max_depth": max(x.depth for x in records.values())}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out)
