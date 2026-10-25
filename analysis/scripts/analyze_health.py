import json, sys, collections
rows = json.load(open(sys.argv[1], encoding="utf-8"))
by_trace = collections.defaultdict(list)
for r in rows:
    by_trace[r["traceId"]].append(r)
roots = sorted([r for r in rows if r["name"] == "alphaprobe-iteration"], key=lambda r: r["startTime"])
print("iter nodes gens parse_fail cands | outcome counts | errors")
tot = collections.Counter()
levels = collections.Counter(r.get("level") for r in rows)
for i, root in enumerate(roots, 1):
    obs = by_trace[root["traceId"]]
    ev = [o for o in obs if o["name"] == "evolve-node"]
    gens = [o for o in obs if o["type"] == "GENERATION"]
    oc = collections.Counter(); errs = collections.Counter(); pf = 0
    for e in ev:
        out = e.get("output") or {}
        if isinstance(out, str):
            out = json.loads(out)
        pf += int(out.get("parse_failed", False))
        for c in out.get("candidates", []):
            oc[c["outcome"]] += 1
            if c.get("error"):
                errs[c["error"][:60]] += 1
    tot += oc
    o = root.get("output") or {}
    if isinstance(o, str): o = json.loads(o)
    print(f"{i:>4} {len(ev):>5} {len(gens):>4} {pf:>10} {sum(oc.values()):>5} | "
          + ", ".join(f"{k}={v}" for k, v in sorted(oc.items())) + f" | pool={o.get('pool_size')} best_ic={o.get('best_single_ic')}"
          + (f" | errs={dict(errs)}" if errs else ""))
print("TOTAL", dict(tot), "sum", sum(tot.values()))
print("levels", dict(levels))
fin = [o for o in rows if o.get("finish_reason") or (isinstance(o.get("metadata"), dict) and o["metadata"].get("finish_reason"))]
fr = collections.Counter((o["metadata"] or {}).get("finish_reason") for o in rows if o["type"] == "GENERATION")
print("finish_reason", dict(fr))
