import json, sys, re, collections
J = lambda x: json.loads(x) if isinstance(x, str) else (x or {})
skel = lambda s: re.sub(r"\d+(\.\d+)?|%d", "N", re.sub(r"\s+", "", s or ""))
tok = lambda s: re.findall(r"[A-Za-z]+(?=\()|\$[a-z]+", s or "")
for path in sys.argv[1:]:
    rows = json.load(open(path, encoding="utf-8"))
    cls = collections.Counter(); ins = collections.Counter(); pct_d = 0; n = 0
    ops_new = collections.Counter()
    for e in (r for r in rows if r["name"] == "evolve-node"):
        p = J(e.get("input")).get("expression", ""); ps = skel(p); pt = set(tok(p))
        for c in J(e.get("output")).get("candidates", []):
            x = c["expression"]; xs = skel(x); n += 1; pct_d += "%d" in x
            if xs == ps: k = "window-only"
            elif ps and ps in xs: k = "wrap-parent"
            else:
                q = set(tok(x)); j = len(pt & q) / len(pt | q) if pt | q else 0
                k = "partial(J>=0.5)" if j >= 0.5 else "new-idea"
            cls[k] += 1; ins[k] += c["outcome"].startswith("inserted")
            if k == "new-idea": ops_new.update(set(tok(x)) - pt)
    print("==", path.split("/")[-1], "candidates", n, f"| use %d window search: {pct_d} ({pct_d / n:.0%})")
    for k in ("window-only", "wrap-parent", "partial(J>=0.5)", "new-idea"):
        print(f"  {k:<16} {cls[k]:>5} ({cls[k] / n:5.1%})  inserted {ins[k]:>3}  insert rate {ins[k] / max(cls[k], 1):5.1%}")
    print("  most common building blocks added by new-idea candidates:", ops_new.most_common(8))
