import json, sys, re, collections
import numpy as np
rows = json.load(open(sys.argv[1], encoding="utf-8"))
J = lambda x: json.loads(x) if isinstance(x, str) else (x or {})
by_trace = collections.defaultdict(list)
for r in rows: by_trace[r["traceId"]].append(r)
roots = sorted([r for r in rows if r["name"] == "alphaprobe-iteration"], key=lambda r: r["startTime"])
norm = lambda s: re.sub(r"\s+", "", s or "")
tok = lambda s: re.findall(r"[A-Za-z]+|\$[a-z]+", s or "")

# 1) why candidates are rejected (below_threshold): weak IC vs failing "beat the parent ICIR"
weak = notbeat = 0; parent_icirs = []; cand_icirs = []
for r in rows:
    if r["name"] != "evolve-node": continue
    out = J(r.get("output")); par = out.get("parent") or {}
    picir = abs(par.get("icir") or 0); parent_icirs.append(picir)
    for c in out.get("candidates", []):
        if c["outcome"] != "below_threshold": continue
        ic, icir = abs(c.get("ic") or 0), abs(c.get("icir") or 0); cand_icirs.append(icir)
        if ic <= 0.006: weak += 1
        elif icir <= picir: notbeat += 1
print(f"below_threshold: |IC|<=0.006: {weak}, IC ok but |ICIR|<=parent |ICIR|: {notbeat}")
print(f"parent |ICIR| median {np.median(parent_icirs):.3f}; rejected-candidate |ICIR| median {np.median(cand_icirs):.3f}")

# 2) retrieval: which parents get picked, how often, how deep
picks = collections.Counter(); depth_by_iter = []; seen = set(); new_parents = []
for i, root in enumerate(roots, 1):
    ret = [o for o in by_trace[root["traceId"]] if o["name"] == "retrieve-parents"][0]
    nodes = [n for n in J(ret.get("output")) or [] if n]
    exprs = [norm(n["expression"]) for n in nodes]
    picks.update(exprs)
    depth_by_iter.append(np.mean([n["depth"] for n in nodes]))
    new_parents.append(sum(e not in seen for e in exprs)); seen.update(exprs)
print("distinct parents over 20 iterations:", len(picks), "of", sum(picks.values()), "picks")
print("most re-picked parents:", [(e[:60], k) for e, k in picks.most_common(5)])
print("new (never picked before) parents per iteration:", new_parents)
print("mean parent depth per iteration:", [round(d, 2) for d in depth_by_iter])

# 3) novelty of generated candidates
allc = []; per_iter_dup = []; seen_c = set()
ops_early, ops_late = collections.Counter(), collections.Counter()
feat_early, feat_late = collections.Counter(), collections.Counter()
for i, root in enumerate(roots, 1):
    dup = 0
    for e in [o for o in by_trace[root["traceId"]] if o["name"] == "evolve-node"]:
        for c in J(e.get("output")).get("candidates", []):
            x = norm(c["expression"]).replace("%d", "N"); x = re.sub(r"\d+", "N", x)
            dup += x in seen_c; seen_c.add(x); allc.append(x)
            t = tok(c["expression"])
            (ops_early if i <= 5 else ops_late if i > 15 else collections.Counter()).update(w for w in t if not w.startswith("$"))
            (feat_early if i <= 5 else feat_late if i > 15 else collections.Counter()).update(w for w in t if w.startswith("$"))
    per_iter_dup.append(dup)
print("candidates", len(allc), "structurally distinct (windows ignored):", len(set(allc)))
print("repeats of an earlier structure per iteration:", per_iter_dup)
f = lambda c: ", ".join(f"{k}:{v}" for k, v in c.most_common(8))
print("ops iters 1-5 :", f(ops_early)); print("ops iters 16-20:", f(ops_late))
print("features 1-5  :", f(feat_early)); print("features 16-20:", f(feat_late))

# 4) child vs parent: how far does a child move from its parent (token Jaccard)
sims = []
for e in [o for o in rows if o["name"] == "evolve-node"]:
    p = set(tok(J(e.get("input")).get("expression")))
    for c in J(e.get("output")).get("candidates", []):
        q = set(tok(c["expression"]))
        if p and q: sims.append(len(p & q) / len(p | q))
print(f"child-parent token Jaccard: median {np.median(sims):.2f}, share >=0.6: {np.mean(np.array(sims) >= 0.6):.2f}")
topics = collections.Counter(J(e.get("input")).get("topic") or "(empty)" for e in rows if e["name"] == "evolve-node")
print("topics of evolved parents:", topics.most_common(8))
