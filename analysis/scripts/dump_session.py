"""Dump every observation of one Langfuse session via langfuse-cli (cursor paging)."""
import json
import subprocess
import sys
import time

session, out = sys.argv[1], sys.argv[2]
rows, cursor = [], None
while True:
    cmd = ["npx", "-y", "langfuse-cli", "api", "observations-v2", "list", "--session-id", session,
           "--fields", "core,basic,io,metadata,model,usage,trace_context", "--expand-metadata",
           "reasoning,finish_reason,latency_s,depth,times,step", "--limit", "500"]
    if cursor:
        cmd += ["--cursor", cursor]
    for attempt in range(6):
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", shell=True)
        if p.returncode == 0:
            break
        time.sleep(10 * (attempt + 1))
    else:
        sys.exit(p.stderr[-2000:])
    body = json.loads(p.stdout)
    rows += body["data"]
    cursor = (body.get("meta") or {}).get("cursor")
    print(f"{len(rows)} rows", file=sys.stderr)
    if not cursor:
        break
json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(session, len(rows))
