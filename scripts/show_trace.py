"""Fetch a trace back from the Langfuse API and print its structure.

    .venv310/bin/python -m scripts.show_trace <trace_id>

Confirms what actually landed server-side — the span tree, token usage, and the
cost Langfuse inferred — rather than trusting that the SDK sent it.
"""

import base64
import os
import sys
import time

import httpx

from app import config  # noqa: F401 — loads .env into os.environ


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    trace_id = sys.argv[1]

    host = os.environ["LANGFUSE_HOST"].rstrip("/")
    auth = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()

    # Ingestion is asynchronous; a freshly flushed trace may 404 briefly.
    for attempt in range(6):
        r = httpx.get(
            f"{host}/api/public/traces/{trace_id}",
            headers={"Authorization": f"Basic {auth}"},
            timeout=30,
        )
        if r.status_code == 200:
            break
        print(f"  attempt {attempt + 1}: {r.status_code} — waiting for ingestion…")
        time.sleep(5)
    else:
        print("Trace not queryable. Ingestion can lag a few seconds.")
        return 1

    t = r.json()
    print(f"name      : {t.get('name')}")
    print(f"tags      : {t.get('tags')}")
    print(f"input     : {str(t.get('input'))[:80]}")
    print(f"output    : {str(t.get('output'))[:80]}")
    print(f"latency   : {t.get('latency')}s")
    print(f"total cost: ${t.get('totalCost')}")

    print("\nobservations:")
    for o in sorted(t.get("observations", []), key=lambda x: x.get("startTime") or ""):
        usage = (o.get("usage") or {}).get("totalUsage")
        cost = o.get("calculatedTotalCost") or o.get("totalCost")
        line = f"  {o['type']:11} {o['name']:16} {o.get('latency', 0):>6.2f}s"
        if usage:
            line += f"  tokens={usage}"
        if cost:
            line += f"  cost=${cost:.8f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
