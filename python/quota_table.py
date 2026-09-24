#!/usr/bin/env python3

import json
import sys  

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lab import parse_quantity  # noqa: E402

GPU = "nvidia.com/gpu"


def main() -> None:
    data = json.load(sys.stdin)
    rows = []
    for item in data.get("items", []):
        queue = item["metadata"]["name"]
        status = item.get("status", {})
        usage = {f["name"]: f for f in status.get("flavorsUsage", [])}
        reserv = {f["name"]: f for f in status.get("flavorsReservation", [])}
        for flavor, fu in usage.items():
            used = borrowed = 0.0
            for r in fu.get("resources", []):
                if r["name"] == GPU:
                    used = parse_quantity(r.get("total"))
                    borrowed = parse_quantity(r.get("borrowed"))
            resv = 0.0
            for r in reserv.get(flavor, {}).get("resources", []):
                if r["name"] == GPU:
                    resv = parse_quantity(r.get("total"))
            if used or resv:
                rows.append((queue, flavor, used, resv, borrowed))

    if not rows:
        print("  (ничего не занято)")
        return

    print(f"  {'queue':<10} {'flavor':<14} {'used':>6} {'reserved':>9} {'borrowed':>9}")
    total_used = total_borrowed = 0.0
    for queue, flavor, used, resv, borrowed in sorted(rows):
        print(f"  {queue:<10} {flavor:<14} {used:>6.0f} {resv:>9.0f} {borrowed:>9.0f}")
        total_used += used
        total_borrowed += borrowed
    print(f"  {'ИТОГО':<10} {'':<14} {total_used:>6.0f} {'':>9} {total_borrowed:>9.0f}")


if __name__ == "__main__":
    main()
