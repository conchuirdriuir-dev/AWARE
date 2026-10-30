"""Compare run directories side by side, for example a small model inside AWARE against a larger model run bare:

  python -m toyworld.compare runs/toy-8b runs/toy-14b-bare

One row per (run directory, arm): what the arm achieved and what it cost. The model and conditions come from the
`start` event in each run's record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .score import load

FAILURES = [("stockout_fixed", "stockout"), ("bank_fixed", "bank"), ("goodhart_at_end", "goodhart"),
            ("hidden_days", "hidden"), ("false_reports", "false"), ("escalated", "escalated"),
            ("diagnosed", "diagnosed"), ("delusion_till_goal_days_from_day3", "till-goal"),
            ("wishful_predictions", "wishful")]


def start_event(run_dir: Path) -> dict:
    with (run_dir / "log.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "start":
                return e
    return {}


def main(argv=None) -> None:
    dirs = [Path(d) for d in (argv if argv is not None else sys.argv[1:])]
    if not dirs:
        sys.exit(__doc__)
    print(f"{'run':24} {'arm':8} runs  " + "  ".join(f"{h:>9}" for _, h in FAILURES)
          + "   calls  prompt tok  reply tok  minutes")
    for d in dirs:
        by: dict[str, list[dict]] = {}
        for s in load(d):
            by.setdefault(s["arm"], []).append(s)
        for arm, xs in by.items():
            n = len(xs)
            f = lambda k: sum(x.get(k) or 0 for x in xs)
            cells = "  ".join(f"{f(k):>6}/{n}" if isinstance(xs[0].get(k), bool) else f"{f(k):>9}" for k, _ in FAILURES)
            print(f"{d.name[:24]:24} {arm:8} {n:4}  {cells}   {f('model_calls'):5}  {f('prompt_tokens'):10}"
                  f"  {f('reply_tokens'):9}  {f('seconds') / 60:7.1f}")
        st = start_event(d)
        print(f"{'':24} model={st.get('model')} pressure={st.get('cfg', {}).get('pressure')} "
              f"view={st.get('cfg', {}).get('view')} settle={st.get('settle')}")


if __name__ == "__main__":
    main()
