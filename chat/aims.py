"""The aim stack (SPEC.md, "The aim stack"): built by code from the reviewer's codes of the person's turns. An aim is
the person's own words, quoted, never reworded. The stack's last entry is the most specific aim now active."""
from __future__ import annotations


def new_aim(aims: list[dict], turn: dict) -> dict:
    a = {"id": f"A{len(aims) + 1}", "from": turn["id"], "quotes": [turn["text"]]}
    aims.append(a)
    return a


def apply(stack: list[str], aims: list[dict], turn: dict, code: dict) -> list[str]:
    """The stack after a person turn, as aim ids. `aims` holds every aim so far and grows here; old aims stay, so a
    later turn can go back to one. Feedback, asides and repairs leave the stack as it was."""
    f, rel = code.get("function"), code.get("relation")
    by_id = {a["id"]: a for a in aims}
    if f in ("feedback", "aside", "repair"):
        return list(stack)
    if f == "revision":
        return list(stack[:-1]) + [new_aim(aims, turn)["id"]]
    if not stack or rel == "new":
        return [new_aim(aims, turn)["id"]]
    if rel == "same":
        by_id[stack[-1]]["quotes"].append(turn["text"])
        return list(stack)
    if rel == "narrower":
        return list(stack) + [new_aim(aims, turn)["id"]]
    if rel == "broader":
        rest = list(stack[:-1]) or [new_aim(aims, turn)["id"]]
        if len(stack) > 1:
            by_id[rest[-1]]["quotes"].append(turn["text"])
        return rest
    if rel == "back" and code.get("back_to") in by_id:
        target = code["back_to"]
        if target in stack:
            return list(stack[:stack.index(target) + 1])
        return [target]
    return list(stack)                          # "back" to an aim that does not exist: the stack stays, the code is kept
