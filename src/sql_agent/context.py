import json

def estimate_context(messages: list, tools: list) -> None:
    by_role = {}
    for m in messages:
        role = m.get("role", "?")
        size = len(json.dumps(m, ensure_ascii=False))
        by_role[role] = by_role.get(role, 0) + size

    tools_size = len(json.dumps(tools, ensure_ascii=False))
    total = sum(by_role.values()) + tools_size

    print(f"{'part':<12} {'char':>8} {'share':>7}")
    print("-" * 20)
    print(f"{'tools':<12} {tools_size:>8,} {tools_size / total:>6.1%}")
    for role, size in sorted(by_role.items(), key=lambda kv: -kv[-1]):
        print(f"{role:<12} {size:>8} {size / total:>6.1%}")
    print("-" * 30)
    print(f"{'TOTAL':<12} {total:>8,}")


MAX_TOOL_RESULT_CHARS = 2000

INNER_NOTICE = "\n\n[Truncated"


def truncate_tool_result(result: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Last-resort truncation, with two rules that are not obvious:

    1. The tool may have truncated already and said so at the *end* of its
       output ("showing 50 of 200 rows"). Cutting by character count would
       remove that sentence, and the model would then believe it has the whole
       result. The notice is preserved and re-appended.
    2. Never cut mid-line. Half a data row still looks like a whole one, and
       the model will happily read "Chengd" as a city name.
    """
    if len(result) <= limit:
        return result

    # Rule 1: split off the tool's own notice, re-attach it verbatim at the end.
    idx = result.rfind(INNER_NOTICE)
    inner_notice = result[idx:] if idx != -1 else ""
    body = result[:idx] if idx != -1 else result

    outer_notice_budget = 120                       # room for our own notice
    body_limit = max(0, limit - len(inner_notice) - outer_notice_budget)
    kept = body[:body_limit]

    # Rule 2: back off to the last complete line.
    last_newline = kept.rfind("\n")
    if last_newline != -1:
        kept = kept[:last_newline]

    return (f"{kept}\n\n[Truncated by the agent: showing the first {kept.count(chr(10)) + 1} "
            f"lines of this result.]{inner_notice}")


def apply_sliding_window_safe(messages: list, keep_recent: int = 10) -> list:
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    cut = max(0, len(rest) - keep_recent)
    while cut < len(rest) and rest[cut].get("role") == "tool":
        cut += 1
    return system + rest[cut:]