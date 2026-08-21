import json

def show(title: str, obj) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    else:
        print(repr(obj))



def format_result(result: str) -> str:
    try:
        return f"{int(result):,}"
    except ValueError:
        return result


def print_table(rows: list) -> None:
    if not rows:
        print("no tools")
        return

    name_width = max(len(name) for name, _, _ in rows)
    name_width = max(name_width, len("tool"))

    print(f"{'tool':<{name_width}}  {'time':>8}  result")
    print("-" * (name_width + 30))
    for name, elapsed, result in rows:
        print(f"{name:<{name_width}}  {elapsed:>7.3f}s  {format_result(result)}")
