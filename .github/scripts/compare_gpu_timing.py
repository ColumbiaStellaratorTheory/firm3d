"""
Compare timing results between a master run and a PR run for one tracing example.

Usage:
    python compare_timing.py <master.json> <pr.json> <example_name> [output.md]

If output.md is omitted, prints to stdout.
"""

import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(val, precision=4):
    """Format a number for display, or return an em-dash for missing values."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def format_metadata_table(master, pr):
    keys = [
        "nparticles",
        "tolerance",
        "resolution",
        "loss_fraction_dbl",
        "loss_fraction_flt",
    ]
    lines = ["| Field | Master | PR |", "| --- | --- | --- |"]
    for key in keys:
        m_val = master.get(key)
        p_val = pr.get(key)
        flag = "!!!" if m_val != p_val else ""
        lines.append(f"| {key} | {fmt(m_val)} | {fmt(p_val)}{flag} |")
    return "\n".join(lines)


def format_timing_table(master, pr):
    master_times = master.get("times", {})
    pr_times = pr.get("times", {})
    all_keys = sorted(set(master_times) | set(pr_times))

    lines = [
        "| Test Name | %Δ | Master (s) | PR (s) | Δ (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in all_keys:
        m_time = master_times.get(key)
        p_time = pr_times.get(key)

        if m_time is None or p_time is None:
            lines.append(f"| {key} | NEW | {fmt(m_time)} | {fmt(p_time)} | — |")
            continue

        delta = p_time - m_time
        pct = (delta / m_time * 100) if m_time != 0 else float("inf")
        lines.append(
            f"| {key} | {pct:+.2f}% | {fmt(m_time)} | {fmt(p_time)} | {delta:+.4f} |"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    master_path, pr_path, example_name = sys.argv[1], sys.argv[2], sys.argv[3]
    output_path = sys.argv[4] if len(sys.argv) > 4 else None

    try:
        master = load(master_path)
    except FileNotFoundError:
        master = {}
    try:
        pr = load(pr_path)
    except FileNotFoundError:
        pr = {}

    if not master or not pr:
        output = (
            f"### {example_name}\n\n"
            f"!!! Missing results — master: {'found' if master else 'MISSING'}, "
            f"PR: {'found' if pr else 'MISSING'}. Skipping comparison.\n"
        )
    else:
        output = (
            f"### {example_name}\n\n"
            f"**Metadata**\n{format_metadata_table(master, pr)}\n\n"
            f"{format_timing_table(master, pr)}\n"
        )

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
