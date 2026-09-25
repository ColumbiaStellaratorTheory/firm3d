"""
Compare timing results between master and PR runs for one or more tracing examples.

Usage:
    python compare_timing.py <master1.json> <pr1.json> <name1> [<master2.json> <pr2.json> <name2> ...] [-o output.md]

Each example is a (master_json, pr_json, example_name) triple. Any number of
triples may be given. If -o/--output is omitted, prints to stdout.

Example:
    python compare_timing.py master_a.json pr_a.json "Example A" \\
                              master_b.json pr_b.json "Example B" \\
                              -o timing_report.md
"""

import argparse
import json


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def fmt(val, precision=4):
    """Format a number for display, or return an em-dash for missing values."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def format_metadata_table(master, pr):
    keys = [
        "tmax",
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

    no_master = not master
    no_pr = not pr

    lines = [
        "| Test Name | %Δ | Master (s) | PR (s) | Δ (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in all_keys:
        m_time = master_times.get(key)
        p_time = pr_times.get(key)

        if m_time is None or p_time is None:
            if no_master:
                label = "NO MASTER"
            elif no_pr:
                label = "NO PR"
            else:
                label = "NEW"
            lines.append(f"| {key} | {label} | {fmt(m_time)} | {fmt(p_time)} | — |")
            continue

        delta = p_time - m_time
        pct = (delta / m_time * 100) if m_time != 0 else float("inf")
        lines.append(
            f"| {key} | {pct:+.2f}% | {fmt(m_time)} | {fmt(p_time)} | {delta:+.4f} |"
        )
    return "\n".join(lines)


def format_example(master_path, pr_path, example_name):
    """Build the markdown section for a single example."""
    master = load(master_path)
    pr = load(pr_path)

    missing_note = ""
    if not master or not pr:
        missing = []
        if not master:
            missing.append("master")
        if not pr:
            missing.append("PR")
        missing_note = (
            f"!!! Missing results file(s): {', '.join(missing)}. "
            "Showing available values only.\n\n"
        )

    return (
        f"### {example_name}\n\n"
        f"{missing_note}"
        f"**Metadata**\n{format_metadata_table(master, pr)}\n\n"
        f"{format_timing_table(master, pr)}\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare timing results for one or more tracing examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "triples",
        nargs="+",
        metavar="master.json pr.json example_name",
        help="One or more (master_json, pr_json, example_name) triples.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="output.md",
        help="Write combined report to this file instead of stdout.",
    )
    args = parser.parse_args()

    if len(args.triples) % 3 != 0:
        parser.error(
            f"Expected triples of (master.json, pr.json, example_name), "
            f"got {len(args.triples)} positional arguments."
        )

    examples = [args.triples[i : i + 3] for i in range(0, len(args.triples), 3)]
    return examples, args.output


def main():
    examples, output_path = parse_args()

    sections = [
        format_example(master_path, pr_path, example_name)
        for master_path, pr_path, example_name in examples
    ]
    output = "# Timing Comparison\n\n" + "\n---\n\n".join(sections)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
