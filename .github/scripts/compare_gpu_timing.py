"""
Compare timing results between base and PR runs for one or more tracing examples.

Usage:
    python compare_gpu_timing.py <base1.json> <pr1.json> <name1> \\
        [<base2.json> <pr2.json> <name2> ...] [-o output.md]
Each example is a (base_json, pr_json, example_name) triple. Any number of
triples may be given. If -o/--output is omitted, prints to stdout.

Example:
    python compare_gpu_timing.py base_a.json pr_a.json "Example A" \\
                                  base_b.json pr_b.json "Example B" \\
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


def fmt(val, spec=".4f"):
    """Format a number for display, or return an em-dash for missing values."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:{spec}}"
    return str(val)


def format_metadata_table(base, pr):
    keys = [
        "tmax",
        "nparticles",
        "tolerance",
        "resolution",
        "loss_fraction_dbl",
        "loss_fraction_flt",
    ]
    lines = ["| Field | Base | PR |", "| --- | --- | --- |"]
    for key in keys:
        b_val = base.get(key)
        p_val = pr.get(key)
        differ = b_val is not None and p_val is not None and b_val != p_val
        spec = "" if differ else ".4g"
        flag = "!!!" if differ else ""
        lines.append(f"| {key} | {fmt(b_val, spec)} | {fmt(p_val, spec)}{flag} |")
    return "\n".join(lines)


def format_timing_table(base, pr):
    base_times = base.get("times", {})
    pr_times = pr.get("times", {})
    all_keys = sorted(set(base_times) | set(pr_times))

    no_base = not base  
    no_pr = not pr

    lines = [
        "| Test Name | %Δ | Base (s) | PR (s) | Δ (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in all_keys:
        b_time = base_times.get(key)
        p_time = pr_times.get(key)

        if b_time is None or p_time is None:
            if no_base:
                label = "NO BASE"
            elif no_pr:
                label = "NO PR"
            else:
                label = "NEW"
            lines.append(f"| {key} | {label} | {fmt(b_time)} | {fmt(p_time)} | — |")
            continue

        delta = p_time - b_time
        pct = (delta / b_time * 100) if b_time != 0 else float("inf")
        lines.append(
            f"| {key} | {pct:+.2f}% | {fmt(b_time)} | {fmt(p_time)} | {delta:+.4f} |"
        )
    return "\n".join(lines)


def format_example(base_path, pr_path, example_name):
    """Build the markdown section for a single example."""
    base = load(base_path)
    pr = load(pr_path)

    missing_note = ""
    if not base or not pr:
        missing = []
        if not base:
            missing.append("base")
        if not pr:
            missing.append("PR")
        missing_note = (
            f"!!! Missing results file(s): {', '.join(missing)}. "
            "Showing available values only.\n\n"
        )

    return (
        f"### {example_name}\n\n"
        f"{missing_note}"
        f"**Metadata**\n{format_metadata_table(base, pr)}\n\n"
        f"{format_timing_table(base, pr)}\n"
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
        metavar="base.json pr.json example_name",
        help="One or more (base_json, pr_json, example_name) triples.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="output.md",
        help="Write combined report to this file instead of stdout.",
    )
    parser.add_argument(
        "--pr-status",
        default="success",
        help="Outcome of the PR GPU job.",
    )
    parser.add_argument(
        "--base-status",
        default="success",
        help="Outcome of the base GPU job.",
    )
    args = parser.parse_args()

    if len(args.triples) % 3 != 0:
        parser.error(
            f"Expected triples of (base.json, pr.json, example_name), "
            f"got {len(args.triples)} positional arguments."
        )

    examples = [args.triples[i : i + 3] for i in range(0, len(args.triples), 3)]
    return examples, args.output, args.pr_status, args.base_status


def main():
    examples, output_path, pr_status, base_status = parse_args()
    sections = [
        format_example(base_path, pr_path, example_name)
        for base_path, pr_path, example_name in examples
    ]
    status_note = ""
    if pr_status != "success":
        status_note += f"!!! PR GPU job did not succeed ({pr_status}).\n\n"
    if base_status != "success":
        status_note += f"!!! Base GPU job did not succeed ({base_status}).\n\n"
    output = "# Timing Comparison\n\n" + status_note + "\n---\n\n".join(sections)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
