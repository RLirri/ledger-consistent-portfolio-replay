"""Generate five compact LaTeX tables from locally held saved analysis."""

from pathlib import Path
import json, hashlib
import os

R = Path(os.environ.get("REPLAY_EVIDENCE_ROOT", "local/evidence")).resolve()
P = Path(os.environ.get("REPLAY_FIGURE_ROOT", "figures/generated")).resolve()
A = R / "experiments/v2_grid/analysis"

S = {s["name"]: s for s in json.loads((A / "scorecard.json").read_text())}
C = json.loads((A / "primary_contrasts.json").read_text())
F = json.loads((A / "factorials.json").read_text())
cons = json.loads((R / "FIGURE_CONTRACTS.json").read_text())
records = {}


def esc(x):
    return (
        str(x)
        .replace("&", r"\&")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("$", r"\$")
    )


def table(key, caption, headers, rows, wide=False):
    spec = {"T1": r"@{}lp{2.05in}@{}", "T2": r"@{}lrp{2.0in}@{}"}.get(
        key, "l" + "r" * (len(headers) - 1)
    )
    pad = "6pt" if wide else "4pt"
    env = "table*" if wide else "table"
    tex = (
        "\\begin{"
        + env
        + "}[t]\n\\centering\\small\n\\caption{"
        + caption
        + "}\\label{tab:"
        + key
        + "}\n\\setlength{\\tabcolsep}{"
        + pad
        + "}\n\\begin{tabular}{"
        + spec
        + "}\n\\toprule\n"
        + " & ".join(headers)
        + r" \\"
        + "\n\\midrule\n"
    )
    tex += (
        "\n".join(" & ".join(str(v) for v in row) + r" \\" for row in rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{"
        + env
        + "}\n"
    )
    (P / "tables" / f"{key}.tex").write_text(tex)
    records[key] = dict(
        caption=caption,
        headers=headers,
        formatted_rows=rows,
        sha256=hashlib.sha256(tex.encode()).hexdigest(),
    )


table(
    "T1",
    "Frozen data and primary execution conventions. Price returns exclude ordinary distributions.",
    ["Item", "Convention"],
    [
        ["Universe", "14 retrospective US ETFs"],
        ["Evaluation", "2010-03-31--2026-06-15"],
        ["Common calendar", "4,077 sessions; zero dropped"],
        ["Sizing / features", "Prior close / lagged 20 sessions"],
        ["Fill / close mark", "Adjusted open / adjusted close"],
        ["Units / cash", "Fractional / no borrowing"],
        ["Momentum", "Positive top four; weights $1/k$"],
        ["Cost / liquidity", "Scenario impact / lagged dollar ADV"],
    ],
)
table(
    "T2",
    "The complete 62-scenario economic design. Diagnostics are distinct from candidate defaults.",
    ["Family", "Economic cells", "Purpose"],
    [
        ["E1", 6, "Controlled timing (4 invalid controls)"],
        ["E2", 24, "2 strategies x 3 capitals x costs x limit"],
        ["E2b", 16, "Cost/minimum/cash/units factorial"],
        ["E3", 4, "Residual-policy diagnostics"],
        ["E4", 2, "Sequential-cash diagnostics"],
        ["E5", 10, "One-at-a-time local sensitivity"],
    ],
)
rows = []
for c in C:
    s = c["arm"].split("_")[1]
    k = c["arm"].split("_")[2]
    b = c["bootstrap"]["20"]["statistics"]["active_return_ann"]
    b60 = c["bootstrap"]["60"]["statistics"]["active_return_ann"]
    rows.append(
        [
            c["contrast_id"],
            ("EW" if s == "A" else "MOM") + " " + k,
            ("Feedback" if c["effect_class"] == "cost_feedback" else "Combined"),
            f"{c['active_return_ann'] * 1e4:.3f}",
            f"[{b['lower'] * 1e4:.3f}, {b['upper'] * 1e4:.3f}]",
            f"[{b60['lower'] * 1e4:.3f}, {b60['upper'] * 1e4:.3f}]",
            f"{c['terminal_gap_ann_bps']:.3f}",
        ]
    )
table(
    "T3",
    "All primary effects against matched post-hoc references. Active return and catch-up ratio are distinct bps/year estimands. Intervals are pointwise 95\%, conditional on modeled paths.",
    ["ID", "Arm", "Class", "Active", "CI: block 20", "CI: block 60", "Catch-up"],
    rows,
    True,
)
rows = []
for s in ["A", "C"]:
    for k in ["10m", "100m", "1b"]:
        r = S[f"E2_{s}_{k}_cost1_cap1"]
        d = r["demand"]
        rows.append(
            [
                ("EW" if s == "A" else "MOM") + " " + k,
                f"{r['liquidity_limited_share'] * 100:.3f}",
                f"{r['full_period']['liquidity_limited_share_of_ticker_sessions'] * 100:.3f}",
                f"{d['executed_share'] * 100:.3f}",
                f"{d['withdrawn_share'] * 100:.3f}",
                f"{d['outstanding_end_share'] * 100:.5f}",
                f"{r['wte_risky_mean'] * 100:.3f}",
                f"{r['cash_weight_mean'] * 100:.3f}",
            ]
        )
table(
    "T4",
    "Both-on liquidity and distortion. Limited rows and ticker-sessions have distinct denominators. Cohort percentages use each arm's own originated value. WTE is mean risky L2 times 100.",
    [
        "Arm",
        "Rows (\%)",
        "Ticker (\%)",
        "Executed (\%)",
        "Withdrawn (\%)",
        "Outstanding (\%)",
        "WTE",
        "Cash (\%)",
    ],
    rows,
    True,
)
rows = []
for f in F["e2"]:
    o = f["outcomes"]["active_return_ann"]
    rows.append(
        [
            ("EW" if f["strategy"] == "A" else "MOM")
            + " "
            + f["initial_capital_label"],
            f"{o['raw_interaction'] * 1e4:.5f}",
            "Yes" if o["economic_materiality_5bps"] else "No",
            "Yes" if o["numerically_zero"] else "No",
        ]
    )
table(
    "T5",
    "Raw H2c interactions: $I=Y_{11}-Y_{10}-Y_{01}+Y_{00}$. Material means $|I|\geq5$ bps/year. Numerical zero uses its separate frozen scale tolerance.",
    ["Group", "I (bps/year)", "Material", "Numerical zero"],
    rows,
)
(P / "tables/table_source_values.json").write_text(json.dumps(records, indent=2))
# A separate source-level receipt; rendered readability is evaluated after compilation.
(P / "manifests/table_generation.json").write_text(
    json.dumps(
        dict(
            tables=records,
            canonical_sources={
                str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    A / "scorecard.json",
                    A / "primary_contrasts.json",
                    A / "factorials.json",
                ]
            },
            note="Formatting receipt, not independent numerical or visual verification",
        ),
        indent=2,
    )
)


# Display-only capital labels, matching the separately verified publication tables.
for key in ["T3", "T4", "T5"]:
    path = P / "tables" / (key + ".tex")
    text = path.read_text()
    for row in records[key]["formatted_rows"]:
        for i, value in enumerate(row):
            if isinstance(value, str) and value.startswith(("EW ", "MOM ")):
                label = (value.replace(" 10m", r" \$10M")
                         .replace(" 100m", r" \$100M")
                         .replace(" 1b", r" \$1B"))
                text = text.replace(value + " &", label + " &")
                row[i] = label
    path.write_text(text)
    records[key]["sha256"] = hashlib.sha256(text.encode()).hexdigest()
(P / "tables/table_source_values.json").write_text(json.dumps(records, indent=2))
receipt = json.loads((P / "manifests/table_generation.json").read_text())
receipt["tables"] = records
receipt["display_only_refinement"] = "Capital labels; numerical cells unchanged"
(P / "manifests/table_generation.json").write_text(json.dumps(receipt, indent=2))
