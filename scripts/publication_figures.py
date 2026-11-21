"""Source-driven IEEE figures. --figure F2, or all. Never runs a replay."""

from pathlib import Path
import argparse, json, hashlib, string
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, colors
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps, ImageDraw
import os

R = Path(os.environ.get("REPLAY_EVIDENCE_ROOT", "local/evidence")).resolve()
P = Path(os.environ.get("REPLAY_FIGURE_ROOT", "figures/generated")).resolve()
A = R / "experiments/v2_grid/analysis"
D = P / "data"
font_path = os.environ.get("REPLAY_FONT_PATH")
if font_path:
    font_manager.fontManager.addfont(font_path)
plt.style.use(P / "paper_v2_ieee.mplstyle")
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#777777"
BLACK = "#222222"
PURPLE = "#9970AB"
S = {x["name"]: x for x in json.loads((A / "scorecard.json").read_text())}
C = json.loads((A / "contrasts.json").read_text())
F = json.loads((A / "factorials.json").read_text())
M = pd.read_csv(D / "mechanism_scorecard.csv").set_index("name")
prim = [c for c in C if c["tier"] == "primary"]
groups = [(s, k) for s in ["A", "C"] for k in ["10m", "100m", "1b"]]
glabel = ["EW $10M", "EW $100M", "EW $1B", "MOM $10M", "MOM $100M", "MOM $1B"]
caplabel = {"10m": "$10M", "100m": "$100M", "1b": "$1B"}


def name(s, k, cost=1, cap=1):
    return f"E2_{s}_{k}_cost{cost}_cap{cap}"


def load(n, suffix="dynamic"):
    return pd.read_csv(
        D / f"{n}.{suffix}.csv",
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def panel(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", pad=7)
    ax.tick_params(pad=2)


def grid(nr, nc, h=3.6, w=7.16):
    return plt.subplots(nr, nc, figsize=(w, h), layout="constrained", squeeze=False)


def zero(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(0, color=BLACK, lw=0.65, zorder=0)


def foot(fig, text):
    fig.supxlabel(text, fontsize=9)


def palette(i):
    return [BLUE, ORANGE, GREEN, PURPLE, GRAY, BLACK][i % 6]


def policy_label(n):
    bits = n.split("_")
    strategy = "EW" if bits[1] == "A" else "MOM"
    capital = caplabel[bits[2]]
    policy = (
        "replace"
        if n.startswith("E2_")
        else (
            "additive"
            if "additive" in n
            else "sequential cash"
            if "sequential" in n
            else "cancel"
        )
    )
    return strategy + " " + capital + ": " + policy


CONTRACTS = {
    "F1": (
        "How does the temporal contract support matched attribution?",
        "C1: separate information and state feedback",
        [
            "inputs and lagged decision",
            "execution gates and reconciliation",
            "closing ledger and feedback",
            "counterfactual construction",
        ],
        7.16,
        3.25,
        "MAIN PAPER",
        "The primary information contract uses prior-close prices and 20-session features ending at d-1; execution uses the next open and valuation the later close. Liquidity precedes cash scaling; final trade minima apply only in diagnostic cells. Reconciliation tracks economic demand independently of working-order cancellation. Counterfactuals share targets and inputs, with their own portfolio states. Post-hoc adjustment is return-only.",
        "Implementation schematic; no simulated intraday path or empirical capacity claim.",
    ),
    "F2": (
        "When does state feedback materially affect returns?",
        "C2: all twelve primary effects",
        [
            "weekly cost feedback",
            "weekly combined",
            "momentum cost feedback",
            "momentum combined",
        ],
        7.16,
        3.7,
        "MAIN PAPER",
        "All 12 primary contrasts: annualized arithmetic active return of stateful arms relative to matched post-hoc references. Filled circles/solid intervals use mean block 20; open squares/dashed intervals use 60. Intervals are pointwise 95% paired stationary bootstrap percentiles (2,000 replicates; seed 20261119). Panel scales differ explicitly; zero is shown throughout. These are conditional modeled comparisons, not alpha.",
        "Panel scales differ; no simultaneous coverage. Combined effects include an added liquidity restriction.",
    ),
    "F3": (
        "Which execution conditions accompany the effects?",
        "C2: discrete strategy-capital condition map",
        ["eight annotated condition rows", "evolving NAV ranges"],
        7.16,
        3.8,
        "MAIN PAPER",
        "Both-on execution conditions for weekly equal weight (EW) and daily momentum (MOM), at three initial capital levels. Each metric row uses its own 0-to-observed-maximum sequential scale; annotations retain natural units. Liquidity shares use distinct order-row and all ticker-session denominators. Participation is measured on nonzero fills. Turnover is annualized gross traded/opening pre-trade NAV. The final row records observed NAV ranges in millions.",
        "Discrete scenarios, no interpolated capacity frontier; row colors cannot be compared across metrics.",
    ),
    "F4": (
        "How do demand service and portfolio distortion coexist?",
        "C2: completion is conditional on endogenous demand",
        [
            "own-demand value shares",
            "originated demand and execution",
            "tracking and cash",
            "residual age",
        ],
        7.16,
        4.7,
        "MAIN PAPER",
        "Demand-cohort fulfillment and portfolio distortion in both-on arms. (a) Origin-price executed, withdrawn and outstanding shares, with matched unconstrained completion and the 95% descriptive line; tiny outstanding shares are printed. (b) Endogenous originated and executed dollar demand. (c) Risky and cash-inclusive mean L2 tracking error, in the same group order. (d) Working-parent age p95 and maximum in sessions. Each arm originates its own demand; no fixed order population is assumed.",
        "Origin prices differ from execution prices; parent lifecycle age is not economic-cohort waiting time. The 95% line is not causal evidence.",
    ),
    "F5": (
        "Is cost-liquidity interaction economically material?",
        "C1/C2: matched factorial attribution",
        [
            "weekly four-cell paths",
            "momentum four-cell paths",
            "raw interaction by group",
        ],
        7.16,
        3.0,
        "MAIN PAPER",
        "Four-cell cost/liquidity attribution relative to the common frictionless arm. Lines connect cost off/on within liquidity settings; colors and marker shapes indicate initial capital; filled/solid versus open/dashed distinguish liquidity off/on. The raw arithmetic-active-return interaction is Y11-Y10-Y01+Y00, in bps/year, with approved +/-5 bps economic boundaries. It equals four times the -1/+1 interaction coefficient, not twice. Numerical-zero annotations use a separate scale tolerance.",
        "Deterministic model contrasts; interaction does not establish execution skill or physical capacity.",
    ),
    "F6": (
        "How do state differences accumulate through the full history?",
        "C2: dynamic mechanism illustration",
        [
            "relative wealth",
            "normalized direct cost accumulation",
            "cash weight",
            "trading intensity",
        ],
        7.16,
        4.4,
        "MAIN PAPER",
        "Full common calendar for the predefined MOM/$1B stress illustration. Stateful $1B cost-only and both-on arms share a post-hoc reference; panel (a) also shows the $10M both-on arm against its own matched post-hoc reference. Cost accumulation is normalized by each matched initial capital; post-hoc effective debits reflect its rate recurrence. Cash is observed daily; the last panel uses a prespecified-for-this-descriptive-analysis 63-session trailing arithmetic mean of gross traded/opening NAV, multiplied by 252, requiring all 63 sessions. Rolling diagnostics are exploratory, not primary estimands.",
        "Costs and wealth effects are not additive attribution. No selected historical interval; daily-bar and extreme uncapped-participation limitations apply.",
    ),
    "F7": (
        "Which registered perturbations move comparable outcomes?",
        "C3: bounded local sensitivity",
        ["all E5 signed deviations", "all E3/E4 policy outcomes"],
        7.16,
        4.0,
        "SUPPLEMENTARY",
        "Alternative main-quality candidate: all ten one-at-a-time parameter contrasts and all six residual/cash-policy contrasts. Annualized catch-up ratios are positive when an arm ends poorer than its registered reference. The separate +/-5 bps catch-up threshold is shown. Additive carry and sequential cash rules are diagnostics; lifecycle implementation shortfall is not used to rank policies.",
        "Local one-at-a-time sensitivity is not joint robustness; favorable wealth can accompany low completion.",
    ),
    "S1": (
        "How extreme are uncapped participation tails?",
        "C2 mechanism limitation",
        ["complete ECDF", "tail exceedance and cost shares"],
        7.16,
        3.0,
        "SUPPLEMENTARY",
        "Uncapped cost-only momentum participation on all nonzero fill rows, by capital: exact empirical survival curves on a symlog participation axis (linear below 0.1% ADV), and fractions of rows, executed notional and direct costs above 100% of lagged ADV. Vertical lines mark 5% and 100%. Zero/negative fills are not logged. No fitted tail or inferred capacity frontier.",
        "Exploratory descriptive diagnostic, not a new primary test. Row and dollar-weighted fractions differ.",
    ),
    "S2": (
        "How does the nonlinear cost model act on observed fills?",
        "C2 mechanism",
        ["cost decomposition", "participation and volatility-normalized impact"],
        7.16,
        3.2,
        "SUPPLEMENTARY",
        "Direct modeled spread and impact costs per executed dollar in E2 cost-only and both-on arms. The second panel plots every uncapped MOM/$1B fill: impact cost rate divided by lagged sigma/0.01, against participation. The overlaid curve is the frozen analytical 15*min(p,1)^0.7 bps formula, not a fitted model or additional scenario.",
        "Analytical curve illustrates the assumed function. Saturation beyond 100% ADV weakens realism; no calibration claim.",
    ),
    "S3": (
        "Are effects stable in predefined historical periods?",
        "C2 conditional historical stability",
        ["cost-feedback subperiods", "combined-effect subperiods"],
        7.16,
        3.4,
        "SUPPLEMENTARY",
        "Annualized arithmetic active returns for all primary groups in each of the three predefined subperiods. Returns are assigned by end session with the preceding common close anchor. All periods are retained; no subperiod confidence intervals are available.",
        "Descriptive effects on one historical path, not independent replications.",
    ),
    "S4": (
        "How sensitive are intervals to block length?",
        "C2 uncertainty",
        ["all primary block-20 intervals", "block-60 interval widths"],
        7.16,
        3.8,
        "SUPPLEMENTARY",
        "All twelve primary paired-bootstrap intervals are reported for blocks 20 and 60; the companion panel compares interval widths. Same observations are resampled jointly within each comparison. Pointwise conditional 95% coverage is not simultaneous coverage.",
        "Two block lengths are a sensitivity check, not independent evidence.",
    ),
    "S5": (
        "Which controlled legacy assumptions contribute?",
        "C1 factorial diagnostics",
        ["sixteen cell outcomes", "all fifteen signed effects"],
        7.16,
        4.7,
        "SUPPLEMENTARY",
        "The complete E2b design at EW/$1M under V2 timing. Cell codes are cost, $1000 minimum, borrowing forbidden, integer units (1=on). All fifteen classical effects are twice their -1/+1 coefficients; labels enumerate main through four-way interactions. Terminal wealth gaps use the E2b frictionless fractional borrowing-allowed baseline.",
        "Controlled diagnostic under V2 conventions; no exact attribution of the historical 267 bps gap.",
    ),
    "S6": (
        "Do timing and residual conventions change outcomes?",
        "C3 temporal and policy sensitivity",
        ["all five timing contrasts", "policy completion vs wealth"],
        7.16,
        3.5,
        "SUPPLEMENTARY",
        "E1 compares five timing configurations with prior-close/lagged. Open/lagged is a disclosed same-open approximation; the four same-day-feature or same-close configurations are future-informed diagnostic controls (cross markers). Policy diagnostics retain every E3/E4 contrast and plot own-demand completion against comparable arithmetic active return.",
        "Invalid timing controls are not candidate defaults. Lifecycle IS endpoints differ and are excluded from rankings.",
    ),
    "S7": (
        "Are longitudinal mechanisms specific to one capital level?",
        "C2 full-history supporting diagnostics",
        [
            "relative wealth across capital",
            "daily participation across capital",
            "cash across capital",
            "tracking across capital",
        ],
        7.16,
        5.5,
        "SUPPLEMENTARY",
        "Complete MOM histories at $10M/$100M/$1B for both-on arms: relative wealth against post-hoc, maximum executed participation, cash, and risky L2 tracking error. No rescaling of time or chosen episode; row-specific y scales are disclosed by ticks.",
        "Descriptive state trajectories; participation is maximum across observed fills, not average market participation.",
    ),
    "S8": (
        "Does favorable additive-carry wealth imply demand service?",
        "C3 residual diagnostic",
        ["all residual policy age distributions", "completion and overshoot"],
        7.16,
        3.7,
        "SUPPLEMENTARY",
        "All four E3 alternatives and their three distinct replace references: working-parent age distributions and own-demand completion alongside gross overshoot/originated demand. Overshoot is separate from cohort service and can exceed a large fraction of originated demand. Favorable wealth is not evidence of correct target fulfillment.",
        "Parent ages are policy-dependent lifecycle durations, not common-horizon IS or cohort waiting times.",
    ),
}


def revise_contract(key, height, caption):
    parts = list(CONTRACTS[key])
    parts[4] = height
    parts[6] = caption
    CONTRACTS[key] = tuple(parts)


revise_contract(
    "F1",
    3.6,
    "Primary event contract: prior-close decision, next-open execution and session-d closing valuation. Forward processing uses black orthogonal arrows; gray feedback carries cash and holdings around the outer perimeter to the next sizing event, where they are revalued at its prescribed mark. Policy-specific lifecycle implementation-shortfall (IS) marks are later evaluation only and never feed execution. The frictionless reference fill tape is priced with the lagged cost model and compounded as a return-only post-hoc comparator. EW denotes weekly equal weight; MOM denotes daily momentum.",
)
revise_contract(
    "F2",
    3.85,
    "All 12 primary contrasts retain four strategy/effect panels. Each diamond shows one shared annualized arithmetic-active-return estimate; vertically separated solid blue and dashed gray segments show pointwise 95% paired stationary-bootstrap intervals for mean blocks 20 and 60 (2,000 replicates; seed 20261119). Panel-specific scales are explicitly different; no axis breaks are used. EW is weekly equal weight and MOM is daily momentum. Combined comparisons add liquidity limits to feedback. Intervals are conditional and do not provide simultaneous coverage.",
)
_nav = "; ".join(
    ("EW" if s == "A" else "MOM")
    + " "
    + caplabel[k]
    + ": "
    + f"{S[name(s, k)]['nav_min'] / 1e6:.1f}–{S[name(s, k)]['nav_max'] / 1e6:.1f}"
    for s, k in groups
)
revise_contract(
    "F3",
    3.6,
    "Vector condition map for both-on arms, grouped by weekly equal weight (EW) and daily momentum (MOM) and initial capital. Cell annotations are unchanged. Each metric has its own color=value/row maximum scale: color indicates magnitude, not uniformly favorable or unfavorable performance. Limited-order and all-ticker-session denominators are distinct; participation uses nonzero fills and lagged ADV. No capacity frontier is interpolated. Observed NAV ranges ($M), also in source tables: "
    + _nav
    + ".",
)
revise_contract(
    "F4",
    4.25,
    "Origin-price demand service and distortion in both-on arms. Executed (blue), withdrawn (gray hatching) and outstanding (orange) shares use each replay's own originated demand. Open diamonds above the bars identify matched unconstrained completion; a dashed line marks the 95% descriptive threshold and 100% is the labeled endpoint. Outstanding percentages are printed to at most four decimal places; exact small values remain in the source data. Full panels retain endogenous dollar demand, risky/cash-inclusive L2 tracking and descriptive parent durations; lifecycle IS is not ranked across policies.",
)
revise_contract(
    "F5",
    3.3,
    "Matched four-cell cost/liquidity paths and raw annualized-arithmetic-active-return interaction I_A=Y11−Y10−Y01+Y00. Capital is distinguished by marker shape and color; filled/solid versus open/dashed denote no limit versus the 5% ADV budget. Every raw interaction is annotated, with material values rounded from canonical values to two decimals. The gray ±5 bps/year economic band is separate from the frozen numerical-zero tolerance (NZ); neither is a statistical-significance label. I_A equals four times the signed interaction coefficient.",
)
revise_contract(
    "F6",
    4.4,
    "Full-calendar MOM/$1B mechanism illustration, with the $10M both-on relative-wealth reference in panel (a). Shared styles identify cost-only, both-on and the reference-rate post-hoc debit. Cumulative modeled debit is divided by initial NAV; it is not a terminal wealth-loss measure. Cash panel retains every daily observation as a thin subdued path and adds a full-window trailing 63-session mean. Trading intensity uses the same trailing window. Both means are explicitly exploratory diagnostics, not new primary outcomes; complete raw capital-specific paths remain in S7.",
)
revise_contract(
    "S3",
    3.5,
    CONTRACTS["S3"][6]
    + " Period boundaries: 1 April 2010–31 December 2014; 1 January 2015–31 December 2019; 1 January 2020–15 June 2026. A single shared legend defines all strategy/capital paths.",
)
revise_contract(
    "S5",
    4.7,
    CONTRACTS["S5"][6].replace(
        "Cell codes are cost, $1000 minimum, borrowing forbidden, integer units (1=on).",
        "Cell bit order is C (cost), M ($1,000 minimum), B (borrowing forbidden) and U (integer units), with 1=on. Display aliases do not change registered factor identities.",
    ),
)
revise_contract(
    "S8",
    4.0,
    CONTRACTS["S8"][6]
    + " Step encodings show integer-age categories, with no interpolation of a continuous waiting-time distribution; shared policy labels identify the cases.",
)
for _key, _contract in list(CONTRACTS.items()):
    if _key != "F1":
        _parts = list(_contract)
        _parts[6] += (
            " Evaluation: 31 March 2010–15 June 2026; 4,077 sessions (4,076 return intervals)."
        )
        CONTRACTS[_key] = tuple(_parts)
for _key, _height in [("F1", 3.0), ("F3", 3.1), ("F4", 3.05)]:
    _parts = list(CONTRACTS[_key])
    _parts[4] = _height
    _parts[6] = "Compact publication alternative. " + _parts[6]
    if _key == "F4":
        _parts[2] = [
            "own-demand service with matched references",
            "risky/cash-inclusive tracking",
        ]
        _parts[6] += (
            " This compact alternative omits the dollar-demand and duration panels; the full F4 and S8 preserve those observations. It makes no common-horizon IS claim."
        )
    CONTRACTS[_key + "_compact"] = tuple(_parts)


def contract_records():
    out = []
    for key, (q, claim, panels, w, h, placement, caption, limits) in CONTRACTS.items():
        figure_key = key
        key = key.split("_")[0]
        if key in ["F2", "S3", "S4"]:
            names = sorted(
                {c["arm"] for c in prim} | {c["reference_source"] for c in prim}
            )
        elif key in ["F3", "F4"]:
            names = [name(s, k) for s, k in groups] + [
                name(s, k, 0, 0) for s, k in groups
            ]
        elif key == "F5":
            names = [n for n in S if n.startswith("E2_")]
        elif key == "S1":
            names = [name("C", k, 1, 0) for k in caplabel]
        elif key == "S2":
            names = [name(s, k, 1, l) for s, k in groups for l in [0, 1]]
        elif key == "F6":
            names = [name("C", "1b", a, b) for a, b in [(0, 0), (1, 0), (1, 1)]] + [
                name("C", "10m"),
                name("C", "10m", 0, 0),
            ]
        elif key == "S7":
            names = [name("C", k) for k in caplabel] + [
                name("C", k, 0, 0) for k in caplabel
            ]
        elif key == "S5":
            names = [n for n in S if n.startswith("E2b")]
        elif key == "F1":
            names = []
        elif key in ["F7", "S6"]:
            exps = ["E3", "E4", "E5"] if key == "F7" else ["E1", "E3", "E4"]
            rows = [c for c in C if c["experiment"] in exps]
            names = sorted(
                {c["arm"] for c in rows} | {c["reference_source"] for c in rows}
            )
        elif key == "S8":
            names = [n for n in S if n.startswith("E3")] + [
                name("A", "1b"),
                name("C", "100m"),
                name("C", "1b"),
            ]
        else:
            names = sorted(S)
        out.append(
            dict(
                figure_id=figure_key,
                source_family=key,
                variant="compact" if figure_key != key else "full",
                scientific_question=q,
                intended_claim=claim,
                evidence_hierarchy="frozen validated tapes > canonical analysis > exploratory saved-tape diagnostics",
                panels=panels,
                width_inches=w,
                height_inches=h,
                placement=placement,
                caption=caption,
                limitations=limits,
                scenario_ids=names,
                run_spec_ids={n: S[n]["run_spec_id"] for n in names},
                canonical_sources=[
                    "experiments/v2_grid/analysis/scorecard.json",
                    "experiments/v2_grid/analysis/contrasts.json",
                    "experiments/v2_grid/analysis/factorials.json",
                ],
                transformations="See executable function "
                + key
                + " in scripts/figures.py and scripts/prepare_evidence.py; exact plotting values exported in data/"
                + figure_key
                + ".plot_data.json",
                filtering="Only final 8e8096f/483217d3 identities; all registered cells retained; nonzero fills only for participation distributions",
                uncertainty="Fixed pointwise paired stationary bootstrap where present; no new interval computation",
                rolling_window="63 full sessions trailing mean, exploratory, F6(c cash and d turnover) only; none elsewhere",
                axis_transformations="Explicit symlog participation in S1/S2 (linear below 0.1% ADV) and log originated dollars in full F4; F3 independent sequential row scales; otherwise linear",
                source_units="Returns decimal converted to bps/year ×10000; weights/shares converted to percent ×100; NAV dollars divided by 1e6 or initial capital where labeled",
            )
        )
    return out


def F1(compact=False):
    fig = plt.figure(figsize=(7.16, 3.0 if compact else 3.6))
    ax = fig.add_axes([0.015, 0.015, 0.97, 0.97])
    ax.set(xlim=(0, 10.6), ylim=(0, 5.8))
    ax.axis("off")

    def box(x, y, w, h, text, col=BLACK):
        ax.add_patch(
            patches.Rectangle((x, y), w, h, facecolor="white", edgecolor=col, lw=0.9)
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            color=BLACK,
        )

    def route(points, col=BLACK):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=col, lw=0.9)
        ax.annotate(
            "",
            xy=points[-1],
            xytext=points[-2],
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9, shrinkA=0, shrinkB=0),
        )

    for x, t in [
        (1.6, "(a) Decision: close d−1"),
        (5.1, "(b) Execution: open d"),
        (8.6, "(c) Valuation and evaluation"),
    ]:
        ax.text(x, 5.45, t, ha="center", weight="bold", fontsize=9)
    box(
        0.2,
        3.55,
        2.8,
        1.35,
        "Adjusted prices and volume\nLagged 20-session features\nWeekly EW / daily MOM",
        BLUE,
    )
    box(
        0.2,
        1.65,
        2.8,
        1.45,
        "Own NAV and target weights\nTarget units minus holdings\nNewest-first withdrawals\nReconcile economic demand",
        BLUE,
    )
    box(
        3.7,
        3.55,
        2.8,
        1.35,
        "Lagged ADV liquidity budget\nSell fills and modeled costs\nCost-inclusive proportional buys",
        ORANGE,
    )
    box(
        3.7,
        1.65,
        2.8,
        1.45,
        "Final minimum: diagnostics only\nUpdate cash and holdings ledger\nResidual reconciliation\nFIFO cohort service",
        ORANGE,
    )
    box(
        7.2,
        3.55,
        2.8,
        1.35,
        "Session-d closing valuation\nMarked holdings + cash → NAV\nTracking and cash exposure",
        GREEN,
    )
    box(
        7.2,
        1.65,
        2.8,
        1.45,
        "Later lifecycle IS: evaluation only\nPolicy-specific horizon marks\nMatched counterfactuals:\nfrictionless fills → modeled costs\n→ post-hoc return compounding",
        GREEN,
    )
    forward = [
        [(1.6, 3.55), (1.6, 3.1)],
        [(3.0, 2.375), (3.35, 2.375), (3.35, 4.225), (3.7, 4.225)],
        [(5.1, 3.55), (5.1, 3.1)],
        [(6.5, 2.375), (6.85, 2.375), (6.85, 4.225), (7.2, 4.225)],
        [(8.6, 3.55), (8.6, 3.1)],
    ]
    for r in forward:
        route(r)
    feedback = [(10.0, 4.225), (10.35, 4.225), (10.35, 0.7), (1.6, 0.7), (1.6, 1.65)]
    route(feedback, GRAY)
    ax.text(
        5.8,
        0.95,
        "Carry cash and holdings to the next decision; revalue at its sizing mark",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    ax.text(
        5.3,
        0.12,
        "EW: weekly equal weight. MOM: daily momentum. Later IS marks do not feed execution.",
        ha="center",
        fontsize=9,
    )
    return fig, dict(
        nodes=[
            "lagged inputs",
            "prior-close target",
            "liquidity",
            "sells",
            "cash scaling",
            "minimum",
            "ledger",
            "closing valuation",
            "next decision",
            "post-hoc reference",
        ],
        forward_routes=forward,
        feedback_route=feedback,
        feedback_source="session-d cash and holdings state; not lifecycle IS",
        feedback_destination="next target sizing, revalued at prior-close mark",
        reference="engine.py run(): nav_pre/cash/holdings and close mark; comparator.py posthoc_adjustment",
    )


def F2():
    fig, axs = grid(2, 2, 3.85)
    data = []
    for row, s in enumerate(["A", "C"]):
        for col, typ in enumerate(["cost_feedback", "feedback_plus_liquidity"]):
            ax = axs[row, col]
            rows = [
                next(c for c in prim if c["arm"] == name(s, k, 1, int(col == 1)))
                for k in caplabel
            ]
            for j, c in enumerate(rows):
                for b, offset, marker, colr in [
                    ("20", -0.07, "o", BLUE),
                    ("60", 0.07, "s", GRAY),
                ]:
                    z = c["bootstrap"][b]["statistics"]["active_return_ann"]
                    x = z["point"] * 1e4
                    lo = z["lower"] * 1e4
                    hi = z["upper"] * 1e4
                    ax.hlines(
                        j + offset,
                        lo,
                        hi,
                        color=colr,
                        lw=1.5 if b == "20" else 0.9,
                        linestyle="-" if b == "20" else "--",
                    )
                    data.append(
                        dict(
                            contrast_id=c["contrast_id"],
                            block=b,
                            point=x,
                            lower=lo,
                            upper=hi,
                        )
                    )
                ax.vlines(x, j - 0.07, j + 0.07, color=BLACK, lw=0.5)
                ax.plot(x, j, "D", color=BLACK, ms=3.5)
            ax.set_yticks(range(3), [caplabel[k] for k in caplabel])
            ax.invert_yaxis()
            zero(ax)
            ax.set_xlabel("Arithmetic active return (bps/year)")
            panel(
                ax,
                string.ascii_lowercase[row * 2 + col],
                ("Weekly EW" if s == "A" else "Daily MOM")
                + " | "
                + ("cost feedback" if col == 0 else "combined effect"),
            )
    foot(
        fig,
        "Diamond: shared estimate; blue solid: block 20; gray dashed: block 60.\nPanel-specific scales. EW: weekly equal weight; MOM: daily momentum.",
    )
    return fig, data


def F3(compact=False):
    fig, axs = grid(1, 1, 3.1 if compact else 3.6)
    ax = axs[0, 0]
    rows = [S[name(s, k)] for s, k in groups]
    values = np.array(
        [
            [r["turnover_gross_annualized"] for r in rows],
            [r["liquidity_limited_share"] * 100 for r in rows],
            [
                r["full_period"]["liquidity_limited_share_of_ticker_sessions"] * 100
                for r in rows
            ],
            [M.loc[r["name"], "part_q0.95"] * 100 for r in rows],
            [r["demand"]["executed_share"] * 100 for r in rows],
            [r["wte_risky_mean"] * 100 for r in rows],
            [r["cash_weight_mean"] * 100 for r in rows],
        ]
    )
    norm = values / values.max(axis=1, keepdims=True)
    ax.pcolormesh(
        np.arange(7) - 0.5,
        np.arange(8) - 0.5,
        norm,
        cmap="Blues",
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=False,
    )
    ax.invert_yaxis()
    labels = [
        "Gross turnover / opening NAV (per year)",
        "Limited / all order rows (%)",
        "Limited / all ticker-sessions (%)",
        "Participation p95 / lagged ADV (%)",
        "Executed / own originated value (%)",
        "Mean risky-weight tracking L2 × 100",
        "Mean cash / closing NAV (%)",
    ]
    for i in range(len(labels)):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}" if i in [1, 2] else f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.63 else BLACK,
            )
    ax.set_xticks(range(6), [caplabel[k] for s, k in groups])
    ax.xaxis.tick_top()
    ax.set_yticks(range(7), labels)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", pad=4)
    ax.text(
        1,
        1.15,
        "Weekly equal weight (EW)",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=9,
        weight="bold",
    )
    ax.text(
        4,
        1.15,
        "Daily momentum (MOM)",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=9,
        weight="bold",
    )
    ax.axvline(2.5, color=GRAY, lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    foot(
        fig,
        "Color = value / metric-row maximum; magnitude, not a common good–bad scale.",
    )
    return fig, dict(
        labels=labels,
        groups=glabel,
        values=values.tolist(),
        normalization="value / metric-row maximum",
        nav_ranges_millions=[[r["nav_min"] / 1e6, r["nav_max"] / 1e6] for r in rows],
    )


def F4(compact=False):
    fig, axs = grid(1, 2, 3.05) if compact else grid(2, 2, 4.25)
    rows = [S[name(s, k)] for s, k in groups]
    y = np.arange(6)
    ax = axs[0, 0]
    e = np.array([r["demand"]["executed_share"] * 100 for r in rows])
    w = np.array([r["demand"]["withdrawn_share"] * 100 for r in rows])
    o = np.array([r["demand"]["outstanding_end_share"] * 100 for r in rows])
    ax.barh(y, e, height=0.54, color=BLUE, label="Executed")
    ax.barh(
        y,
        w,
        height=0.54,
        left=e,
        color="#DDDDDD",
        hatch="///",
        edgecolor=GRAY,
        lw=0.3,
        label="Withdrawn",
    )
    ax.barh(
        y,
        o,
        height=0.54,
        left=e + w,
        color=ORANGE,
        edgecolor=ORANGE,
        lw=0.3,
        label="Outstanding",
    )
    ax.plot(
        [S[name(s, k, 0, 0)]["demand"]["executed_share"] * 100 for s, k in groups],
        y - 0.34,
        "d",
        mfc="white",
        mec=BLACK,
        ms=4,
        label="Unconstrained completion",
    )
    ax.axvline(95, color=BLACK, ls="--", lw=0.8, label="95% descriptive")
    ax.axvline(100, color=GRAY, lw=0.5)
    ax.set_xlim(0, 124)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks(y, glabel)
    ax.invert_yaxis()
    ax.set_xlabel("Own originated value (%)")
    for j, v in enumerate(o):
        ax.text(104, j, "0" if v == 0 else f"{v:.4f}", va="center", fontsize=9)
    ax.text(104, -0.9, "Out (%)", fontsize=9)
    ax.annotate(
        "95%",
        xy=(95, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-8, 2),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    panel(ax, "a", "Origin-value fulfillment")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )
    orig = np.array([r["demand"]["originated_value"] / 1e9 for r in rows])
    exe = np.array([r["demand"]["executed_value"] / 1e9 for r in rows])
    if not compact:
        ax = axs[0, 1]
        ax.plot(orig, y, "o", color=GRAY, label="Originated")
        ax.plot(exe, y, "s", color=BLUE, label="Executed")
        ax.set_xscale("log")
        ax.set_yticks(y, glabel)
        ax.invert_yaxis()
        ax.set_xlabel("Origin-price demand ($bn; log scale)")
        ax.legend(frameon=False)
        panel(ax, "b", "Endogenous demand volume")
    ax = axs[0, 1] if compact else axs[1, 0]
    for j, r in enumerate(rows):
        yr = r["wte_risky_mean"] * 100
        yc = r["wte_with_cash_mean"] * 100
        ax.plot([yr, yc], [j, j], color=BLUE)
        ax.plot(yr, j, "o", color=BLUE)
        ax.plot(yc, j, "s", mfc="white", color=GRAY)
    ax.set_yticks(y, glabel)
    ax.invert_yaxis()
    ax.set_xlabel("Mean tracking L2 × 100")
    ax.margins(0.12)
    panel(ax, "b" if compact else "c", "Risky (circle) / with cash (square)")
    p95 = [M.loc[r["name"], "age_p95"] for r in rows]
    mx = [M.loc[r["name"], "age_max"] for r in rows]
    if not compact:
        ax = axs[1, 1]
        ax.plot(p95, y, "o", color=BLUE, label="p95")
        ax.plot(mx, y, "s", mfc="white", color=GRAY, label="Maximum")
        ax.set_yticks(y, glabel)
        ax.invert_yaxis()
        ax.set_xlim(-0.5, max(mx) + 1)
        ax.set_xlabel("Working-parent age (sessions)")
        ax.legend(frameon=False)
        panel(ax, "d", "Lifecycle durations; descriptive only")
    return fig, dict(
        executed_pct=e.tolist(),
        withdrawn_pct=w.tolist(),
        outstanding_pct=o.tolist(),
        originated_bn=orig.tolist(),
        executed_bn=exe.tolist(),
        age_p95=p95,
        age_max=mx,
    )


def F5():
    fig, axs = grid(1, 3, 3.3)
    dat = []
    for i, s in enumerate(["A", "C"]):
        ax = axs[0, i]
        for j, k in enumerate(caplabel):
            f = next(
                f
                for f in F["e2"]
                if f["strategy"] == s and f["initial_capital_label"] == k
            )["outcomes"]["active_return_ann"]
            v = f["cells"]
            for cap, ls in [(0, "-"), (1, "--")]:
                ax.plot(
                    [0, 1],
                    [v[f"0{cap}"] * 1e4, v[f"1{cap}"] * 1e4],
                    ls,
                    marker=["o", "^", "D"][j],
                    color=palette(j),
                    mfc="white" if cap else palette(j),
                    label=caplabel[k] if cap == 0 else None,
                )
            dat.append(
                dict(
                    strategy=s,
                    capital=k,
                    cells_bps={a: b * 1e4 for a, b in v.items()},
                    interaction_bps=f["raw_interaction"] * 1e4,
                    numerically_zero=f["numerically_zero"],
                )
            )
        ax.set_xticks([0, 1], ["Cost off", "Cost on"])
        ax.set_ylabel("Active vs frictionless (bps/year)")
        panel(ax, string.ascii_lowercase[i], ("Weekly EW" if s == "A" else "Daily MOM"))
        zero(ax, "y")
        ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 2]
    vals = [d["interaction_bps"] for d in dat]
    ax.hlines(range(6), 0, vals, color=[palette(i % 3) for i in range(6)])
    ax.scatter(vals, range(6), c=[palette(i % 3) for i in range(6)], marker="o")
    ax.set_yticks(range(6), glabel)
    ax.invert_yaxis()
    ax.axvspan(-5, 5, color="#eeeeee")
    zero(ax)
    ax.axvline(5, color=GRAY, ls=":")
    ax.axvline(-5, color=GRAY, ls=":")
    ax.set_xlim(-15, 780)
    ax.set_xlabel(r"Raw $I_A$ (bps/year)")
    panel(ax, "c", "Arithmetic-return interaction")
    for j, d in enumerate(dat):
        text = (
            f"{d['interaction_bps']:.2f}"
            if abs(d["interaction_bps"]) >= 5
            else (
                r"$0\;[\mathrm{NZ}]$"
                if d["numerically_zero"]
                else f"{d['interaction_bps']:.3f}"
            )
        )
        ax.annotate(
            text,
            (d["interaction_bps"], j),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=9,
        )
    foot(
        fig,
        "Filled/solid: no limit; open/dashed: 5% ADV limit.\nGray band: ±5 bps/year economic threshold. NZ: numerical zero; not statistical significance.",
    )
    return fig, dat


def F6():
    fig, axs = grid(2, 2, 4.4)
    base = name("C", "1b", 0, 0)
    ph = load(base, "posthoc_diagnostics")
    series = [
        (name("C", "1b", 1, 0), BLUE, "-", "Cost only"),
        (name("C", "1b"), ORANGE, "--", "Both on"),
    ]
    data = {}
    for n, col, ls, label in series:
        d = load(n)
        x = d.index
        axs[0, 0].plot(x, (d.nav / ph.nav - 1) * 100, color=col, ls=ls, label=label)
        axs[0, 1].plot(x, d.cost.cumsum() / 1e9 * 100, color=col, ls=ls, label=label)
        axs[1, 0].plot(x, d.cash_weight * 100, color=col, ls=ls, lw=0.35, alpha=0.3)
        cashmean = d.cash_weight.rolling(63, min_periods=63).mean() * 100
        axs[1, 0].plot(x, cashmean, color=col, ls=ls, lw=1.25)
        rolling = d.turnover.rolling(63, min_periods=63).mean() * 252
        axs[1, 1].plot(x, rolling, color=col, ls=ls)
        data[n] = dict(
            dates=[str(v) for v in x],
            relative_wealth_pct=((d.nav / ph.nav - 1) * 100).tolist(),
            cumulative_cost_initial_pct=(d.cost.cumsum() / 1e9 * 100).tolist(),
            cash_weight_pct=(d.cash_weight * 100).tolist(),
            cash_mean_63_pct=[None if pd.isna(v) else float(v) for v in cashmean],
            rolling_turnover=[None if pd.isna(v) else float(v) for v in rolling],
        )
    axs[0, 1].plot(
        ph.index,
        ph.effective_debit.cumsum() / 1e9 * 100,
        color=GRAY,
        ls=":",
        label="Post-hoc effective debit",
    )
    low = load(name("C", "10m"))
    lowph = load(name("C", "10m", 0, 0), "posthoc_diagnostics")
    lowrel = (low.nav / lowph.nav - 1) * 100
    axs[0, 0].plot(
        low.index, lowrel, color=GREEN, ls="-.", lw=0.9, label="$10M both on (a)"
    )
    data["lower_capital_relative_wealth"] = dict(
        arm=name("C", "10m"),
        reference=name("C", "10m", 0, 0),
        values_pct=lowrel.tolist(),
    )
    for ax in axs.flat:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    for i, (title, ylabel) in enumerate(
        [
            ("MOM: matched relative wealth", "Stateful / post-hoc − 1 (%)"),
            (
                "Modeled debit, not terminal wealth loss",
                "Cumulative debit / initial NAV (%)",
            ),
            ("Cash: daily + 63-session mean", "Cash / closing NAV (%)"),
            ("Trading intensity (exploratory)", "63-session annualized gross turnover"),
        ]
    ):
        panel(axs.flat[i], string.ascii_lowercase[i], title)
        axs.flat[i].set_ylabel(ylabel)
    zero(axs[0, 0], "y")
    h, l = axs[0, 0].get_legend_handles_labels()
    h2, l2 = axs[0, 1].get_legend_handles_labels()
    fig.legend(
        [h[0], h[1], h[2], h2[-1]],
        ["$1B cost only", "$1B both on", "$10M both on (a)", "Post-hoc debit (b)"],
        loc="outside upper center",
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )
    return fig, data


def contrastbars(ax, rows, title, label, threshold=True):
    vals = [c["terminal_gap_ann_bps"] for c in rows]
    y = np.arange(len(rows))
    labels = []
    for c in rows:
        n = c["arm"]
        if n.startswith("E5"):
            parameter, value = n[3:].rsplit("_", 1)
            value = float(value)
            symbol = {
                "a_bps": "a",
                "b": "b",
                "c": "c",
                "half_spread_bps": "h",
                "phi": r"\phi",
            }[parameter]
            display = f"{value * 100:g}\\%" if parameter == "phi" else f"{value:g}"
            labels.append(
                "$"
                + symbol
                + "="
                + display
                + "$"
                + (" bps" if parameter in ["a_bps", "half_spread_bps"] else "")
            )
        else:
            labels.append(policy_label(n))
    ax.hlines(y, 0, vals, color=BLUE)
    ax.plot(vals, y, "o", color=BLUE)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    zero(ax)
    if threshold:
        ax.axvspan(-5, 5, color="#eeeeee", zorder=-1)
    if rows and rows[0]["experiment"] == "E5":
        for cut in [1.5, 3.5, 5.5, 7.5]:
            ax.axhline(cut, color="#dddddd", lw=0.5, zorder=-2)
    ax.set_xlabel("Annualized catch-up ratio (bps/year)")
    panel(ax, label, title)


def F7():
    fig, axs = grid(1, 2, 4)
    e5 = [c for c in C if c["experiment"] == "E5"]
    pol = [c for c in C if c["experiment"] in ["E3", "E4"]]
    contrastbars(axs[0, 0], e5, "All local parameter deviations", "a")
    contrastbars(axs[0, 1], pol, "All residual / cash diagnostics", "b")
    foot(
        fig,
        "Positive = arm poorer; negative = arm richer. Gray band: separate +/-5 bps catch-up threshold.",
    )
    return fig, [
        dict(contrast_id=c["contrast_id"], catchup_bps=c["terminal_gap_ann_bps"])
        for c in e5 + pol
    ]


def S1():
    fig, axs = grid(1, 2, 3.0)
    data = {}
    ax = axs[0, 0]
    for j, k in enumerate(caplabel):
        n = name("C", k, 1, 0)
        fills = pd.read_csv(D / f"{n}.fills.csv", float_precision="round_trip")
        x = np.sort(fills.participation.to_numpy())
        y = (len(x) - np.arange(len(x))) / len(x)
        ax.step(
            x,
            y * 100,
            where="post",
            color=palette(j),
            ls=["-", "--", ":"][j],
            label=caplabel[k],
        )
        data[n] = dict(participation=x.tolist(), survival_pct=(y * 100).tolist())
    ax.set_xscale("symlog", linthresh=0.001, linscale=0.45)
    ax.set_xticks([0, 0.001, 0.01, 0.1, 1, 10], ["0", "0.1", "1", "10", "100", "1000"])
    ax.axvline(0.05, color=GRAY, ls=":")
    ax.axvline(1, color=BLACK, ls="--")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Participation (% ADV; linear below 0.1%)")
    ax.set_ylabel("Fill-row survival (%)")
    ax.legend(frameon=False)
    panel(ax, "a", "Uncapped cost-only MOM")
    for value, label in [(0.05, "5% ADV"), (1, "100% ADV")]:
        ax.annotate(
            label,
            (value, 52),
            xytext=(3, 0),
            textcoords="offset points",
            rotation=90,
            va="center",
            fontsize=9,
        )
    ax = axs[0, 1]
    xx = np.arange(3)
    for i, (f, label) in enumerate(
        [("rows", "Rows"), ("notional", "Notional"), ("cost", "Direct costs")]
    ):
        vals = [M.loc[name("C", k, 1, 0), "part_gt_1_" + f] * 100 for k in caplabel]
        ax.bar(
            xx + (i - 1) * 0.23,
            vals,
            0.23,
            color=[BLUE, GRAY, ORANGE][i],
            edgecolor=BLACK,
            linewidth=0.3,
            hatch=["", "///", "xx"][i],
            label=label,
        )
        data[label] = vals
    ax.set_xticks(xx, list(caplabel.values()))
    ax.set_ylabel("Fraction above 100% ADV (%)")
    ax.legend(frameon=False)
    panel(ax, "b", "Extreme-fill concentration")
    return fig, data


def S2():
    fig, axs = grid(1, 2, 3.2)
    data = {}
    ax = axs[0, 0]
    labels = []
    spread = []
    impact = []
    for s, k in groups:
        for cap in [0, 1]:
            n = name(s, k, 1, cap)
            r = M.loc[n]
            labels.append(("E" if s == "A" else "M") + k + ("/B" if cap else "/C"))
            spread.append(r.spread_cost / r.notional * 1e4)
            impact.append(r.impact_cost / r.notional * 1e4)
    x = np.arange(12)
    ax.bar(x, spread, color=GRAY, label="Half-spread")
    ax.bar(
        x,
        impact,
        bottom=spread,
        color=BLUE,
        edgecolor=BLACK,
        linewidth=0.3,
        hatch="///",
        label="Impact",
    )
    ax.set_xticks(
        np.arange(6) * 2 + 0.5,
        [(("EW" if s == "A" else "MOM") + "\n" + caplabel[k]) for s, k in groups],
    )
    ax.set_xticks(x, ["C", "B"] * 6, minor=True)
    ax.tick_params(axis="x", which="major", pad=28, length=0)
    ax.tick_params(axis="x", which="minor", pad=1, length=0, labelsize=8)
    ax.set_ylabel("Direct cost / executed notional (bps)")
    ax.legend(frameon=False)
    panel(ax, "a", "C: cost only; B: both on")
    ax.tick_params(axis="x", which="major", pad=17, length=0)
    ax.tick_params(axis="x", which="minor", pad=1, length=0)
    f = pd.read_csv(
        D / f"{name('C', '1b', 1, 0)}.fills.csv", float_precision="round_trip"
    )
    p = f.participation.to_numpy()
    z = (f.impact_cost_recalc / f.notional.abs() * 1e4) / (f.sigma_lagged / 0.01)
    ax = axs[0, 1]
    ax.scatter(p, z, s=2, color=GRAY, alpha=0.5, rasterized=False)
    xp = np.geomspace(p.min(), p.max(), 400)
    ax.plot(
        xp, 15 * np.minimum(xp, 1) ** 0.7, color=ORANGE, lw=1.2, label="Frozen formula"
    )
    ax.set_xscale("symlog", linthresh=0.001, linscale=0.45)
    ax.set_xticks([0, 0.001, 0.01, 0.1, 1, 10], ["0", "0.1", "1", "10", "100", "1000"])
    ax.set_xlabel("Participation (% ADV; linear below 0.1%)")
    ax.set_ylabel("Impact rate / (sigma / 0.01) (bps)")
    ax.axvline(1, color=BLACK, ls="--")
    ax.legend(frameon=False)
    panel(ax, "b", "Observed fills and model saturation")
    data = dict(
        labels=labels,
        spread_bps=spread,
        impact_bps=impact,
        participation=p.tolist(),
        normalized_impact_bps=z.tolist(),
    )
    return fig, data


def S3():
    fig, axs = grid(1, 2, 3.5)
    data = []
    for ii, typ in enumerate(["cost_feedback", "feedback_plus_liquidity"]):
        ax = axs[0, ii]
        for j, c in enumerate([c for c in prim if c["effect_class"] == typ]):
            y = [p["active_return_ann"] * 1e4 for p in c["subperiods"]]
            ax.plot(
                range(3),
                y,
                marker=["o", "^", "D"][j % 3],
                mfc="white" if j < 3 else palette(j % 3),
                color=palette(j % 3),
                ls="--" if j < 3 else "-",
                label=glabel[j],
            )
            data.append(dict(contrast_id=c["contrast_id"], subperiod_active_bps=y))
        ax.set_xticks(
            range(3),
            ["Apr 2010–\nDec 2014", "Jan 2015–\nDec 2019", "Jan 2020–\nJun 2026"],
        )
        ax.set_ylabel("Arithmetic active return (bps/year)")
        zero(ax, "y")
        panel(ax, string.ascii_lowercase[ii], typ.replace("_", " ").capitalize())
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )
    return fig, data


def S4():
    fig, axs = grid(1, 2, 3.8)
    data = []
    for j, c in enumerate(prim):
        row = {}
        for block, off, col, mark in [
            ("20", -0.09, BLUE, "o"),
            ("60", 0.09, GRAY, "s"),
        ]:
            z = c["bootstrap"][block]["statistics"]["active_return_ann"]
            lo = z["lower"] * 1e4
            hi = z["upper"] * 1e4
            point = z["point"] * 1e4
            axs[0, 0].hlines(
                j + off, lo, hi, color=col, lw=0.9, ls="-" if block == "20" else "--"
            )
            axs[0, 0].plot(
                point,
                j + off,
                mark,
                color=col,
                ms=3,
                mfc=col if block == "20" else "white",
            )
            row[block] = hi - lo
        axs[0, 1].plot(
            row["20"],
            row["60"],
            "o" if c["effect_class"] == "cost_feedback" else "s",
            mfc=BLUE if "_A_" in c["arm"] else "white",
            color=BLUE if "_A_" in c["arm"] else ORANGE,
        )
        data.append(
            dict(contrast_id=c["contrast_id"], width20=row["20"], width60=row["60"])
        )
    ax = axs[0, 0]
    ax.set_yticks(
        range(12),
        [
            (
                ("EW" if "_A_" in c["arm"] else "MOM")
                + " "
                + caplabel[c["arm"].split("_")[2]]
                + " | "
                + ("F" if c["effect_class"] == "cost_feedback" else "C")
            )
            for c in prim
        ],
    )
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel("Arithmetic active return (bps/year)")
    panel(ax, "a", "Primary pointwise intervals")
    ax = axs[0, 1]
    lims = max(max(d["width20"], d["width60"]) for d in data)
    ax.plot([0, lims], [0, lims], color=GRAY, ls=":")
    ax.set_xlabel("Block-20 CI width (bps/year)")
    ax.set_ylabel("Block-60 CI width (bps/year)")
    panel(ax, "b", "Width sensitivity (linear scales)")
    ax.text(
        0.04,
        0.92,
        "Filled: EW; open: MOM\nCircle: feedback (F)\nSquare: combined (C)",
        transform=ax.transAxes,
        va="top",
    )
    foot(
        fig,
        "F: cost feedback. C: feedback plus liquidity. Solid: block 20; dashed: block 60.",
    )
    return fig, data


def S5():
    fig, axs = grid(1, 2, 4.7)
    out = F["e2b"]["outcomes"]["terminal_gap_bps"]
    cells = sorted(out["cells"])
    effects = out["factorial_effects_2beta"]
    ax = axs[0, 0]
    v = [out["cells"][c] for c in cells]
    ax.hlines(range(16), 0, v, color=GRAY)
    ax.plot(v, range(16), "o", color=BLUE)
    ax.set_yticks(range(16), cells)
    ax.invert_yaxis()
    ax.set_xlabel("Terminal wealth gap (bps)")
    zero(ax)
    panel(ax, "a", "All 16 cells: C, M, B, U bit order")
    terms = list(effects)
    labels = [
        "$"
        + r"\times".join(r"\mathsf{" + "CMBU"[int(i)] + "}" for i in t.split(","))
        + "$"
        for t in terms
    ]
    vals = [effects[t] for t in terms]
    ax = axs[0, 1]
    ax.hlines(range(15), 0, vals, color=GRAY)
    ax.scatter(vals, range(15), color=ORANGE)
    ax.set_yticks(range(15), labels)
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel(r"Classical effect $2\beta$ (bps)")
    panel(ax, "b", "C cost; M minimum; B no borrowing; U integer")
    return fig, dict(cells=dict(zip(cells, v)), effects=effects)


def S6():
    fig, axs = grid(1, 2, 3.5)
    tim = [c for c in C if c["experiment"] == "E1"]
    ax = axs[0, 0]
    for j, c in enumerate(tim):
        valid = c["arm"] == "E1_open_lagged"
        x = c["active_return_ann"] * 1e4
        ax.hlines(j, 0, x, color=GRAY)
        ax.plot(x, j, "o" if valid else "x", color=BLUE if valid else ORANGE)
    ax.set_yticks(
        range(5), [c["arm"].replace("E1_", "").replace("_", " ") for c in tim]
    )
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel("Arithmetic active return (bps/year)")
    panel(ax, "a", "E1: crosses = invalid controls")
    pol = [c for c in C if c["experiment"] in ["E3", "E4"]]
    ax = axs[0, 1]
    for j, c in enumerate(pol):
        ax.plot(
            c["arm_completion"] * 100,
            c["active_return_ann"] * 1e4,
            marker=["o", "^", "v", "D", "s", "P"][j],
            ls="none",
            color=palette(j),
            label=policy_label(c["arm"]),
        )
    zero(ax, "y")
    ax.set_xlabel("Own-demand completion (%)")
    ax.set_ylabel("Arithmetic active return (bps/year)")
    ax.margins(0.15)
    panel(ax, "b", "Residual and cash diagnostics")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    return fig, [
        dict(
            contrast_id=c["contrast_id"],
            active_bps=c["active_return_ann"] * 1e4,
            completion=c["arm_completion"],
        )
        for c in tim + pol
    ]


def S7():
    fig, axs = grid(3, 4, 5.5)
    data = {}
    for row, k in enumerate(caplabel):
        d = load(name("C", k))
        ph = load(name("C", k, 0, 0), "posthoc_diagnostics")
        series = [
            (d.nav / ph.nav - 1) * 100,
            d.part_max * 100,
            d.cash_weight * 100,
            d.wte * 100,
        ]
        for col, y in enumerate(series):
            ax = axs[row, col]
            ax.plot(d.index, y, color=palette(row), lw=0.4, alpha=0.8)
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(10))
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
            ax.set_title(
                "("
                + string.ascii_lowercase[row * 4 + col]
                + ")"
                + (
                    " "
                    + [
                        "Relative wealth (%)",
                        "Max participation (%)",
                        "Cash weight (%)",
                        "Risky L2 × 100",
                    ][col]
                    if row == 0
                    else ""
                ),
                loc="left",
                pad=5,
            )
            if col == 1:
                ax.set_ylim(0, 5.25)
            if col == 2:
                ax.set_ylim(0, 103)
            if col == 0:
                ax.set_ylabel(caplabel[k])
                zero(ax, "y")
        data[k] = dict(
            terminal_relative_pct=series[0].iloc[-1],
            maximum_participation_pct=series[1].max(),
            mean_cash_pct=series[2].mean(),
            mean_tracking=series[3].mean(),
        )
    foot(
        fig,
        "MOM both-on, full period; independent y scales across rows (disclosed by ticks).",
    )
    return fig, data


def S8():
    fig, axs = grid(1, 2, 4.0)
    names = [n for n in S if n.startswith("E3")] + [
        name("A", "1b"),
        name("C", "100m"),
        name("C", "1b"),
    ]
    ax = axs[0, 0]
    data = []
    for j, n in enumerate(names):
        z = pd.read_csv(D / f"{n}.ages.csv", index_col=0).reindex(
            range(6), fill_value=0
        )
        ax.step(
            z.index,
            z["count"] / z["count"].sum() * 100,
            where="mid",
            marker=["o", "s", "^", "v", "D", "P", "X"][j],
            ls=["-", "--", ":", "-.", "-", "--", ":"][j],
            color=palette(j),
            label=policy_label(n),
        )
    ax.set_xlabel("Working-parent age (integer sessions)")
    ax.set_xticks(range(6))
    ax.set_ylabel("Parent-row share (%)")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
    )
    panel(ax, "a", "Discrete parent-lifecycle duration")
    ax = axs[0, 1]
    for j, n in enumerate(names):
        r = M.loc[n]
        x = r.completion * 100
        y = r.overshoot / r.originated * 100
        ax.plot(
            x,
            y,
            marker=["o", "s", "^", "v", "D", "P", "X"][j],
            color=palette(j),
            ls="none",
        )
        data.append(dict(name=n, completion_pct=x, overshoot_origin_pct=y))
    ax.set_xlabel("Own-demand completion (%)")
    ax.set_ylabel("Gross overshoot / originated value (%)")
    ax.margins(0.2)
    panel(ax, "b", "Overshoot is separate from service")
    return fig, data


def F1_compact():
    return F1(compact=True)


def F3_compact():
    return F3(compact=True)


def F4_compact():
    return F4(compact=True)


def export(key, fig, data):
    con = next(c for c in contract_records() if c["figure_id"] == key)
    base = P / "figures" / key
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer)
            fb = fig.bbox
            if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5:
                bounds.append(t.get_text())
    outputs = {}
    for ext, dpi in [("pdf", None), ("png", 600), ("1200.png", 1200), ("svg", None)]:
        path = base.with_suffix("." + ext)
        kw = dict(facecolor="white")
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
        outputs[str(path.relative_to(P))] = sha(path)
    preview = base.with_suffix(".png")
    im = Image.open(preview)
    im.thumbnail((1600, 1300))
    im.save(P / "review" / f"{key}.png")
    ImageOps.grayscale(im).save(P / "review" / f"{key}.gray.png")
    (D / f"{key}.plot_data.json").write_text(
        json.dumps(data, indent=1, allow_nan=False, default=lambda x: float(x))
    )
    ev = json.loads((P / "manifests/evidence_verification.json").read_text())
    sources = {
        f: h
        for f, h in ev["sources"].items()
        if f.startswith("experiments/v2_grid/analysis/")
        or f.startswith("experiments/v2_grid/derived/")
        or any(f.startswith(S[n]["run_directory"] + "/") for n in con["scenario_ids"])
        or (key in ["F1"] and f == "EXPERIMENTAL_PROTOCOL_V2_FINAL.md")
    }
    for p in [
        Path(__file__),
        P / "manifests/evidence_verification.json",
        P / "paper_v2_ieee.mplstyle",
        D / f"{key}.plot_data.json",
    ] + [p for p in D.glob("*.csv")]:
        sources[str(p)] = sha(p)
    if key.startswith("F1"):
        for p in (
            R / "implementation/v2-checkout/src/ledger_consistent_etf_trading/v2"
        ).glob("*.py"):
            sources[str(p)] = sha(p)
    manifest = dict(
        contract=con,
        generator_sha256=sha(Path(__file__)),
        style_sha256=sha(P / "paper_v2_ieee.mplstyle"),
        sources=sources,
        outputs=outputs,
        clipped_text_candidates=bounds,
        exports="vector PDF/SVG; native 600 and 1200 DPI PNG; unscaled final-size canvas",
        matplotlib=matplotlib.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
    )
    (P / "manifests" / f"{key}.json").write_text(json.dumps(manifest, indent=2))
    plt.close(fig)
    print(key, "exported; clipping candidates:", bounds, flush=True)


def contact():
    keys = list(CONTRACTS)
    nr = int(np.ceil(len(keys) / 3))
    fig, axs = plt.subplots(nr, 3, figsize=(12, nr * 3), layout="constrained")
    for ax, k in zip(axs.flat, keys):
        ax.imshow(Image.open(P / "review" / f"{k}.png"))
        ax.set_title(k + " | " + CONTRACTS[k][5], fontsize=11)
        ax.axis("off")
    fig.savefig(P / "review/contact_sheet.pdf")
    fig.savefig(P / "review/contact_sheet.png", dpi=150)
    plt.close(fig)
    with PdfPages(P / "review/publication_size_preview.pdf") as pdf:
        for k in keys:
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.67 / 8.5, 0.48, 7.16 / 8.5, CONTRACTS[k][4] / 11])
            ax.imshow(Image.open(P / "figures" / f"{k}.png"))
            ax.axis("off")
            fig.text(0.08, 0.97, k + " | exact 7.16-inch width", fontsize=12)
            fig.text(
                0.08, 0.12, CONTRACTS[k][6], fontsize=10, wrap=True, parse_math=False
            )
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--figure", default="all")
    args = p.parse_args()
    (P / "FIGURE_CONTRACTS.json").write_text(json.dumps(contract_records(), indent=2))
    (P / "captions.md").write_text(
        "\n\n".join(
            "## " + k + "\n\n" + v[6] + "\n\nLimit: " + v[7]
            for k, v in CONTRACTS.items()
        )
    )
    for k in CONTRACTS:
        if args.figure in ["all", k]:
            fig, data = globals()[k]()
            export(k, fig, data)
    if args.figure == "all":
        contact()
