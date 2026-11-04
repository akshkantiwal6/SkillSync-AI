"""
dashboard.py – Plotly chart helpers for the SkillSync AI performance dashboard.
All charts use the dark navy theme to match the UI.
"""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.express as px

# ── Theme tokens (mirror the CSS variables) ───────────────────────────────────
BG       = "#0B1020"
SURFACE  = "#151D32"
SURFACE2 = "#1E2744"
BORDER   = "#2A3558"
ACCENT   = "#6366F1"
ACCENT2  = "#22D3EE"
TEXT     = "#F8FAFC"
MUTED    = "#CBD5E1"
SUCCESS  = "#22C55E"
WARNING  = "#F59E0B"
DANGER   = "#EF4444"

_LAYOUT_BASE = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE2,
    font=dict(family="'Segoe UI', system-ui, sans-serif", color=TEXT, size=12),
    margin=dict(t=50, b=30, l=20, r=20),
    hoverlabel=dict(bgcolor=SURFACE2, font_color=TEXT, bordercolor=BORDER),
)


def _score_color(score: float, max_score: float = 10.0) -> str:
    pct = score / max_score
    if pct >= 0.8:  return SUCCESS
    if pct >= 0.5:  return ACCENT
    if pct >= 0.3:  return WARNING
    return DANGER


def scores_bar_chart(evaluations: list[dict]) -> go.Figure:
    """Bar chart of scores per question — dark theme."""
    if not evaluations:
        return go.Figure()

    labels = [f"Q{i+1}" for i in range(len(evaluations))]
    scores = [e.get("score", 0) for e in evaluations]
    colors = [_score_color(s) for s in scores]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=scores,
            marker=dict(
                color=colors,
                line=dict(color=BORDER, width=0.5),
            ),
            text=scores,
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            hovertemplate="<b>%{x}</b><br>Score: %{y}/10<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Score per Question", font=dict(color=TEXT, size=14)),
        yaxis=dict(
            range=[0, 12],
            title="Score (out of 10)",
            gridcolor=BORDER,
            color=MUTED,
            zerolinecolor=BORDER,
        ),
        xaxis=dict(title="Question", color=MUTED, gridcolor=BORDER),
        height=340,
        bargap=0.35,
    )
    return fig


def radar_chart(category_scores: dict[str, float]) -> go.Figure:
    """Radar chart for performance across question categories — dark theme."""
    if not category_scores:
        return go.Figure()

    categories = list(category_scores.keys())
    values     = [category_scores[c] for c in categories]
    # Close the radar loop
    cats_loop   = categories + [categories[0]]
    vals_loop   = values    + [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=vals_loop,
            theta=cats_loop,
            fill="toself",
            fillcolor=f"rgba(99,102,241,0.2)",
            line=dict(color=ACCENT, width=2),
            marker=dict(color=ACCENT, size=6),
            name="Your Performance",
            hovertemplate="<b>%{theta}</b><br>Avg Score: %{r:.1f}/10<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        polar=dict(
            bgcolor=SURFACE2,
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor=BORDER,
                color=MUTED,
                tickfont=dict(color=MUTED, size=10),
            ),
            angularaxis=dict(color=MUTED, gridcolor=BORDER),
        ),
        title=dict(text="Performance by Category", font=dict(color=TEXT, size=14)),
        showlegend=False,
        height=360,
    )
    return fig


def skill_gap_donut(matching: int, missing: int) -> go.Figure:
    """Donut chart showing skill match vs gap — dark theme."""
    total = matching + missing
    if total == 0:
        return go.Figure()

    match_pct = int(matching / total * 100)
    fig = go.Figure(
        go.Pie(
            labels=["Matched Skills", "Skill Gaps"],
            values=[matching, missing],
            hole=0.60,
            marker=dict(
                colors=[SUCCESS, DANGER],
                line=dict(color=SURFACE, width=2),
            ),
            textfont=dict(color=TEXT, size=12),
            hovertemplate="<b>%{label}</b><br>%{value} skills (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(text="Skill Match Overview", font=dict(color=TEXT, size=14)),
        height=310,
        showlegend=True,
        legend=dict(
            font=dict(color=MUTED, size=11),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
        ),
        annotations=[
            dict(
                text=f"<b>{match_pct}%</b>",
                x=0.5, y=0.5,
                font=dict(size=22, color=ACCENT2),
                showarrow=False,
            )
        ],
    )
    return fig


def progress_gauge(score_pct: float) -> go.Figure:
    """Gauge chart for overall interview readiness — dark theme."""
    if score_pct >= 80:
        bar_color = SUCCESS
    elif score_pct >= 55:
        bar_color = ACCENT
    elif score_pct >= 35:
        bar_color = WARNING
    else:
        bar_color = DANGER

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score_pct,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text="Interview Readiness", font=dict(color=MUTED, size=13)),
            number=dict(
                suffix="%",
                font=dict(color=TEXT, size=28),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=MUTED,
                    tickfont=dict(color=MUTED, size=10),
                    nticks=6,
                ),
                bar=dict(color=bar_color, thickness=0.7),
                bgcolor=SURFACE2,
                borderwidth=1,
                bordercolor=BORDER,
                steps=[
                    dict(range=[0,  35],  color="#1c1130"),
                    dict(range=[35, 55],  color="#1a1e2e"),
                    dict(range=[55, 80],  color="#131e2e"),
                    dict(range=[80, 100], color="#0d2215"),
                ],
                threshold=dict(
                    line=dict(color=ACCENT2, width=2),
                    thickness=0.85,
                    value=70,
                ),
            ),
        )
    )
    fig.update_layout(
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT),
        margin=dict(t=50, b=20, l=20, r=20),
        height=260,
    )
    return fig


def score_distribution_histogram(all_scores: list[float]) -> go.Figure:
    """Histogram of score distribution — dark theme."""
    if not all_scores:
        return go.Figure()
    fig = px.histogram(
        x=all_scores,
        nbins=10,
        labels={"x": "Score"},
        title="Score Distribution",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(range=[0, 10], title="Score", gridcolor=BORDER, color=MUTED),
        yaxis=dict(title="Count", gridcolor=BORDER, color=MUTED),
        height=300,
        bargap=0.1,
    )
    fig.update_traces(marker_line_color=BORDER, marker_line_width=0.5)
    return fig
