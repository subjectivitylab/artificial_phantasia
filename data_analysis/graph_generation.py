"""Standalone graph generation for the Artificial Phantasia analyses.

This script centralizes graph creation in Python without changing the existing
notebook or R Markdown reports. It reads the current CSV outputs and writes PNG
figures under ``data_analysis/graphs/`` when run.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_CSVS = ROOT / "output_csvs"
STATISTICAL_RESULTS = ROOT / "statistical_results"
GRAPH_DIR = ROOT / "graphs"
FINKE_BLOCKS = set(range(48, 60))
ALL_BLOCKS = set(range(1, 61))
NOVEL_BLOCKS = ALL_BLOCKS - FINKE_BLOCKS
MEAN_CI_LIGHTNESS_FACTOR = 0.62
MEAN_CI_SATURATION_FACTOR = 1.35


@dataclass(frozen=True)
class PlotResult:
    name: str
    path: Path


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _center_value(values: pd.Series | np.ndarray, center_stat: str) -> float:
    if center_stat == "mean":
        return float(np.mean(values))
    return float(np.median(values))


def _center_interval(values: pd.Series | np.ndarray, center_stat: str) -> tuple[float, float]:
    if len(values) <= 1:
        center = _center_value(values, center_stat)
        return center, center
    if center_stat == "mean":
        center = _center_value(values, "mean")
        se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        return center - 1.96 * se, center + 1.96 * se
    low, high = np.percentile(values, [25, 75])
    return float(low), float(high)


def _grouped_values_by_center(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    center_stat: str = "median",
) -> list[tuple[object, pd.Series, float]]:
    groups = []
    for name, group in frame.groupby(group_col, sort=False):
        values = _numeric(group[value_col])
        if len(values):
            groups.append((name, values, _center_value(values, center_stat)))
    return sorted(groups, key=lambda item: item[2], reverse=True)


def _save(
    fig: plt.Figure,
    out_dir: Path,
    name: str,
    show: bool = False,
    *,
    tight_layout: bool = True,
) -> PlotResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return PlotResult(name=name, path=path)


def _mean_ci_color(color: str) -> str:
    red, green, blue = to_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    lightness = max(0, min(1, lightness * MEAN_CI_LIGHTNESS_FACTOR))
    saturation = max(0, min(1, saturation * MEAN_CI_SATURATION_FACTOR))
    return matplotlib.colors.to_hex(colorsys.hls_to_rgb(hue, lightness, saturation))


def _hist(
    values: pd.Series,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    bins: int | Sequence[float] = 10,
    color: str = "skyblue",
    mean_line: float | None = None,
    density: bool = False,
    xlim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(_numeric(values), bins=bins, color=color, edgecolor="black", alpha=0.8, density=density)
    if mean_line is not None and not math.isnan(mean_line):
        ax.axvline(mean_line, color="red", linestyle="--", linewidth=2, label="Mean")
        ax.legend()
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)
    return fig


def _boxplot_by_group(
    frame: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    horizontal: bool = False,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    groups = _grouped_values_by_center(frame, group_col, value_col)
    labels = [str(name) for name, _, _ in groups]
    values = [values for _, values, _ in groups]

    fig, ax = plt.subplots(figsize=figsize)
    if horizontal:
        positions = np.arange(len(values) - 1, -1, -1)
        ax.boxplot(values, positions=positions, vert=False, patch_artist=True, showfliers=True)
        ax.set_yticks(positions, labels)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(title, pad=15)
    ax.grid(True, alpha=0.3, axis="y" if not horizontal else "x")
    return fig


def _distribution_overlay(
    frame: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str = "Density",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for name, values, _ in _grouped_values_by_center(frame, group_col, value_col):
        ax.hist(values, bins=20, density=True, alpha=0.35, label=str(name), edgecolor="black")
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def _plot_half_violins(
    frame: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    out_dir: Path,
    name: str,
    color_col: str | None = None,
    legend_title: str = "Model Family",
    show: bool = False,
    xlim: tuple[float, float] | None = (1, 5),
    center_stat: str = "median",
) -> PlotResult:
    groups = []
    for label, group in frame.groupby(group_col, sort=False):
        values = _numeric(group[value_col])
        if len(values):
            color = group[color_col].iloc[0] if color_col and color_col in group else "#8da0cb"
            center = _center_value(values, center_stat)
            groups.append((str(label), values.to_numpy(dtype=float), color, center))
    groups.sort(key=lambda item: item[3], reverse=True)

    labels = [item[0] for item in groups]
    values = [item[1] for item in groups]
    colors = [item[2] for item in groups]
    positions = np.arange(len(groups) - 1, -1, -1)

    fig, ax = plt.subplots(figsize=(10, max(6, len(groups) * 0.35)))
    fig.subplots_adjust(left=0.34, right=0.94, bottom=0.12, top=0.9)
    if values:
        violin = ax.violinplot(
            values,
            positions=positions,
            vert=False,
            widths=1.8,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for position, body, color in zip(positions, violin["bodies"], colors):
            vertices = body.get_paths()[0].vertices
            vertices[:, 1] = np.maximum(vertices[:, 1], position)
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.42)
            body.set_linewidth(1)

        for position, vals, color in zip(positions, values, colors):
            point_color = _mean_ci_color(color)
            center = _center_value(vals, center_stat)
            ci_low, ci_high = _center_interval(vals, center_stat)
            if xlim:
                ci_low = max(xlim[0], ci_low)
                ci_high = min(xlim[1], ci_high)
            _draw_horizontal_ci(ax, ci_low, ci_high, position, point_color)
            ax.scatter([center], [position], color=point_color, edgecolor="black", linewidth=0.4, s=28, zorder=3)
            if len(vals) <= 8:
                jitter = np.linspace(0.04, 0.24, len(vals))
                ax.scatter(vals, position + jitter, color=color, alpha=0.65, s=12, linewidth=0, zorder=2)

    ax.set_yticks(positions, labels, fontsize=12)
    #ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    #ax.set_ylabel("Model", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3, axis="x")
    if color_col and colors:
        legend_frame = pd.DataFrame({
            "color": colors,
            "shape": [_historical_marker(label) for label in labels],
        })
        _add_historical_legend(ax, legend_frame, legend_title, show_title = False, fontsize=12)
    return _save(fig, out_dir, name, show, tight_layout=False)


def _draw_horizontal_ci(
    ax: plt.Axes,
    low: float,
    high: float,
    y: float,
    color: str,
    *,
    linestyle: str = "-",
) -> None:
    ax.plot(
        [low, high],
        [y, y],
        color=color,
        linestyle=linestyle,
        linewidth=1.2,
        marker="|",
        markersize=9,
        markeredgewidth=1.2,
        zorder=2,
    )


def _facet_boxplots(
    frame: pd.DataFrame,
    *,
    facet_col: str,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_dir: Path,
    name: str,
    show: bool = False,
) -> PlotResult:
    facets = list(frame[facet_col].dropna().unique())
    n = len(facets)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for ax, facet in zip(axes.ravel(), facets):
        sub = frame[frame[facet_col] == facet]
        groups = [(k, _numeric(g[value_col])) for k, g in sub.groupby(group_col, sort=False)]
        labels = [str(k) for k, values in groups if len(values)]
        values = [values for k, values in groups if len(values)]
        ax.boxplot(values, tick_labels=labels, patch_artist=True)
        ax.set_title(str(facet))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, alpha=0.3, axis="y")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    return _save(fig, out_dir, name, show)


def _facet_distributions(
    frame: pd.DataFrame,
    *,
    facet_col: str,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    out_dir: Path,
    name: str,
    show: bool = False,
) -> PlotResult:
    facets = list(frame[facet_col].dropna().unique())
    n = len(facets)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
    for ax, facet in zip(axes.ravel(), facets):
        sub = frame[frame[facet_col] == facet]
        for group_name, group in sub.groupby(group_col, sort=False):
            values = _numeric(group[value_col])
            if len(values):
                ax.hist(values, bins=15, density=True, alpha=0.35, label=str(group_name), edgecolor="black")
        ax.set_title(str(facet))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        ax.legend()
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    return _save(fig, out_dir, name, show)


def _rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    if not len(x) or not len(y):
        return float("nan")
    pooled = pd.Series(np.concatenate([x, y])).rank(method="average").to_numpy()
    rank_x = pooled[: len(x)].sum()
    u_x = rank_x - len(x) * (len(x) + 1) / 2
    return 1 - (2 * u_x) / (len(x) * len(y))


def _mann_whitney_pvalue(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    pooled = pd.Series(np.concatenate([x, y]))
    ranks = pooled.rank(method="average").to_numpy()
    rank_x = ranks[:n1].sum()
    u1 = rank_x - n1 * (n1 + 1) / 2
    _, counts = np.unique(pooled, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    n = n1 + n2
    variance = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0
    if variance <= 0:
        return u1, float("nan")
    z = (u1 - n1 * n2 / 2) / math.sqrt(variance)
    return u1, math.erfc(abs(z) / math.sqrt(2))


def _bonferroni(values: Sequence[float]) -> list[float]:
    vals = list(values)
    n = len(vals)
    return [min(1.0, v * n) if not math.isnan(v) else float("nan") for v in vals]


def _effect_vs_p_plot(
    frame: pd.DataFrame,
    *,
    label_col: str,
    title: str,
    xlabel: str,
    out_dir: Path,
    name: str,
    show: bool = False,
) -> PlotResult:
    plot_data = frame.copy()
    plot_data["p_adj_bonferroni"] = plot_data["p_adj_bonferroni"].clip(lower=np.nextafter(0, 1))
    fig, ax = plt.subplots(figsize=(7, 7))
    y_values = -np.log10(plot_data["p_adj_bonferroni"])
    ax.scatter(plot_data["effsize"], y_values)
    for _, row in plot_data.iterrows():
        ax.annotate(
            str(row[label_col]),
            (row["effsize"], -math.log10(row["p_adj_bonferroni"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0, linestyle="--", color="black", linewidth=1)
    ax.set_title(title, pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("-log10(Bonferroni-adjusted p-value)")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_dir, name, show)


def _human_model_long(llm_file: Path) -> tuple[pd.DataFrame, list[str]]:
    humans = _read_csv(OUTPUT_CSVS / "h_grade_distribution.csv").iloc[:, 0]
    llms = _read_csv(llm_file)
    rows = [pd.DataFrame({"model": "Human", "score": humans})]
    for col in llms.columns:
        rows.append(pd.DataFrame({"model": col, "score": llms[col]}))
    return pd.concat(rows, ignore_index=True), list(llms.columns)


def _with_distribution_colors(frame: pd.DataFrame, group_col: str, task: str = "") -> pd.DataFrame:
    out = frame.copy()
    out["color"] = out[group_col].map(lambda model: _historical_color(task, str(model)))
    return out


def generate_core_pipeline_graphs(
    out_dir: Path = GRAPH_DIR / "core_pipeline",
    show: bool = False,
) -> list[PlotResult]:
    results: list[PlotResult] = []
    tidy_expert = _read_csv(OUTPUT_CSVS / "tidy_expert_data.csv")
    tidy_crowd = _read_csv(OUTPUT_CSVS / "tidy_crowdsourced_data.csv")
    means = _read_csv(OUTPUT_CSVS / "means_with_canon.csv")
    vviq = _read_csv(OUTPUT_CSVS / "vviq_scores.csv")
    difficulty = _read_csv(OUTPUT_CSVS / "difficulty_score_summary.csv")
    h_all = _read_csv(OUTPUT_CSVS / "h_graded_results.csv")
    h_finke = _read_csv(OUTPUT_CSVS / "h_graded_results_finke.csv")
    h_novel = _read_csv(OUTPUT_CSVS / "h_graded_results_novel.csv")
    llm_all = _read_csv(OUTPUT_CSVS / "llm_graded_results.csv")
    llm_finke = _read_csv(OUTPUT_CSVS / "llm_graded_results_finke.csv")
    llm_novel = _read_csv(OUTPUT_CSVS / "llm_graded_results_novel.csv")

    results.append(_save(_hist(tidy_expert["score"], title="Distribution of Expert Scores",
                               xlabel="Score Value", bins=5), out_dir, "expert_scores_histogram", show))

    score_std = tidy_crowd.groupby("block")["score"].std().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(len(score_std)), score_std.values, color="orange", alpha=0.7)
    ax.set_title("Blocks by Score Standard Deviation", pad=15)
    ax.set_xlabel("Score Standard Deviation")
    ax.set_ylabel("Block")
    ax.set_yticks(range(len(score_std)), score_std.index)
    ax.grid(True, alpha=0.3)
    results.append(_save(fig, out_dir, "crowd_block_score_std", show))

    score_by_block = tidy_crowd.groupby("block_num")["score"].mean()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(score_by_block.index[:47], score_by_block.values[:47], color="#377eb8", label="Novel 48")
    ax.bar(score_by_block.index[47:59], score_by_block.values[47:59], color="#4daf4a", label="Finke")
    if len(score_by_block) > 59:
        ax.bar(score_by_block.index[59], score_by_block.values[59], color="#377eb8")
    ax.set_title("Mean Score by Instruction Set", pad=15)
    ax.set_xlabel("Instruction Set")
    ax.set_ylabel("Average Score")
    ax.set_xlim(0, 61)
    ax.grid(True, alpha=0.3)
    ax.legend()
    results.append(_save(fig, out_dir, "crowd_mean_score_by_instruction_set", show))

    results.append(_save(_hist(tidy_crowd["score"], title="Distribution of Scores",
                               xlabel="Score Value", bins=5), out_dir, "crowd_scores_histogram", show))
    results.append(_save(_hist(means["normal_sd_score"], title="Distribution of Score Standard Deviations",
                               xlabel="Standard Deviation", bins=30), out_dir, "canon_score_sd_histogram", show))
    results.append(_save(_hist(means["normal_mean_score"], title="Distribution of Mean Scores",
                               xlabel="Mean Score", bins=30,
                               mean_line=means["normal_mean_score"].mean()), out_dir,
                         "canon_mean_score_histogram", show))

    response_counts = {
        i: [len(vviq[vviq[f"VVIQ - {str(j).zfill(2)}"] == i]) for j in range(1, 17)]
        for i in range(1, 6)
    }
    vviq_vals = np.arange(1, 17)
    width = 0.15
    fig, ax = plt.subplots(figsize=(15, 7))
    for offset_index, (response, counts) in enumerate(response_counts.items()):
        ax.bar(vviq_vals + width * offset_index, counts, width, label=f"Score {response}", alpha=0.7)
    ax.set_xlabel("VVIQ Question Number")
    ax.set_ylabel("Count of Responses")
    ax.set_title("Distribution of Responses for Each VVIQ Question", pad=15)
    ax.set_xticks(vviq_vals + width * 2, range(1, 17))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    results.append(_save(fig, out_dir, "vviq_response_distribution", show))

    vviq_cols = [f"VVIQ - {str(i).zfill(2)}" for i in range(1, 17)]
    correlations = vviq[vviq_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(correlations, cmap="RdYlBu_r", aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(16), range(1, 17), rotation=45)
    ax.set_yticks(range(16), range(1, 17))
    ax.set_xlabel("VVIQ Question Number")
    ax.set_ylabel("VVIQ Question Number")
    ax.set_title("Correlation Matrix of VVIQ Questions", pad=15)
    results.append(_save(fig, out_dir, "vviq_question_correlation_matrix", show))

    results.append(_save(_hist(vviq["VVIQ_sum"], title="Distribution of VVIQ Sum Scores",
                               xlabel="VVIQ Sum Score", bins=20,
                               mean_line=vviq["VVIQ_sum"].mean()), out_dir, "vviq_sum_histogram", show))
    results.append(_save(_hist(difficulty["Difficulty_Score"], title="Distribution of Difficulty Scores",
                               xlabel="Instruction Set", bins=10,
                               mean_line=difficulty["Difficulty_Score"].mean()), out_dir,
                         "difficulty_scores_histogram", show))

    for df, name, title in [
        (h_finke, "human_finke_mean_score_histogram", "Human Distribution of Mean Score per Item (Finke Tasks)"),
        (h_novel, "human_novel_mean_score_histogram", "Human Distribution of Mean Score per Item (Novel Tasks)"),
        (h_all, "human_overall_mean_score_density", "Human Distribution of Mean Scores per Item"),
    ]:
        mean = df["overall_score"].sum() / df["n_total"].sum()
        results.append(_save(_hist(df["mean_score_per_item"], title=title, xlabel="Mean Score per Item",
                                   bins=15, color="salmon", mean_line=mean,
                                   density=name.endswith("density"), xlim=(1, 5) if name.endswith("density") else None),
                             out_dir, name, show))

    results.append(_save(_hist(h_all["overall_score"], title="Distribution of Overall Scores",
                               xlabel="Overall Score", bins=20), out_dir, "human_overall_scores_histogram", show))
    results.append(_save(_hist(h_all["n_graded"], title="Distribution of Number of Graded Items",
                               xlabel="Number of Graded Items", color="lightgreen"), out_dir,
                         "human_n_graded_histogram", show))

    for df, name, title in [
        (llm_finke, "llm_finke_mean_score_histogram", "Overall LLM Distribution of Mean Score per Item (Finke Tasks)"),
        (llm_novel, "llm_novel_mean_score_histogram", "Overall LLM Distribution of Mean Score per Item (Novel Tasks)"),
        (llm_all, "llm_overall_mean_score_histogram", "Overall LLM Distribution of Mean Scores per Item"),
    ]:
        mean = df["overall_score"].sum() / df["n_total"].sum()
        results.append(_save(_hist(df["mean_score_per_item"], title=title, xlabel="Mean Score per Item",
                                   bins=15 if "finke" in name or "novel" in name else 10,
                                   color="salmon" if "finke" in name or "novel" in name else "skyblue",
                                   mean_line=mean, xlim=(1, 5) if "overall" in name else None),
                             out_dir, name, show))

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.bar(range(len(llm_all["Model"])), llm_all["overall_score"], color="skyblue")
    ax.set_xticks(range(len(llm_all["Model"])), llm_all["Model"], rotation=45, ha="right")
    ax.set_title("Overall Score by Model", pad=20)
    ax.set_xlabel("Model Name", labelpad=10)
    ax.set_ylabel("Overall Score", labelpad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    results.append(_save(fig, out_dir, "llm_overall_score_by_model", show))

    results.append(_save(_hist(llm_all["n_graded"], title="Distribution of Number of Graded Items",
                               xlabel="Number of Graded Items", bins=10), out_dir,
                         "llm_n_graded_histogram", show))
    return results


def generate_human_vs_model_graphs(
    *,
    llm_file: Path = OUTPUT_CSVS / "llm_collapsed_grade_dist.csv",
    summary_file: Path = STATISTICAL_RESULTS / "human_vs_model_significance_summary.csv",
    out_dir: Path = GRAPH_DIR / "human_vs_model",
    prefix: str = "human_vs_model",
    color_task: str = "",
    legend_title: str = "Model Family",
    show: bool = False,
) -> list[PlotResult]:
    scores_long, model_cols = _human_model_long(llm_file)
    colored_scores = _with_distribution_colors(scores_long, "model", color_task)
    results = [
        _save(_distribution_overlay(
            scores_long[scores_long["model"].isin(["Human", *model_cols[:3]])],
            group_col="model",
            value_col="score",
            title="Score Distributions: Humans vs Selected Models",
            xlabel="Score",
        ), out_dir, f"{prefix}_selected_distribution", show),
        _save(_boxplot_by_group(
            scores_long,
            group_col="model",
            value_col="score",
            title="Score Distributions: Human vs All LLM Models",
            xlabel="Model",
            ylabel="Score",
            horizontal=True,
            figsize=(10, max(6, 0.35 * scores_long["model"].nunique() + 2)),
        ), out_dir, f"{prefix}_all_boxplot", show),
        _plot_half_violins(
            colored_scores,
            group_col="model",
            value_col="score",
            color_col="color",
            title="Mean Score Distributions: Human vs All LLM Models",
            xlabel="Score",
            out_dir=out_dir,
            name=f"{prefix}_half_violins_mean",
            legend_title=legend_title,
            show=show,
            center_stat="mean",
        ),
        _plot_half_violins(
            colored_scores,
            group_col="model",
            value_col="score",
            color_col="color",
            title="Median Score Distributions: Human vs All LLM Models",
            xlabel="Score",
            out_dir=out_dir,
            name=f"{prefix}_half_violins",
            legend_title=legend_title,
            show=show,
        ),
    ]
    if summary_file.exists():
        summary = _read_csv(summary_file)
        results.append(_effect_vs_p_plot(
            summary,
            label_col="model",
            title="Effect Size vs Adjusted p-value (Human vs Model)",
            xlabel="Rank-biserial effect size",
            out_dir=out_dir,
            name=f"{prefix}_effect_vs_p",
            show=show,
        ))
    return results


def generate_reasoning_comparison_graphs(show: bool = False) -> list[PlotResult]:
    return generate_human_vs_model_graphs(
        llm_file=OUTPUT_CSVS / "openai_reasoning_comparison_grade_dist.csv",
        summary_file=STATISTICAL_RESULTS / "human_vs_reasoning_model_significance_summary.csv",
        out_dir=GRAPH_DIR / "reasoning_comparison",
        prefix="reasoning_comparison",
        color_task="Reasoning",
        legend_title="Reasoning Level",
        show=show,
    )


def _image_scores_long() -> pd.DataFrame:
    dat = _read_csv(OUTPUT_CSVS / "image_comparison_grade_dist.csv")
    image_cols = [c for c in dat.columns if c.endswith("_images")]
    rows = []
    for image_col in image_cols:
        base = image_col.removesuffix("_images")
        if base in dat.columns:
            rows.append(pd.DataFrame({"model_base": base, "variant": "no_images", "score": dat[base]}))
            rows.append(pd.DataFrame({"model_base": base, "variant": "with_images", "score": dat[image_col]}))
    return pd.concat(rows, ignore_index=True)


def _unpaired_summary_by_pair(frame: pd.DataFrame, pair_col: str, group_col: str) -> pd.DataFrame:
    rows = []
    for pair, sub in frame.groupby(pair_col, sort=False):
        groups = list(sub[group_col].dropna().unique())
        if len(groups) != 2:
            continue
        x = _numeric(sub[sub[group_col] == groups[0]]["score"])
        y = _numeric(sub[sub[group_col] == groups[1]]["score"])
        _, p = _mann_whitney_pvalue(x, y)
        rows.append({"label": pair, "effsize": _rank_biserial(x, y), "p.value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj_bonferroni"] = _bonferroni(out["p.value"])
    return out


def _signed_rank_summary(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in wide.groupby("model", sort=False):
        diffs = _numeric(sub["mc"] - sub["sc"])
        diffs = diffs[diffs != 0]
        n = len(diffs)
        if n == 0:
            rows.append({"label": model, "effsize": 0.0, "p.value": 1.0})
            continue
        ranks = diffs.abs().rank(method="average")
        w_plus = ranks[diffs > 0].sum()
        w_minus = ranks[diffs < 0].sum()
        total_rank = ranks.sum()
        effsize = (w_plus - w_minus) / total_rank if total_rank else 0.0
        mean_w = n * (n + 1) / 4
        var_w = n * (n + 1) * (2 * n + 1) / 24
        p_value = math.erfc(abs((min(w_plus, w_minus) - mean_w) / math.sqrt(var_w)) / math.sqrt(2))
        rows.append({"label": model, "effsize": effsize, "p.value": p_value})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj_bonferroni"] = _bonferroni(out["p.value"])
    return out


def generate_image_comparison_graphs(show: bool = False) -> list[PlotResult]:
    out_dir = GRAPH_DIR / "image_comparison"
    scores_long = _image_scores_long()
    colored_scores = _with_distribution_colors(
        scores_long.assign(group=scores_long["model_base"] + " " + scores_long["variant"]),
        "group",
    )
    results = [
        _facet_boxplots(scores_long, facet_col="model_base", group_col="variant", value_col="score",
                        title="Model Scores: With Images vs Without Images",
                        xlabel="Variant", ylabel="Score", out_dir=out_dir,
                        name="image_comparison_boxplots", show=show),
        _facet_distributions(scores_long, facet_col="model_base", group_col="variant", value_col="score",
                             title="Score Distributions by Model and Image Variant",
                             xlabel="Score", out_dir=out_dir,
                             name="image_comparison_distributions", show=show),
        _plot_half_violins(
            colored_scores,
            group_col="group",
            value_col="score",
            color_col="color",
            title="Mean Score Distributions by Model and Image Variant",
            xlabel="Score",
            out_dir=out_dir,
            name="image_comparison_half_violins_mean",
            legend_title="Model Family",
            show=show,
            center_stat="mean",
        ),
        _plot_half_violins(
            colored_scores,
            group_col="group",
            value_col="score",
            color_col="color",
            title="Median Score Distributions by Model and Image Variant",
            xlabel="Score",
            out_dir=out_dir,
            name="image_comparison_half_violins",
            legend_title="Model Family",
            show=show,
        ),
    ]
    summary = _unpaired_summary_by_pair(scores_long, "model_base", "variant")
    if not summary.empty:
        results.append(_effect_vs_p_plot(summary, label_col="label",
                                         title="Effect Size vs Adjusted p-value (With Images vs Without)",
                                         xlabel="Rank-biserial effect size (with vs without images)",
                                         out_dir=out_dir, name="image_comparison_effect_vs_p", show=show))
    return results


def generate_qwen_comparison_graphs(show: bool = False) -> list[PlotResult]:
    out_dir = GRAPH_DIR / "qwen_comparison"
    dat = _read_csv(OUTPUT_CSVS / "qwen_comparison_grade_dist.csv")
    scores_long = dat.melt(var_name="model", value_name="score")
    colored_scores = _with_distribution_colors(scores_long, "model")
    return [
        _save(_distribution_overlay(scores_long, group_col="model", value_col="score",
                                    title="Score Distributions: Independent Model Comparison",
                                    xlabel="Score"), out_dir, "qwen_comparison_distribution", show),
        _save(_boxplot_by_group(scores_long, group_col="model", value_col="score",
                                title="Boxplot Comparison", xlabel="Model", ylabel="Score",
                                figsize=(8, 5)), out_dir, "qwen_comparison_boxplot", show),
        _plot_half_violins(colored_scores,
                           group_col="model", value_col="score", color_col="color",
                           title="Qwen Mean Score Distributions",
                           xlabel="Score", out_dir=out_dir,
                           name="qwen_comparison_half_violins_mean",
                           legend_title="Model Family", show=show,
                           center_stat="mean"),
        _plot_half_violins(colored_scores,
                           group_col="model", value_col="score", color_col="color",
                           title="Qwen Median Score Distributions",
                           xlabel="Score", out_dir=out_dir,
                           name="qwen_comparison_half_violins",
                           legend_title="Model Family", show=show),
    ]


def generate_temperature_graphs(show: bool = False) -> list[PlotResult]:
    out_dir = GRAPH_DIR / "temperature_comparison"
    dat = _read_csv(OUTPUT_CSVS / "gemini_3_pro_temp_grade_dist.csv")
    long_df = dat.melt(var_name="temperature", value_name="score")
    colored_scores = _with_distribution_colors(long_df, "temperature", "Temperature")
    return [
        _save(_boxplot_by_group(long_df, group_col="temperature", value_col="score",
                                title="Score Distributions by Temperature Setting",
                                xlabel="Temperature", ylabel="Score",
                                figsize=(8, 5)), out_dir, "temperature_boxplot", show),
        _plot_half_violins(colored_scores,
                           group_col="temperature", value_col="score", color_col="color",
                           title="Mean Score Distributions by Temperature Setting",
                           xlabel="Score", out_dir=out_dir,
                           name="temperature_half_violins_mean",
                           legend_title="Model Family", show=show,
                           center_stat="mean"),
        _plot_half_violins(colored_scores,
                           group_col="temperature", value_col="score", color_col="color",
                           title="Median Score Distributions by Temperature Setting",
                           xlabel="Score", out_dir=out_dir,
                           name="temperature_half_violins",
                           legend_title="Model Family", show=show),
    ]


def _paired_sc_mc_long() -> tuple[pd.DataFrame, pd.DataFrame]:
    dat = _read_csv(OUTPUT_CSVS / "single_vs_multiple_context_grade_dist.csv")
    sc_cols = [c for c in dat.columns if c.endswith("_sc")]
    models = [c.removesuffix("_sc") for c in sc_cols if f"{c.removesuffix('_sc')}_mc" in dat.columns]
    wide_rows = []
    long_rows = []
    for model in models:
        sc = dat[f"{model}_sc"]
        mc = dat[f"{model}_mc"]
        item_id = range(1, len(dat) + 1)
        wide_rows.append(pd.DataFrame({"item_id": item_id, "model": model, "sc": sc, "mc": mc}))
        long_rows.append(pd.DataFrame({"item_id": item_id, "model": model, "context": "sc", "score": sc}))
        long_rows.append(pd.DataFrame({"item_id": item_id, "model": model, "context": "mc", "score": mc}))
    return pd.concat(wide_rows, ignore_index=True), pd.concat(long_rows, ignore_index=True)


def generate_sc_vs_mc_graphs(show: bool = False) -> list[PlotResult]:
    out_dir = GRAPH_DIR / "sc_vs_mc"
    wide, scores_long = _paired_sc_mc_long()
    colored_scores = _with_distribution_colors(
        scores_long.assign(group=scores_long["model"] + "_" + scores_long["context"]),
        "group",
    )
    results = [
        _facet_boxplots(scores_long, facet_col="model", group_col="context", value_col="score",
                        title="Single vs Multi-Context Scores (sc vs mc)",
                        xlabel="Context", ylabel="Score", out_dir=out_dir,
                        name="sc_vs_mc_boxplots", show=show)
    ]
    results.append(_plot_half_violins(
        colored_scores,
        group_col="group",
        value_col="score",
        color_col="color",
        title="Single vs Multi-Context Mean Scores",
        xlabel="Score",
        out_dir=out_dir,
        name="sc_vs_mc_half_violins_mean",
        legend_title="Model Family / Context",
        show=show,
        center_stat="mean",
    ))
    results.append(_plot_half_violins(
        colored_scores,
        group_col="group",
        value_col="score",
        color_col="color",
        title="Single vs Multi-Context Median Scores",
        xlabel="Score",
        out_dir=out_dir,
        name="sc_vs_mc_half_violins",
        legend_title="Model Family / Context",
        show=show,
    ))
    diff_long = wide.assign(diff=wide["mc"] - wide["sc"])
    results.append(_save(_boxplot_by_group(diff_long, group_col="model", value_col="diff",
                                           title="Paired Difference (mc - sc) per Model",
                                           xlabel="Model", ylabel="Difference in score (mc - sc)",
                                           horizontal=True,
                                           figsize=(10, max(6, 0.4 * diff_long["model"].nunique() + 2))),
                         out_dir, "sc_vs_mc_paired_differences", show))
    summary = _signed_rank_summary(wide)
    if not summary.empty:
        results.append(_effect_vs_p_plot(summary, label_col="label",
                                         title="Paired sc vs mc: Effect Size vs Adjusted p-value",
                                         xlabel="Signed-rank effect size (mc vs sc)",
                                         out_dir=out_dir, name="sc_vs_mc_effect_vs_p", show=show))
    return results


def generate_historical_graphs(show: bool = False) -> list[PlotResult]:
    out_dir = GRAPH_DIR / "primary_output_graphs"
    results: list[PlotResult] = []
    path = STATISTICAL_RESULTS / "proportion_test_results.csv"
    if not path.exists():
        return results
    sc_mc_path = OUTPUT_CSVS / "single_vs_multiple_context_results.csv"
    if sc_mc_path.exists():
        wide_sc_mc = _read_csv(OUTPUT_CSVS / "single_vs_multiple_context_grade_dist.csv")
        sc_mc_dist = wide_sc_mc.melt(var_name="model", value_name="score")
        sc_mc_dist["model"] = sc_mc_dist["model"].map(lambda value: _historical_sc_mc_label(str(value)))
        sc_mc_dist["color"] = sc_mc_dist["model"].map(lambda model: _historical_color("Single-Context vs Multiple-Context", model))
        results.append(_plot_proportion_errorbar(
            _proportion_frame_from_scores(sc_mc_dist, "Single-Context vs Multiple-Context", "mean"),
            title="Single-Context vs Multiple-Context Models - Mean Score / 5 with 95% CI",
            out_dir=out_dir,
            name="single_context_vs_multiple_context_proportion_ci",
            show=show,
            legend_title="Model Family / Context",
            center_stat="mean",
        ))
        results.append(_plot_proportion_errorbar(
            _proportion_frame_from_scores(sc_mc_dist, "Single-Context vs Multiple-Context", "median"),
            title="Single-Context vs Multiple-Context Models - Median Score / 5 with IQR",
            out_dir=out_dir,
            name="single_context_vs_multiple_context_median_proportion_iqr",
            show=show,
            legend_title="Model Family / Context",
            center_stat="median",
        ))
        results.append(_plot_mean_score_ci_lines(
            sc_mc_dist,
            title="Single-Context vs Multiple-Context Models - Mean Scores with 95% CI",
            out_dir=out_dir,
            name="single_context_vs_multiple_context_mean_score_ci",
            show=show,
            legend_title="Model Family / Context",
            center_stat="mean",
        ))
        results.append(_plot_mean_score_ci_lines(
            sc_mc_dist,
            title="Single-Context vs Multiple-Context Models - Median Scores with IQR",
            out_dir=out_dir,
            name="single_context_vs_multiple_context_median_score_iqr",
            show=show,
            legend_title="Model Family / Context",
            center_stat="median",
        ))
        results.append(_plot_half_violins(
            sc_mc_dist,
            group_col="model",
            value_col="score",
            color_col="color",
            title="Single-Context vs Multiple-Context Models - Mean Score Distributions",
            xlabel="Score",
            out_dir=out_dir,
            name="single_context_vs_multiple_context_half_violins_mean",
            legend_title="Model Family / Context",
            show=show,
            center_stat="mean",
        ))
        results.append(_plot_half_violins(
            sc_mc_dist,
            group_col="model",
            value_col="score",
            color_col="color",
            title="Single-Context vs Multiple-Context Models - Median Score Distributions",
            xlabel="Score",
            out_dir=out_dir,
            name="single_context_vs_multiple_context_half_violins",
            legend_title="Model Family / Context",
            show=show,
        ))
    tests = _read_csv(path)
    for task, sub in tests.groupby("task", sort=False):
        models = sorted(set(sub["model1"]).union(set(sub["model2"])))
        pvals = pd.DataFrame(np.nan, index=models, columns=models)
        props: dict[str, float] = {}
        for _, row in sub.iterrows():
            pvals.loc[row["model1"], row["model2"]] = row["p_value"]
            pvals.loc[row["model2"], row["model1"]] = row["p_value"]
            props.setdefault(row["model1"], row["prop1"])
            props.setdefault(row["model2"], row["prop2"])
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.5), max(6, len(models) * 0.5)))
        image = ax.imshow(pvals.to_numpy(dtype=float), cmap="Blues_r")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(models)), models, rotation=90)
        ax.set_yticks(range(len(models)), models)
        ax.set_title(f"P-values Heatmap - {task}")
        results.append(_save(fig, out_dir, f"{_slug(task)}_pvalue_heatmap", show))

        dist_frame = _historical_distribution_frame(task)
        if not dist_frame.empty:
            prop_frame = _proportion_frame_from_scores(dist_frame, task, "mean")
            median_prop_frame = _proportion_frame_from_scores(dist_frame, task, "median")
        else:
            prop_frame = pd.DataFrame({"model": list(props), "proportion": list(props.values())})
            prop_frame["mean_score"] = prop_frame["proportion"].astype(float) * 5
            prop_frame["max_score"] = prop_frame.apply(
                lambda row: _historical_max_score_estimate(task, row["model"]),
                axis=1,
            )
            prop_frame["color"] = prop_frame["model"].map(lambda model: _historical_color(task, model))
            prop_frame["shape"] = prop_frame["model"].map(_historical_marker)
            median_prop_frame = prop_frame.copy()
            median_prop_frame["median_score"] = median_prop_frame["mean_score"]
        results.append(_plot_proportion_errorbar(
            prop_frame,
            title=f"{task} - Mean Score / 5 with 95% CI",
            out_dir=out_dir,
            name=f"{_slug(task)}_proportion_ci",
            show=show,
            legend_title=_historical_legend_title(task),
            center_stat="mean",
        ))
        results.append(_plot_proportion_errorbar(
            median_prop_frame,
            title=f"{task} - Median Score / 5 with IQR",
            out_dir=out_dir,
            name=f"{_slug(task)}_median_proportion_iqr",
            show=show,
            legend_title=_historical_legend_title(task),
            center_stat="median",
        ))
        if not dist_frame.empty:
            results.append(_plot_mean_score_ci_lines(
                dist_frame,
                title=f"{task} - Mean Scores with 95% CI",
                out_dir=out_dir,
                name=f"{_slug(task)}_mean_score_ci",
                show=show,
                legend_title=_historical_legend_title(task),
                center_stat="mean",
            ))
            results.append(_plot_mean_score_ci_lines(
                dist_frame,
                title=f"{task} - Median Scores with IQR",
                out_dir=out_dir,
                name=f"{_slug(task)}_median_score_iqr",
                show=show,
                legend_title=_historical_legend_title(task),
                center_stat="median",
            ))
            results.append(_plot_half_violins(
                dist_frame,
                group_col="model",
                value_col="score",
                color_col="color",
                title=f"{task} - Mean Score Distributions",
                xlabel="Score",
                out_dir=out_dir,
                name=f"{_slug(task)}_half_violins_mean",
                legend_title=_historical_legend_title(task),
                show=show,
                center_stat="mean",
            ))
            results.append(_plot_half_violins(
                dist_frame,
                group_col="model",
                value_col="score",
                color_col="color",
                title=f"{task} - Median Score Distributions",
                xlabel="Score",
                out_dir=out_dir,
                name=f"{_slug(task)}_half_violins",
                legend_title=_historical_legend_title(task),
                show=show,
            ))
    return results


def _plot_proportion_errorbar(
    frame: pd.DataFrame,
    *,
    title: str,
    out_dir: Path,
    name: str,
    show: bool = False,
    legend_title: str = "",
    center_stat: str = "mean",
) -> PlotResult:
    plot_data = frame.copy()
    center_col = f"{center_stat}_score"
    if center_col not in plot_data:
        plot_data[center_col] = plot_data["proportion"].astype(float) * 5
    plot_data = plot_data.sort_values(center_col, ascending=False).reset_index(drop=True)
    y = np.arange(len(plot_data))
    p = plot_data["proportion"].astype(float)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.35)))
    for idx, row in plot_data.iterrows():
        lower = max(0, row.get("proportion_low", row["proportion"]))
        upper = min(1, row.get("proportion_high", row["proportion"]))
        point_color = _mean_ci_color(row["color"])
        ax.errorbar(
            row["proportion"],
            idx,
            xerr=[[row["proportion"] - lower], [upper - row["proportion"]]],
            fmt=row["shape"],
            color=point_color,
            ecolor=point_color,
            elinewidth=1,
            capsize=3,
            markersize=6,
        )

    ax.set_yticks(y, plot_data["model"])
    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xlabel(f"{center_stat.title()} Score / 5")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    _add_historical_legend(ax, plot_data, legend_title)
    return _save(fig, out_dir, name, show)


def _proportion_frame_from_scores(frame: pd.DataFrame, task: str, center_stat: str) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        scores = _numeric(group["score"])
        if scores.empty:
            continue
        color = group["color"].iloc[0] if "color" in group else _historical_color(task, str(model))
        center = _center_value(scores, center_stat)
        low, high = _center_interval(scores, center_stat)
        rows.append({
            "model": str(model),
            f"{center_stat}_score": center,
            "proportion": float(center / 5),
            "proportion_low": float(low / 5),
            "proportion_high": float(high / 5),
            "max_score": float(len(scores) * 5),
            "color": color,
            "shape": _historical_marker(str(model)),
        })
    return pd.DataFrame(rows)


def _plot_mean_score_ci_lines(
    frame: pd.DataFrame,
    *,
    title: str,
    out_dir: Path,
    name: str,
    show: bool = False,
    legend_title: str = "",
    center_stat: str = "mean",
) -> PlotResult:
    rows = []
    for model, group in frame.groupby("model", sort=False):
        scores = _numeric(group["score"])
        if scores.empty:
            continue
        center = _center_value(scores, center_stat)
        ci_low, ci_high = _center_interval(scores, center_stat)
        color = group["color"].iloc[0] if "color" in group else _historical_color("", str(model))
        rows.append({
            "model": str(model),
            f"{center_stat}_score": center,
            "ci_low": max(1, ci_low),
            "ci_high": min(5, ci_high),
            "color": color,
            "line_style": _historical_line_style(str(model)),
        })
    center_col = f"{center_stat}_score"
    plot_data = pd.DataFrame(rows).sort_values(center_col, ascending=False).reset_index(drop=True)
    y = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.35)))
    for idx, row in plot_data.iterrows():
        point_color = _mean_ci_color(row["color"])
        _draw_horizontal_ci(ax, row["ci_low"], row["ci_high"], idx, point_color, linestyle=row["line_style"])
        ax.scatter(
            [row[center_col]],
            [idx],
            color=point_color,
            edgecolor="black",
            linewidth=0.4,
            s=28,
            zorder=3,
        )

    ax.set_yticks(y, plot_data["model"], fontsize=25)
    ax.set_title(title, pad=15, fontweight="bold")
    ax.set_xlabel(f"{center_stat.title()} Score", fontsize=25)
    ax.set_ylabel("Model", fontsize=25)
    ax.tick_params(axis="x", labelsize=25)
    ax.set_xlim(1, 5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    _add_historical_legend(ax, plot_data, legend_title, line_style_key=True)
    return _save(fig, out_dir, name, show)


def _historical_sc_mc_frame(path: Path) -> pd.DataFrame:
    raw = _read_csv(path)
    labels = {
        "o3_sc": "o3-SC",
        "o3_mc": "o3-MC",
        "o3_pro_sc": "o3-Pro-SC",
        "o3_pro_mc": "o3-Pro-MC",
        "o4_mini_sc": "o4-mini-SC",
        "o4_mini_mc": "o4-mini-MC",
        "sonnet_sc": "Sonnet-SC",
        "sonnet_mc": "Sonnet-MC",
        "gemini_2.0_flash_sc": "Gemini-2.0-Flash-SC",
        "gemini_2.0_flash_mc": "Gemini-2.0-Flash-MC",
        "gemini_2.5_pro_sc": "Gemini-2.5-SC",
        "gemini_2.5_pro_mc": "Gemini-2.5-MC",
        "chatgpt4o_sc": "ChatGPT-4o-SC",
        "chatgpt4o_mc": "ChatGPT-4o-MC",
        "gpt4.1_sc": "GPT-4.1-SC",
        "gpt4.1_mc": "GPT-4.1-MC",
        "gpt4.1_images_sc": "GPT-4.1-GPT-Image-SC",
        "gpt4.1_images_mc": "GPT-4.1-GPT-Image-MC",
    }
    rows = []
    for _, row in raw.iterrows():
        model = _historical_sc_mc_label(row["Model"])
        rows.append({
            "model": model,
            "median_score": row["overall_score"] / row["n_total"],
            "proportion": row["overall_score"] / (row["n_total"] * 5),
            "max_score": row["n_total"] * 5,
            "color": _historical_color("Single-Context vs Multiple-Context", model),
            "shape": _historical_marker(model),
        })
    return pd.DataFrame(rows)


def _historical_sc_mc_label(value: str) -> str:
    labels = {
        "o3_sc": "o3-SC",
        "o3_mc": "o3-MC",
        "o3_pro_sc": "o3-Pro-SC",
        "o3_pro_mc": "o3-Pro-MC",
        "o4_mini_sc": "o4-mini-SC",
        "o4_mini_mc": "o4-mini-MC",
        "o4_mini_med_sc": "o4-mini-Medium-SC",
        "o4_mini_med_mc": "o4-mini-Medium-MC",
        "sonnet_sc": "Sonnet-SC",
        "sonnet_mc": "Sonnet-MC",
        "gemini_2.0_flash_sc": "Gemini-2.0-Flash-SC",
        "gemini_2.0_flash_mc": "Gemini-2.0-Flash-MC",
        "gemini_2.5_pro_sc": "Gemini-2.5-SC",
        "gemini_2.5_pro_mc": "Gemini-2.5-MC",
        "chatgpt4o_sc": "ChatGPT-4o-SC",
        "chatgpt4o_mc": "ChatGPT-4o-MC",
        "gpt4.1_sc": "GPT-4.1-SC",
        "gpt4.1_mc": "GPT-4.1-MC",
        "gpt4.1_images_sc": "GPT-4.1-GPT-Image-SC",
        "gpt4.1_images_mc": "GPT-4.1-GPT-Image-MC",
    }
    return labels.get(value, value)


def _historical_distribution_frame(task: str) -> pd.DataFrame:
    if "Temperature" in task:
        return _historical_wide_distribution(
            OUTPUT_CSVS / "gemini_3_pro_temp_grade_dist.csv",
            _temperature_distribution_labels(),
            task=task,
        )
    if "Reasoning" in task:
        return _historical_wide_distribution(
            OUTPUT_CSVS / "openai_reasoning_comparison_grade_dist.csv",
            _reasoning_distribution_labels(),
            task=task,
            include_humans=True,
        )
    return _historical_wide_distribution(
        OUTPUT_CSVS / "llm_collapsed_grade_dist.csv",
        _family_distribution_labels(),
        task=task,
        include_humans=True,
    )


def _historical_wide_distribution(
    path: Path,
    labels: dict[str, str],
    *,
    task: str,
    include_humans: bool = False,
) -> pd.DataFrame:
    wide = _filter_distribution_rows(_read_csv(path), task)
    rows = []
    if include_humans:
        human_scores = _historical_human_scores(task)
        rows.append(pd.DataFrame({
            "model": "Humans",
            "score": human_scores,
            "color": _historical_color(task, "Humans"),
        }))
    for column, label in labels.items():
        if column in wide:
            rows.append(pd.DataFrame({
                "model": label,
                "score": wide[column],
                "color": _historical_color(task, label),
            }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["model", "score", "color"])


def _filter_distribution_rows(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if len(frame) % 60 != 0:
        return frame
    blocks = _task_blocks(task)
    block_numbers = (np.arange(len(frame)) % 60) + 1
    return frame.loc[np.isin(block_numbers, list(blocks))].reset_index(drop=True)


def _historical_human_scores(task: str) -> pd.Series:
    if "Collapsed" in task or "Combined" in task:
        human = _read_csv(OUTPUT_CSVS / "h_grade_distribution.csv")
        return human.iloc[:, 0]
    human = _read_csv(OUTPUT_CSVS / "h_full_results.csv")
    return _filter_distribution_rows_by_block(human, task)["average_mean"]


def _filter_distribution_rows_by_block(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if "block" not in frame:
        return _filter_distribution_rows(frame, task)
    return frame[frame["block"].isin(_task_blocks(task))].reset_index(drop=True)


def _task_blocks(task: str) -> set[int]:
    if "Collapsed" in task or "Combined" in task:
        return ALL_BLOCKS
    if "48 Novel" in task:
        return NOVEL_BLOCKS
    if "Finke" in task and "Collapsed" not in task:
        return FINKE_BLOCKS
    return ALL_BLOCKS


def _family_distribution_labels() -> dict[str, str]:
    return {
        "o3": "o3",
        "o3_pro": "o3-Pro",
        "o3_images": "o3-GPT-Image",
        "gpt5": "GPT-5",
        "opus": "Opus-4.1",
        "sonnet": "Sonnet-4",
        "gemini_2.0_flash": "Gemini-2.0-Flash",
        "gemini_2.0_flash_images": "Gemini-2.0-Flash-Images",
        "gemini_2.5_pro": "Gemini-2.5",
        "gemini_3.0_pro": "Gemini-3 Pro",
        "deepseek_r1": "DeepSeek R1",
        "qwen_3": "Qwen-3",
        "qwen_3_vl": "Qwen-3-VL",
        "gpt_oss_120b": "GPT-OSS-120B",
        "chatgpt-4o": "ChatGPT-4o",
        "gpt4.1": "GPT-4.1",
        "gpt4.1_images": "GPT-4.1-GPT-Image",
        "o4mini_high": "o4-mini",
    }


def _reasoning_distribution_labels() -> dict[str, str]:
    return {
        "o3_pro": "o3-Pro",
        "o3_high": "o3-High",
        "o3_images_high": "o3-GPT-Image-High",
        "o3_med": "o3-Medium",
        "o3_images_med": "o3-GPT-Image-Medium",
        "o3_low": "o3-Low",
        "gpt5_high": "GPT-5-High",
        "gpt5_med": "GPT-5-Medium",
        "gpt5_low": "GPT-5-Low",
        "gpt5_minimal": "GPT-5-Minimal",
        "o4mini_high": "o4-mini-High",
        "o4mini_med": "o4-mini-Medium",
    }


def _temperature_distribution_labels() -> dict[str, str]:
    return {
        "temp_0.1": "Gemini-3 Pro (Temp 0.1)",
        "temp_0.55": "Gemini-3 Pro (Temp 0.55)",
        "temp_1.0": "Gemini-3 Pro (Temp 1.0)",
    }


def _historical_family_members() -> dict[str, list[str]]:
    return {
        "o3": [
            "OpenAI: o3 - Single Context - High Reasoning (2025-07-21)",
            "OpenAI: o3 - Single Context - High Reasoning (2025-07-21)",
            "OpenAI: o3 - Multiple Context - High Reasoning (2025-09-15)",
        ],
        "o3-GPT-Image": [
            "OpenAI: o3 w/ GPT-image-1 - Multiple Context - High Reasoning (2025-07-21)",
            "OpenAI: o3 w/ GPT-image-1 - Multiple Context - High Reasoning (2025-07-22)",
            "OpenAI: o3 w/ GPT-image-1 - Multiple Context - High Reasoning (2025-07-23)",
            "OpenAI: o3 w/ GPT-image-1 - Multiple Context - High Reasoning (2025-07-24)",
        ],
        "o3-Pro": [
            "OpenAI: o3 Pro - Multiple Context - High Reasoning (2025-07-21)",
            "OpenAI: o3 Pro - Multiple Context - High Reasoning (2025-07-21)",
            "OpenAI: o3 Pro - Multiple Context - High Reasoning (2025-09-16)",
        ],
        "GPT-4.1": [
            "OpenAI: GPT 4.1 - Multiple Context (2025-07-21)",
            "OpenAI: GPT 4.1 - Single Context (2025-07-21)",
        ],
        "GPT-4.1-GPT-Image": [
            "OpenAI: GPT 4.1 w/ GPT-image-1 - Multiple Context (2025-07-21)",
            "OpenAI: GPT 4.1 w/ GPT-Image-1 - Single Context (2025-07-21)",
        ],
        "ChatGPT-4o": [
            "OpenAI: ChatGPT-4o - Multiple Context (2025-07-25)",
            "OpenAI: ChatGPT-4o - Single Context (2025-07-25)",
        ],
        "o4-mini": [
            "OpenAI: o4-mini - Multiple Context - High Reasoning (2025-07-21)",
            "OpenAI: o4-mini - Single Context - High Reasoning (2025-07-21)",
        ],
        "Gemini-2.5": [
            "DeepMind: Gemini 2.5 Pro - Multiple Context - Dynamic Thinking (2025-07-21)",
            "DeepMind: Gemini 2.5 Pro - Single Context - Dynamic Thinking (2025-07-21)",
        ],
        "Gemini-2.0-Flash": [
            "DeepMind: Gemini 2.0 Flash - Multiple Context (2025-07-21)",
            "DeepMind: Gemini 2.0 Flash - Single Context (2025-07-21)",
        ],
        "Gemini-2.0-Flash-Images": [
            "DeepMind: Gemini 2.0 Flash w/ Images - Multiple Context (2025-07-25)",
        ],
        "Sonnet-4": [
            "Anthropic: Claude Sonnet 4 - Multiple Context - Extended Thinking 4000t (2025-09-11)",
            "Anthropic: Claude Sonnet 4 - Single Context - Extended Thinking 4000t (2025-09-11)",
        ],
        "Opus-4.1": [
            "Anthropic: Claude Opus 4.1 - Multiple Context - Extended Thinking 9000t (2025-09-11)",
        ],
        "GPT-5": [
            "OpenAI: GPT 5 - Multiple Context - High Reasoning (2025-09-11)",
            "OpenAI: GPT 5 - Multiple Context - High Reasoning (2025-09-15)",
        ],
        "Gemini-3 Pro": [
            "DeepMind: Gemini 3 Pro - High Reasoning - 1.0 Temperature - Multiple Context (2025-11-19)",
            "DeepMind: Gemini 3 Pro - High Reasoning - 0.55 Temperature - Multiple Context (2025-11-19)",
            "DeepMind: Gemini 3 Pro - High Reasoning - 0.1 Temperature - Multiple Context (2025-11-19)",
        ],
        "DeepSeek R1": ["DeepSeek: R1 0528 - Multiple Context (2025-11-20)"],
        "GPT-OSS-120B": ["OpenAI: gpt-oss-120b - High Reasoning - Multiple Context (2025-11-20)"],
        "Qwen-3": ["Alibaba: Qwen 3 235b a22b Thinking 2507 - Multiple Context (2025-11-20)"],
        "Qwen-3-VL": ["Alibaba: Qwen 3 VL 235b a22b Thinking - Multiple Context (2025-11-20)"],
    }


def _historical_reasoning_members() -> dict[str, list[str]]:
    return {
        "o3-High": _historical_family_members()["o3"],
        "o3-Medium": ["OpenAI: o3 - Multiple Context - Medium Reasoning (2025-09-12)"],
        "o3-Low": ["OpenAI: o3 - Multiple Context - Low Reasoning (2025-09-12)"],
        "GPT-5-High": _historical_family_members()["GPT-5"],
        "o3-Pro": _historical_family_members()["o3-Pro"],
        "GPT-5-Medium": ["OpenAI: GPT 5 - Multiple Context - Medium Reasoning (2025-09-16)"],
        "GPT-5-Low": ["OpenAI: GPT 5 - Multiple Context - Low Reasoning (2025-09-15)"],
        "GPT-5-Minimal": ["OpenAI: GPT 5 - Multiple Context - Minimal Reasoning (2025-09-16)"],
        "o4-mini-High": _historical_family_members()["o4-mini"],
        "o4-mini-Medium": [
            "OpenAI: o4-mini - Multiple Context - Medium Reasoning (2025-07-14)",
            "OpenAI: o4-mini - Single Context - Medium Reasoning (2025-07-14)",
        ],
        "o3-GPT-Image-High": _historical_family_members()["o3-GPT-Image"],
        "o3-GPT-Image-Medium": ["OpenAI: o3 w/ GPT-image-1 - Multiple Context - Med Reasoning (2025-07-14)"],
    }


def _historical_color(task: str, model: str) -> str:
    model_key = model.lower()
    if "Temperature" in task:
        return "#8da0cb"
    if "Reasoning" in task:
        if model == "Humans":
            return "#66c2a5"
        if "minimal" in model_key:
            return "#d7b5d8"
        if "low" in model_key:
            return "#df65b0"
        if "medium" in model_key or model_key.endswith("_med") or "_med_" in model_key or "-med" in model_key:
            return "#dd1c77"
        return "#980043"
    if model == "Humans":
        return "#66c2a5"
    if any(name in model_key for name in ["deepseek", "gpt-oss", "gpt_oss", "qwen"]):
        return "#a6d854"
    if "gemini" in model_key or "temp_" in model_key:
        return "#8da0cb"
    if "sonnet" in model_key or "opus" in model_key:
        return "#e78ac3"
    return "#fc8d62"


def _historical_marker(model: str) -> str:
    model_key = model.lower()
    if model.endswith("-SC") or model_key.endswith("_sc"):
        return "o"
    if model.endswith("-MC") or model_key.endswith("_mc"):
        return "D"
    return "o"


def _historical_line_style(model: str) -> str:
    model_key = model.lower()
    if model.endswith("-MC") or model_key.endswith("_mc"):
        return "--"
    return "-"


def _historical_legend_title(task: str) -> str:
    if "Reasoning" in task:
        return "Reasoning Level"
    if "Temperature" in task:
        return "Model Family"
    if "Single-Context" in task:
        return "Model Family / Context"
    return "Model Family"


def _add_historical_legend(
    ax: plt.Axes,
    plot_data: pd.DataFrame,
    title: str,
    line_style_key: bool = False,
    *,
    fontsize: int = 12,
    title_fontsize: int = 12,
    show_title: bool = True,
) -> None:
    from matplotlib.lines import Line2D

    color_labels = _historical_color_labels(title)
    handles = [
        Line2D([0], [0], marker="o", color="none", label=label,
               markerfacecolor=color, markeredgecolor=color, markersize=7)
        for color, label in color_labels
        if color in set(plot_data["color"])
    ]
    if line_style_key and "line_style" in plot_data and any(style == "--" for style in plot_data["line_style"]):
        handles.extend([
            Line2D([0], [0], color="black", label="Single Context",
                   linestyle="-", linewidth=2),
            Line2D([0], [0], color="black", label="Multiple Context",
                   linestyle="--", linewidth=2),
        ])
    elif "shape" in plot_data and any(marker == "D" for marker in plot_data["shape"]):
        handles.extend([
            Line2D([0], [0], marker="o", color="black", label="Single Context",
                   linestyle="none", markersize=6),
            Line2D([0], [0], marker="D", color="black", label="Multiple Context",
                   linestyle="none", markersize=6),
        ])
    if handles:
        legend_kwargs = {
            "handles": handles,
            "loc": "lower right",
            "fontsize": fontsize,
        }
        if show_title:
            legend_kwargs["title"] = title
            legend_kwargs["title_fontsize"] = title_fontsize
        ax.legend(**legend_kwargs)


def _historical_color_labels(title: str) -> list[tuple[str, str]]:
    if title == "Reasoning Level":
        return [
            ("#980043", "High"),
            ("#dd1c77", "Medium"),
            ("#df65b0", "Low"),
            ("#d7b5d8", "Minimal"),
            ("#66c2a5", "Human Baseline"),
        ]
    return [
        ("#fc8d62", "OpenAI"),
        ("#8da0cb", "Gemini"),
        ("#e78ac3", "Claude"),
        ("#a6d854", "Open Models"),
        ("#66c2a5", "Human Baseline"),
    ]


def _historical_max_score_estimate(task: str, model: str) -> float:
    """Estimate the R plot denominator for CI width when only proportions remain.

    The archived R report builds these data frames in-memory and only persists
    pairwise test outputs. This preserves the same visual formula and uses the
    task/model structure to choose the denominator scale.
    """
    if model == "Humans":
        if "Collapsed" in task or "Combined" in task:
            return _human_max_score("all")
        if "48 Novel" in task:
            return _human_max_score("novel")
        return _human_max_score("finke")

    if "Collapsed" in task or "Combined" in task:
        unit = 300
    elif "48 Novel" in task:
        unit = 240
    else:
        unit = 60
    return unit * _historical_run_count(task, model)


def _historical_run_count(task: str, model: str) -> int:
    if "Temperature" in task:
        return 1
    if "Reasoning" in task:
        return {
            "o3-High": 3,
            "o3-Medium": 1,
            "o3-Low": 1,
            "GPT-5-High": 2,
            "o3-Pro": 3,
            "GPT-5-Medium": 1,
            "GPT-5-Low": 1,
            "GPT-5-Minimal": 1,
            "o4-mini-High": 2,
            "o4-mini-Medium": 2,
            "o3-GPT-Image-High": 4,
            "o3-GPT-Image-Medium": 1,
        }.get(model, 1)
    return {
        "o3": 3,
        "o3-GPT-Image": 4,
        "o3-Pro": 3,
        "GPT-4.1": 2,
        "GPT-4.1-GPT-Image": 2,
        "ChatGPT-4o": 2,
        "o4-mini": 2,
        "Gemini-2.5": 2,
        "Gemini-2.0-Flash": 2,
        "Gemini-2.0-Flash-Images": 1,
        "Sonnet-4": 2,
        "Opus-4.1": 1,
        "GPT-5": 2,
        "Gemini-3 Pro": 3,
        "DeepSeek R1": 1,
        "GPT-OSS-120B": 1,
        "Qwen-3": 1,
        "Qwen-3-VL": 1,
    }.get(model, 1)


def _human_max_score(which: str) -> float:
    if which == "finke":
        return (_read_csv(OUTPUT_CSVS / "h_graded_results_finke.csv")["n_total"] * 5).sum()
    if which == "novel":
        return (_read_csv(OUTPUT_CSVS / "h_graded_results_novel.csv")["n_total"] * 5).sum()
    return _human_max_score("finke") + _human_max_score("novel")


def _slug(value: object) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def generate_all(show: bool = False) -> list[PlotResult]:
    results = []
    results.extend(generate_core_pipeline_graphs(show=show))
    results.extend(generate_human_vs_model_graphs(show=show))
    results.extend(generate_reasoning_comparison_graphs(show=show))
    results.extend(generate_image_comparison_graphs(show=show))
    results.extend(generate_qwen_comparison_graphs(show=show))
    results.extend(generate_temperature_graphs(show=show))
    results.extend(generate_sc_vs_mc_graphs(show=show))
    results.extend(generate_historical_graphs(show=show))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis graphs as PNG files.")
    parser.add_argument(
        "--target",
        choices=[
            "all",
            "core",
            "human-vs-model",
            "reasoning",
            "images",
            "qwen",
            "temperature",
            "sc-vs-mc",
            "historical",
        ],
        default="all",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively while saving them.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generators = {
        "all": generate_all,
        "core": generate_core_pipeline_graphs,
        "human-vs-model": generate_human_vs_model_graphs,
        "reasoning": generate_reasoning_comparison_graphs,
        "images": generate_image_comparison_graphs,
        "qwen": generate_qwen_comparison_graphs,
        "temperature": generate_temperature_graphs,
        "sc-vs-mc": generate_sc_vs_mc_graphs,
        "historical": generate_historical_graphs,
    }
    results = generators[args.target](show=args.show)
    for result in results:
        print(result.path.relative_to(ROOT))


if __name__ == "__main__":
    main()
