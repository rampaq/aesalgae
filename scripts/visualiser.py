#!/usr/bin/env python3
"""Used to visualise json log files into tables and plots. See time_map and data_map for available metrics."""
import argparse
import itertools
import json
import math
import operator
import os
import signal
import sys
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sage.all import latex
from scipy import stats

ALL_TOGETHER = True

AESTuple = t.Tuple[int, int, int, int]


def ntn(x):
    if x is None:
        return float("nan")
    else:
        return x


# fmt: off
time_map = {
    "all": [("AES","run")],
    "gb": [("AES.compute_groebner_and_check_key.magma_gb_solve", "gb")],
    "preprocess": [("AES","preprocess")],
    "generate_aes": [("AES","generate_aes")],
    "generate_aes": [("AES","generate_aes")],
}
data_map = {
    "dim-variety": [("AES.compute_groebner_and_check_key","dim-variety"), lambda data: 1],
   "dim-variety-log2": [("AES.compute_groebner_and_check_key","dim-variety-log2"), lambda data: np.log2(ntn(get(data, (("AES.compute_groebner_and_check_key","dim-variety")))) or 1)],
   "hilbert-series": [("AES.compute_groebner_and_check_key.get_num_solutions_from_gb","hilbert-series")],
   "highest-gb-compute-deg": [("AES.compute_groebner_and_check_key.magma_gb_solve","highest-deg")],
   "memory-mb": [("AES.compute_groebner_and_check_key.magma_gb_solve","memory_MB")],
   "len-monomials": [("AES.gather_info_system","len-monomials")],
   "len-monomials-pp": [("AES.gather_info_psystem-post-processing","len-monomials")],
   "avg-monomials": [
       lambda data: np.max([1,ntn(get(data, ("AES.gather_info_system", "len-monomials"))) * ntn(get(data, ("AES.gather_info_system", "matrix-density")))])
       ],
   "avg-monomials-pp": [
       lambda data: np.max([1,ntn(get(data, ("AES.gather_info_psystem-post-processing", "len-monomials"))) * ntn(get(data, ("AES.gather_info_psystem-post-processing", "matrix-density")))])
       ],
   "max-deg": [lambda data: int(np.max(ntn(get(data, ("AES.gather_info_system", "monomial-degree-set")))))],
   "max-deg-pp": [lambda data: np.max(ntn(get(data, ("AES.gather_info_psystem-post-processing", "monomial-degree-set"))))],
   "semireg": [("AES.gather_info_system","semireg-degree")],
   "semireg-pp": [("AES.gather_info_psystem-post-processing","semireg-degree")],
   "effective-pc": [("AES.preprocess.LLL","effective-pc"), ("AES.preprocess.MonomialElimination","effective-pc")],
   "density": [("AES.gather_info_system","matrix-density")],
   "density-pp": [("AES.gather_info_psystem-post-processing","matrix-density")],
   "rank": [("AES.gather_info_system","matrix-rank")],
   "rank-pp": [("AES.gather_info_psystem-post-processing","matrix-rank")],
   "weight-lll": [("AES.preprocess.LLL","metric")],
   "dgv-lll": [("AES.preprocess.LLL","dgv")],
   "eliminated-monoms":[
        ("AES.preprocess.MonomialElimination.compute_combinations","eliminated-monoms"),
        ("AES.preprocess.MonomialElimination.compute_combinations","reductions")
    ],
    "method-lll": [("AES.preprocess.LLL","method")],
}
# fmt: on


def mean_without_outlier(x):
    iqr = x.quantile(0.75) - x.quantile(0.25)
    y = x[x.between(x.quantile(0.25) - 1.5 * iqr, x.quantile(0.75) + 1.5 * iqr)]
    return y.mean()


def std_without_outlier(x):
    # https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe
    iqr = x.quantile(0.75) - x.quantile(0.25)
    y = x[x.between(x.quantile(0.25) - 1.5 * iqr, x.quantile(0.75) + 1.5 * iqr)]
    return y.std(ddof=0)


class SingleExperimentInfo:
    crash: bool
    crash_reason: str | None
    key_contained: bool | None
    preprocessing: str | None

    def __init__(
        self, crash: bool | str, key_contained: bool | None, preprocessing: str | None
    ):
        self.crash = False if crash is False else True
        self.crash_reason = None
        if self.crash:
            if crash == "":
                self.crash_reason = "?"
            elif isinstance(crash, str):
                self.crash_reason = crash

        self.key_contained = key_contained

        # if preprocessing is not None:
        #    assert preprocessing in ("monomelim", "lll")
        self.preprocessing = preprocessing


class Visualiser:
    df: pd.DataFrame

    def __init__(self, batch_directories: list[Path]):

        self.df = self.get_data(batch_directories)

    def plot(
        self,
        metrics: list[str],
        aes_tup: tuple | None = None,
        output_dir: Path | None = None,
        table: bool = False,
        best: tuple = (False, None),
        only_valid: bool = True,
        target_preproc: str | None = None,
        latex: bool = False,
    ):
        best_only, best_metric = best
        if best_only:
            self.plot_preprocessing_table(
                self.df,
                metrics=metrics,
                output_dir=output_dir,
                best_metric=best_metric,
                only_valid=only_valid,
                target_preproc=target_preproc,
                latex=latex,
            )
            return
        if table:
            self.plot_table(
                self.df,
                aes_tup_target=aes_tup,
                metrics=metrics,
                output_dir=output_dir,
                only_valid=only_valid,
            )
        else:
            self.plot_preprocessing(
                self.df,
                aes_tup_target=aes_tup,
                metrics=metrics,
                output_dir=output_dir,
                only_valid=only_valid,
            )

    def plot_preprocessing_table(
        self,
        df: pd.DataFrame,
        metrics: list[str],
        only_valid: bool = True,
        output_dir: Path | None = None,
        target_preproc: str | None = None,
        best_metric: str = None,
        latex: bool = False,
    ):

        if not sys.stdout.isatty():
            # for terminal, pandas sets it automatically based on available space
            pd.set_option("display.max_rows", 100000)
            pd.set_option("display.max_columns", 15)
            pd.set_option("display.width", 200)

        if only_valid:
            df = df[(~df["crash"]) & (df["key_contained"])]
        if target_preproc:
            if target_preproc == "ref":
                df = df[df["preprocessing"].isnull()]
            elif target_preproc != "OVERVIEW":
                df = df[df["preprocessing"] == target_preproc]

        x = df.loc[:, ["n", "r", "c", "e"]].astype(str).agg("".join, axis=1)
        df["nrce"] = x
        df = df.drop(["n", "r", "c", "e"], axis=1)

        # params = df.groupby(["nrce", "pc", "rdim", "preprocessing"], dropna=False)[
        #    metrics
        # ]
        params = df.groupby(["nrce", "pc", "rdim", "preprocessing"], dropna=False)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        num_cols = [
            x for x in metrics if x in num_cols
        ]  # ordered intersection list(set(num_cols) & set(metrics))

        means = params[num_cols].agg(mean_without_outlier)
        means = means.reset_index()
        # stds = params.agg(std_without_outlier)
        # stds = stds.reset_index()

        if target_preproc == "OVERVIEW":
            means_group = means.groupby(["nrce"], dropna=False)
        else:
            means_group = means.groupby(["preprocessing", "nrce"], dropna=False)
        min_idxs = means_group[best_metric].transform("min") == means[best_metric]
        # if not only_valid:
        #    min_idxs = min_idxs + (
        #        means_group["time/generate_aes"].transform("max")
        #        == means["time/generate_aes"]
        #    )
        #    print(min_idxs)
        sel_means = means[min_idxs]

        if "data/method-lll" in metrics:
            textual_maxes = params.apply(
                lambda x: x["data/method-lll"].value_counts(dropna=False).idxmax()
            )
            sel_means["data/method-lll"] = textual_maxes.reset_index()[0]

        def format_num(x):
            if np.isnan(x):
                return "---"
            if x < 1:
                return "$<1$" if latex else "<1"
            if x < 10:
                return "{:g}".format(float("{:.{p}g}".format(x, p=2)))
            return "{:g}".format(
                float("{:.{p}g}".format(x, p=int(np.ceil(np.log10(x)))))
            )
            # two significant figures

            # return (
            #    # str(
            #    #    (round(x, 1) if x != round(x) else round(x)) if x < 10 else round(x)
            #    # )
            #    # (round(x, 1) if x != round(x) else round(x)) if x < 10 else round(x)
            #    float("{0:.2g}".format(x))  # two significant digits
            #    if x >= 1
            # )

        if target_preproc == "OVERVIEW":
            g = sel_means
            num_cols = g.select_dtypes(include="number").columns.tolist()
            g[num_cols] = g[num_cols].map(format_num)
            #
            if latex:
                print(g.to_latex(index=False))
            else:
                print(g.to_string(index=False))
        else:
            grouped = sel_means.groupby(["preprocessing"], dropna=False)
            for preproc, g in grouped:
                # print(g)
                preproc = preproc[0]
                g = g.drop(["preprocessing"], axis=1)
                g = g.dropna(axis=1, how="all")  # drop all nan columns
                if not isinstance(preproc, str):
                    # for reference, there is no
                    preproc = "ref"
                    # g = g.drop(["rdim", "time/preprocess"], axis=1)
                else:
                    g["rdim"] = g["rdim"].apply(np.int64)
                if target_preproc and preproc != target_preproc:
                    continue
                print("======", preproc, "=======")
                num_cols = g.select_dtypes(include="number").columns.tolist()
                g[num_cols] = g[num_cols].map(format_num)
                #
                if latex:
                    print(g.to_latex(index=False))
                else:
                    print(g.to_string(index=False))

        # for i, (pc_annot, val_annot) in enumerate(dict(means.max(axis=1)).items()):
        # for idx, group in means:
        #    max_metrics = means.idxmax(axis=1).values
        #    maxmin = means.max(axis=1).min()
        #    print(idx)
        #    print(group)

    def plot_table(
        self,
        df: pd.DataFrame,
        aes_tup_target: tuple | None,
        metrics: list[str] | None = None,
        only_valid: bool = True,
        output_dir: Path | None = None,
    ):
        if not sys.stdout.isatty():
            # for terminal, pandas sets it automatically based on available space
            pd.set_option("display.max_rows", 100000)
            pd.set_option("display.max_columns", 15)
            pd.set_option("display.width", 200)

        if only_valid:
            df = df[(~df["crash"]) & (df["key_contained"])]

        aes_types = df.groupby(["n", "r", "c", "e"])
        # metrics = [col for col in df.columns if col.startswith("time/")]
        # metrics_ref = ["time/]
        for aes_tup, params_aes in aes_types:
            # n, r, c, e = aes_tup
            if aes_tup_target is not None and aes_tup_target != aes_tup:
                continue
            print()
            print("__", f'SR({"".join(map(str, aes_tup))})', "__")
            params = params_aes.sort_values(
                by=["preprocessing"], na_position="first"
            ).groupby(["rdim", "pc", "preprocessing"], dropna=False)[metrics]

            means = params.agg(mean_without_outlier)
            stds = params.agg(std_without_outlier)
            # means = params.mean()
            # stds = params.std(ddof=0)
            combined = (
                means.map("{:<8.3g}".format) + "\u00b1" + stds.map("{:<7.2g}".format)
            )
            print(combined)

            # fig, ax = plt.subplots()
            # ax.axis("off")
            # means.plot(table=True)  # ax, combined)
            # plt.show()

    def plot_preprocessing(
        self,
        df: pd.DataFrame,
        aes_tup_target: tuple | None,
        metrics: list[str] | None = None,
        bars=False,
        only_valid: bool = True,
        compare_ref: bool = True,
        highlight_min: bool = True,
        output_dir: Path | None = None,
    ):
        """Plot all metrics listed into one barchart plot with subplots indicating different PCs

        bars: bars are plotted categorically, so curve plots need to be plotted with x-axis as indices
        """
        if metrics is None:
            metrics = []
        unit = "[s]" if any(["time/" in metric for metric in metrics]) else ""

        def plot_metrics(ax, pcs, means, stds):
            for im, metric in enumerate(metrics):
                if bars:
                    pc_indices = list(map(pcs.index, means[metric].index))
                else:
                    pc_indices = means[metric].index
                ax.plot(
                    pc_indices,
                    means[metric].values,
                    marker="x",
                    color=f"C{im}",
                    label=metric,
                )
                ax.errorbar(
                    pc_indices,
                    means[metric].values,
                    fmt=",",
                    yerr=stds[metric].values,
                    alpha=0.5,
                    capsize=2,
                    color=f"C{im}",
                )

        def annotate(ax, pcs, means, stds):
            max_metrics = means.idxmax(axis=1).values
            maxmin = means.max(axis=1).min()
            for i, (pc_annot, val_annot) in enumerate(dict(means.max(axis=1)).items()):
                if math.isnan(val_annot):
                    continue
                if val_annot == maxmin:
                    color = "black"
                    bold = True
                else:
                    color = "black"
                    bold = False
                if bars:
                    x = pcs.index(pc_annot)
                else:
                    x = pc_annot
                ax.annotate(
                    (r"$\bf{0}$" if bold else "${0}$").format(
                        rf"{latex_float(val_annot,2)}"  # {{\pm}}{latex_float(stds[max_metrics[i]][pc_annot], 2)}"
                    ),
                    (x, val_annot),
                    ha="center",
                    va="bottom",
                    zorder=10,
                    color=color,
                    xytext=(0, 4),
                    textcoords="offset points",
                )

        if only_valid:
            df = df[(~df["crash"]) & (df["key_contained"])]
        aes_types = df.groupby(["n", "r", "c", "e"])
        # metrics = [col for col in df.columns if col.startswith("time/")]
        # metrics_ref = ["time/]
        for aes_tup, params_aes in aes_types:
            n, r, c, e = aes_tup
            if aes_tup_target is not None and aes_tup_target != aes_tup:
                continue
            print(aes_tup)
            params_ref = params_aes.loc[params_aes["preprocessing"].isnull()]
            pcs_ref = sorted(list(params_ref["pc"].value_counts().keys()))
            use_ref = compare_ref and params_aes["preprocessing"].isnull().values.any()

            for preprocessing, params in params_aes.sort_values(
                "preprocessing"
            ).groupby("preprocessing", dropna=True):
                print(preprocessing)
                pcs = sorted(
                    set(params["pc"].value_counts().keys()).union(set(pcs_ref))
                )
                rdims = sorted(list(params["rdim"].value_counts().keys()))
                if len(rdims) == 0 or len(pcs) == 0:
                    continue

                if not use_ref:
                    fig, axs = plt.subplots(
                        nrows=len(rdims),
                        sharex=True,
                        gridspec_kw=dict(height_ratios=[1 for _ in rdims], hspace=0),
                    )
                    ax_ref = None
                else:
                    fig, (ax_ref, *axs) = plt.subplots(
                        figsize=(9, 8),
                        nrows=len(rdims) + 1,
                        sharex=True,
                        # figsize=(6, 2),
                        gridspec_kw=dict(
                            height_ratios=[1.5, *[1 for _ in rdims]],
                            hspace=0,  # if bars else 0.1,
                        ),
                    )
                    ref_metrics = (
                        params_ref.sort_values(by="pc")
                        .set_index("pc")[metrics]
                        .groupby("pc")
                    )

                    # filter out outliers, >3sigma
                    means = ref_metrics.agg(mean_without_outlier)
                    stds = ref_metrics.agg(std_without_outlier)

                    # means = ref_metrics.mean()
                    # stds = ref_metrics.std(ddof=0)
                    plot_metrics(ax_ref, pcs_ref, means, stds)
                    annotate(ax_ref, pcs_ref, means, stds)

                    ax_ref.set_yscale("log")  # , nonposy='clip')
                    ax_ref.set_ylabel(f"reference{"\n"+unit if unit else ""}")
                    # ax_ref.set_ylim(0)
                    # ax_ref.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                    ax_ref.yaxis.grid(True, which="both", alpha=0.5)
                    ax_ref.xaxis.grid(True, which="major", alpha=0.5)
                    ax_ref.set_axisbelow(True)
                    # means.plot(yerr=stds, style=["o"]*4, ax=ax_ref) # for some reason cannot change style plot
                    # ax_ref.get_legend().remove()

                if not isinstance(axs, (list, tuple, np.ndarray)):
                    axs = [axs]

                rdim_groupby = (
                    params.loc[~(params["rdim"].isnull())]
                    .sort_values(by="rdim", ascending=True)
                    .astype({"rdim": "int32"})
                    .groupby("rdim", dropna=True)
                )
                for rdim, data in rdim_groupby:
                    i = rdims.index(rdim)
                    ax: matplotlib.Axes = axs[len(rdims) - i - 1]
                    data_metrics = data.sort_values(by="pc", ascending=True).set_index(
                        "pc"
                    )[metrics]
                    # fill in missing pcs to align subplots
                    if bars:
                        pcs_missing = set(pcs).difference(set(data_metrics.index))
                        if len(pcs_missing) > 0:
                            data_metrics = pd.concat(
                                [
                                    data_metrics,
                                    pd.DataFrame(
                                        [{"pc": pc_miss} for pc_miss in pcs_missing]
                                    ).set_index("pc"),
                                ]
                            ).sort_values(by="pc")
                    data_metrics = data_metrics.sort_values(by="pc").groupby("pc")

                    means = data_metrics.agg(mean_without_outlier)
                    stds = data_metrics.agg(std_without_outlier)
                    # means = data_metrics.mean()
                    # stds = data_metrics.std(ddof=0)
                    if bars:
                        means.plot.bar(yerr=stds, ax=ax, sharex=True, ylim=0)
                        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                        ax.set_ylim(0)
                        ax.yaxis.grid(True, which="major", alpha=0.5)
                    else:
                        plot_metrics(ax, pcs, means, stds)
                        ax.set_xscale("log", base=2)  # , nonposy='clip')
                        ax.yaxis.grid(True, which="major", alpha=0.5)
                        # ax.set_ylim(0, top=means.max(None) * 1.1)
                        ax.set_yscale("log")  # , nonposy='clip')
                        ax.set_ylim(means.min(None) / 2, top=means.max(None) * 10)

                    annotate(ax, pcs, means, stds)

                    # ax.set_ylabel("reference\n[s]")
                    ax.set_ylabel(f"red{rdim}{"\n" + unit if unit else ""}")
                    ax.xaxis.grid(True, which="major", alpha=0.5)
                    ax.set_axisbelow(True)

                axs[-1].set_xlabel(f"PC pairs")
                handles, labels = axs[-1].get_legend_handles_labels()
                for ax in axs:
                    leg = ax.get_legend()
                    if leg is not None:
                        leg.remove()

                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncols=4,
                    bbox_to_anchor=[0.5, 0.97],
                    shadow=True,
                )
                fig.suptitle(
                    f"SR({n},{r},{c},{e}), {r*c*e} key bits, preprocessing={preprocessing.upper()}"
                )
                fig.subplots_adjust(hspace=0)
                fig.tight_layout()
                if output_dir:
                    plt.savefig(output_dir / f"{n}{r}{c}{e}_{preprocessing}.png")
                    plt.savefig(output_dir / f"{n}{r}{c}{e}_{preprocessing}.pdf")
                else:
                    plt.show()
                # valid_data.sort_values(by="pc", ascending=True).set_index("pc")[time_cols].mean().plot.bar(title=f"{aes_tup}, rdim={rd}")

    def get_data(
        self,
        log_dirs,
    ):
        dicts = []
        invalid_json = 0
        key_error = 0
        for log_dir in log_dirs:
            for file in log_dir.iterdir():
                if not file.is_file() or file.suffix != ".json":
                    continue
                try:
                    log = json.load(open(file, "r", encoding="utf8"))
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue

                try:
                    log_time = log["log"]["time"]
                    log_data = log["log"]["data"]
                    args = log["args"]
                    main_dict = dict(
                        n=args["n"],
                        r=args["r"],
                        c=args["c"],
                        e=args["e"],
                        pc=args["pc_pairs"],
                        rdim=args.get("pre_reduce_mult"),
                    )
                    preproc = args.get("preprocessing")
                    crash = log["_crash"]
                    key_contained = log["_key-contained"]
                    experiment_info = SingleExperimentInfo(
                        crash=crash,
                        key_contained=key_contained,
                        preprocessing=preproc,
                    )
                    main_dict.update(
                        crash=experiment_info.crash,
                        key_contained=experiment_info.key_contained,
                        crash_reason=experiment_info.crash_reason,
                        preprocessing=experiment_info.preprocessing,
                    )

                    time_single = {
                        "time/" + key: get(log_time, *tuples)
                        for key, tuples in time_map.items()
                    }
                    data_single = {
                        "data/" + key: get(log_data, *tuples)
                        for key, tuples in data_map.items()
                    }
                    if experiment_info.crash:
                        # when occured, gb timing is misleading
                        # if "time/gb" in time_single:
                        time_single["time/all"] = None
                        time_single["time/gb"] = None

                    dicts.append(main_dict | time_single | data_single)
                except KeyError as e:
                    key_error += 1
                    continue
        if invalid_json > 0:
            print(f"{invalid_json} invalid JSON files")
        if key_error > 0:
            print(f"{key_error} key errors")
        print(f"loaded {len(dicts)} files")
        df = pd.DataFrame(dicts)
        return df


def get(dic: dict, *args):
    """Obtain key tuple from dict. If not present, try to use *args tuples. If none of them are present, return None.
    Alternatively, if arg is a function, supply dic and return its value. If none/nan, retry with next value.
    """
    first_arg: t.Callable[[dict], str | float] | tuple[str, str] = args[0]
    if callable(first_arg):
        res = first_arg(dic)
        if isinstance(res, float) and math.isnan(res):
            res = None
    else:
        k0, k1 = first_arg

        d1 = dic.get(k0)
        if d1 is None:
            res = None
        else:
            res = d1.get(k1)

    args = args[1:]
    if res is None and args:
        return get(dic, *args)

    return res


def latex_float(f, n=3):
    format_n = "{0:." + str(n) + "g}"
    format_np1 = "{0:." + str(n + 1) + "g}"
    float_str = format_np1.format(float(format_n.format(f)))
    # float_str = format_n.format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \cdot 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "aes-batch", description="run experiments in batch"
    )
    argparser.set_defaults(do_table=False, only_best=False)
    argparser.add_argument(
        "batch_directories",
        help="directory with batch logs",
        nargs="+",
        type=Path,
    )
    argparser.add_argument(
        "-m",
        dest="metric",
        default="time/",
        help="metric to display. For multiple metrics, delimit with comma ','",
        # choices=["time/all", "time/gb", "time/preprocess"],
        type=str,
    )
    argparser.add_argument(
        "-o",
        dest="output_dir",
        help="Directory to save plots to",
        type=Path,
    )
    argparser.add_argument(
        "--crashes",
        help="allow also crashes/timeouts into the statistics",
        action="store_true",
    )
    best_grouper = argparser.add_argument_group(
        "Group by preprocessing", "Select the best runs"
    )
    best_grouper.add_argument(
        "--best",
        dest="only_best",
        action="store_true",
        help="display table with best results for each cipher",
    )
    best_grouper.add_argument(
        "--best-metric",
        dest="best_metric",
        default="time/all",
        help="metric to determine what is the best and what to subsequently show",
    )
    best_grouper.add_argument(
        "--latex",
        action="store_true",
        help="Display best data as a latex table",
    )
    best_grouper.add_argument(
        "--preprocessing",
        help='relevant for --best. Specify to output only a single preprocessing and not all available. Specify "OVERVIEW" to generate a table where each cipher has the best preprocessing.',
    )
    cipher_grouper = argparser.add_argument_group("Group by ciphers")
    cipher_grouper.add_argument(
        "-a",
        dest="aes_tup",
        help="aes tuple; eg 3224. If not specified, output all",
        type=str,
    )
    cipher_grouper.add_argument(
        "-t",
        dest="do_table",
        action="store_true",
        help="Display data in a table",
    )
    args = argparser.parse_args()

    if not all(dir.is_dir() for dir in args.batch_directories):
        print(
            f"Folder(s) {list(filter(lambda dir: not dir.is_dir(), args.batch_directories))} do not exist"
        )
        exit(1)

    output_dir = None
    if args.output_dir is not None and args.output_dir.is_dir():
        output_dir = args.output_dir

    def lift_10(x):
        return x if x % 10 else 10

    aes_tup = (
        tuple(map(lambda x: lift_10(int(x)), args.aes_tup))
        if args.aes_tup is not None
        else None
    )

    metrics = args.metric.split(",")
    print(metrics)
    if "time/" in metrics:
        i = metrics.index("time/")
        metrics[i] = list(map(lambda key: "time/" + key, time_map.keys()))
    elif "data/" in metrics:
        i = metrics.index("data/")
        metrics[i] = list(map(lambda key: "data/" + key, data_map.keys()))

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                for subitem in item:
                    yield subitem
            else:
                yield item

    metrics = list(flatten(metrics))

    signal.signal(signal.SIGINT, lambda x, y: exit())
    visualiser = Visualiser(batch_directories=args.batch_directories)
    visualiser.plot(
        metrics=metrics,
        aes_tup=aes_tup,
        output_dir=output_dir,
        table=args.do_table,
        best=(args.only_best, args.best_metric),
        target_preproc=args.preprocessing,
        latex=args.latex,
        only_valid=not args.crashes,
    )
