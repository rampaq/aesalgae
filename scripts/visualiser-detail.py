#!/usr/bin/env python3
""" Old visualiser, can be used to neatly display experimental errors """
import argparse
import json
import operator
import os
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

AESTuple = t.Tuple[int, int, int, int]

# fmt: off
time_map = {
    "all": [("AES","run")],
    "gb": [("AES.compute_groebner_and_check_key.magma_gb_solve", "gb")],
    "preprocess": [("AES","preprocess")],
    "generate_aes": [("AES","generate_aes")],
}
data_map = {
   "dim-variety": [("AES.compute_groebner_and_check_key","dim-variety")],
#   "hilbert-series": [("AES.compute_groebner_and_check_key.get_num_solutions_from_gb","hilbert-series")],
   "highest-gb-compute-deg": [("AES.compute_groebner_and_check_key.magma_gb_solve","highest-deg")],
   "memory-gb": [("AES.compute_groebner_and_check_key.magma_gb_solve","memory_MB")],
   "len-monomials": [("AES.gather_info_system","len-monomials")],
   "len-monomials-pp": [("AES.gather_info_psystem-post-processing","len-monomials")],
   "semireg": [("AES.gather_info_system","semireg-degree")],
   "semireg-pp": [("AES.gather_info_psystem-post-processing","semireg-degree")],
   "rank": [("AES.gather_info_system","matrix-rank")],
   "rank-pp": [("AES.gather_info_psystem-post-processing","matrix-rank")],
   "eliminated-monoms":[
        ("AES.preprocess.MonomialElimination.compute_combinations","eliminated-monoms"),
        ("AES.preprocess.MonomialElimination.compute_combinations","reductions")
    ]
}
# fmt: on


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


def filter_noncrash(lst: t.List[t.Tuple[SingleExperimentInfo, float]]) -> list[float]:
    """Return entries for which experiment did not crash"""
    return [x[1] for x in lst if not x[0].crash and x[0].key_contained]


def filter_none(lst: t.List[t.Tuple[SingleExperimentInfo, float]]) -> list[float]:
    return [x[1] for x in lst if x[1] is not None and x[0].key_contained]


def get_crashes(
    lst: t.List[t.Tuple[SingleExperimentInfo, float]],
) -> list[SingleExperimentInfo]:
    """Return entries for which experiment did not crash"""
    return [x[0] for x in lst if x[0].crash]


def get_crash_reasons(lst: t.List[t.Tuple[SingleExperimentInfo, float]], n=4):
    return list(set(map(lambda info: str(info.crash_reason), get_crashes(lst))))[:n]


def apply_filter(metric: str):
    if metric in ["all"]:
        return filter_noncrash
    else:
        return filter_none


def process_experiment_runs(
    data_dict: t.Mapping[str, list[t.Tuple[SingleExperimentInfo, float]]], metric: str
) -> tuple[np.floating | None, np.floating | None, str]:
    data = apply_filter(metric)(data_dict[metric])
    std = mean = text = None
    if data:
        count = len(data)
        std = np.std(data)
        mean = np.average(data)
        data = mean
        text = (
            rf"${latex_float(mean,3)}\pm{latex_float(std,2)}$" + "\n" + rf"$({count})$"
        )
    else:
        data = None
        crashes = "\n".join(get_crash_reasons(data_dict[metric]))
        text = crashes

    return (mean, std, text)


tAggregate = t.Mapping[
    AESTuple,
    t.Mapping[
        t.Tuple[int, int | None],  # [PC, preprocess_dim]
        t.Mapping[str, t.List[t.Tuple[SingleExperimentInfo, float]]],
    ],
]


class Visualiser:
    timings: tAggregate
    data: tAggregate

    def __init__(self, batch_directories: list[Path]):

        self.timings, self.data = self.get_data(batch_directories)

    def plot(self, metric: str, is_preprocessing: bool, aes_tup: tuple | None = None):
        origin, metric = metric.split("/")
        data = self.data if origin == "data" else self.timings
        if is_preprocessing:
            self.plot_preprocessing_detailed(
                data, aes_tup_target=aes_tup, metric=metric
            )
        else:
            self.plot_ref(data, aes_tup_target=aes_tup, metric=metric)

    def plot_ref(
        self,
        timings: t.Mapping,
        aes_tup_target: tuple | None = None,
        metric: str = "all",
    ):
        for aes_tup, params in timings.items():
            if aes_tup_target is not None and aes_tup != aes_tup_target:
                continue
            n, r, c, e = aes_tup
            pcs = sorted(list(set(argtup[0] for argtup in params)))
            fig, ax = plt.subplots()

            y = [
                np.average(
                    apply_filter(metric)(
                        params[(pc, None)][metric],
                    )
                )
                for pc in pcs
            ]

            ax.semilogx(pcs, y, base=2)

            ax.set_title(f"SR({n},{r},{c},{e}) - {metric}")
            # ax.set_xlabel("log2(PC)")
            # ax.set_ylabel("log2(reduce_dim)")
            plt.show()

    def plot_preprocessing(
        self,
        data_points: tAggregate,
        aes_tup_target: tuple | None,
        metrics: list[str] | None = None,
    ):
        """Plot all metrics listed into one barchart plot with subplots indicating different PCs"""
        if metrics is None:
            metrics = []

        colors = plt.cm.gist_ncar
        for aes_tup, params in data_points.items():
            if aes_tup_target is not None and aes_tup != aes_tup_target:
                continue
            n, r, c, e = aes_tup
            pcs_ref = sorted(set(argtup[0] for argtup in params if argtup[1] is None))
            pcs_preproc = sorted(
                set(argtup[0] for argtup in params if argtup[1] is not None)
            )
            pcs = sorted(set(pcs_ref).union(set(pcs_preproc)))
            rdims = sorted(set(argtup[1] for argtup in params if argtup[1] is not None))
            use_ref = len(pcs_ref) > 0

            if len(pcs_preproc) == 0:
                continue

            if not use_ref:
                fig, axs = plt.subplots(
                    nrows=len(rdims), sharex=True, layout="restricted"
                )
                ax_ref = None
            else:
                fig, (ax_ref, *axs) = plt.subplots(
                    nrows=len(rdims) + 1,
                    sharex=True,
                    # figsize=(6, 2),
                    gridspec_kw=dict(
                        height_ratios=[2, *[1 for _ in rdims]], wspace=0.1
                    ),
                    layout="restricted",
                )

            width_bar = 0.25  # the width of the bars
            for i, rd in enumerate(rdims):
                ax = axs[i]
                for j, pc in enumerate(pcs):
                    try:
                        data = params[pc, rd]
                    except KeyError:
                        continue
                    means, stds = np.zeros_like(pcs), np.zeros_like(pcs)
                    valid_metrics = sorted(
                        set(metrics).intersection(set(data)),
                        key=lambda metric: metrics.index(metric),
                    )  # sort by original list
                    x_group = np.arange(len(valid_metrics))  # the label locations
                    for im, metric in enumerate(valid_metrics):
                        # im = metrics.index(metric)
                        means[im], stds[i], text = process_experiment_runs(data, metric)
                        offset_bar = width_bar * im
                        rects = ax.bar(
                            x_group + offset_bar, means, width_bar, label=f"PC {pc}"
                        )
                        ax.bar_label(rects, padding=len(valid_metrics))
                        ax.errorbar(
                            x_group + offset_bar, y_ref, yerr=stds, fmt="-o"
                        )  # format="ko",

    def plot_preprocessing_detailed(
        self,
        data_points: tAggregate,
        aes_tup_target: tuple | None = None,
        metric: str = "all",
    ):
        for aes_tup, params in data_points.items():
            if aes_tup_target is not None and aes_tup != aes_tup_target:
                continue
            n, r, c, e = aes_tup
            pcs_ref = sorted(set(argtup[0] for argtup in params if argtup[1] is None))
            pcs_preproc = sorted(
                set(argtup[0] for argtup in params if argtup[1] is not None)
            )
            pcs = sorted(set(pcs_ref).union(set(pcs_preproc)))
            rdims = sorted(set(argtup[1] for argtup in params if argtup[1] is not None))
            use_ref = len(pcs_ref) > 0

            if len(pcs_preproc) == 0:
                continue

            PCS, RDS = np.meshgrid(pcs, rdims)
            XX, YY = np.log2(PCS), np.log2(RDS)
            if use_ref:
                PCS_ref, YY_ref = np.meshgrid(pcs, np.array([0, 1]))
                XX_ref = np.log2(PCS_ref)
            # XX, YY = PCS, RDS

            if not use_ref:
                fig, ax = plt.subplots()
                ax_ref = None
            else:
                fig, (ax_ref, ax) = plt.subplots(
                    nrows=2,
                    sharex=True,
                    # figsize=(6, 2),
                    gridspec_kw=dict(height_ratios=[2, len(rdims)], wspace=0.1),
                )

            def generate_regions(ax, PCS, XX, RDS=None, YY=None) -> np.ndarray:
                pp = np.empty_like(PCS, dtype=float)
                for i in range(PCS.shape[1]):  # pcs
                    for j in range(RDS.shape[0] if RDS is not None else 1):  # rds
                        try:
                            xi = PCS[j, i]
                            yi = RDS[j, i] if RDS is not None else None
                            mean, std, text = process_experiment_runs(
                                params[xi, yi], metric
                            )
                            ax.text(
                                XX[j, i],
                                YY[j, i] if YY is not None else 0,
                                text,
                                horizontalalignment="center",
                                verticalalignment="center",
                            )
                            pp[j, i] = mean
                        except KeyError:
                            pp[j, i] = None
                return pp

            pp = generate_regions(ax, PCS, XX, RDS, YY)
            cbar = ax.pcolor(
                XX,
                YY,
                pp,
                cmap=matplotlib.cm.Wistia,
                linewidth=0,
                antialiased=False,
                # norm=matplotlib.colors.LogNorm(vmin=np.nanmin(pp), vmax=np.nanmin(pp)),
            )
            box = ax.get_position()
            pad, width = 0.02, 0.02
            cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
            fig.colorbar(cbar, ax=ax, label="time [s]", cax=cax)

            if use_ref:
                ax_ref: matplotlib.pyplot.Axes

                assert ax_ref is not None
                x_ref = np.log2(pcs)
                y_ref = np.zeros_like(pcs, dtype=float)
                stds = np.zeros_like(pcs, dtype=float)
                texts = []
                for i, pc in enumerate(pcs):
                    mean, std, text = process_experiment_runs(params[pc, None], metric)
                    y_ref[i] = mean
                    texts.append(text)
                    stds[i] = std
                for i, pc in enumerate(pcs):

                    ax_ref.text(
                        x_ref[i],
                        np.nanmax(y_ref + stds, initial=0) * 1.1,
                        texts[i],
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

                ax_ref.errorbar(x_ref, y_ref, yerr=stds, fmt="-o")  # format="ko",
                # ax_ref.errorbar(x_ref, stds, yerr=yerr, label='both limits (default)')
                ax_ref.grid(which="both")
                # ax_ref.grid(which="minor", axis="y")

                # cbar_ref = ax_ref.pcolor(
                #    XX_ref,
                #    YY_ref,
                #    pp_ref,
                #    cmap=matplotlib.cm.Wistia,
                #    linewidth=0,
                #    antialiased=False,
                #    # norm=matplotlib.colors.LogNorm(vmin=np.nanmin(pp), vmax=np.nanmin(pp)),
                # )
                # fig.colorbar(
                #    cbar_ref,
                #    ax=ax_ref,
                #    label="time [s]",
                # )

            ax.set_title(f"SR({n},{r},{c},{e}) - {metric}")
            ax.set_xlabel("log2(PC)")
            ax.set_ylabel("log2(reduce_dim)")
            plt.show()

    def get_data(
        self,
        log_dirs,
    ):
        timings = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for log_dir in log_dirs:
            for file in log_dir.iterdir():
                if not file.is_file() or file.suffix != ".json":
                    continue
                try:
                    log = json.load(open(file, "r", encoding="utf8"))
                except json.JSONDecodeError:
                    continue

                try:
                    log_time = log["log"]["time"]
                    log_data = log["log"]["data"]
                    args = log["args"]
                    n, r, c, e, pc, preprocessing, pre_reduce_mult = (
                        args["n"],
                        args["r"],
                        args["c"],
                        args["e"],
                        args["pc_pairs"],
                        args.get("preprocessing"),
                        args.get("pre_reduce_mult"),
                    )

                    crash = log["_crash"]
                    key_contained = log["_key-contained"]
                    experiment_info = SingleExperimentInfo(
                        crash=crash,
                        key_contained=key_contained,
                        preprocessing=preprocessing,
                    )

                    aes_tup = (n, r, c, e)
                    param_tup = (pc, pre_reduce_mult)
                    time_single = {
                        key: get(log_time, *tuples) for key, tuples in time_map.items()
                    }
                    data_single = {
                        key: get(log_time, *tuples) for key, tuples in time_map.items()
                    }
                except KeyError as e:
                    continue

                timings = aggregate_data(
                    timings, aes_tup, param_tup, time_single, experiment_info
                )
                data = aggregate_data(
                    data, aes_tup, param_tup, data_single, experiment_info
                )

        return timings, data


def aggregate_data(
    data: tAggregate,
    aes_tup: AESTuple,
    param_tup: t.Tuple[int, int],
    data_metric: dict,
    experiment_info: SingleExperimentInfo,
):
    for metric, val in data_metric.items():
        data[aes_tup][param_tup][metric].append((experiment_info, val))
    return data


def get(dic: dict, key: tuple, *args):
    """Obtain key tuple from dict. If not present, try to use *args tuples. If none of them are present, return None."""
    k0, k1 = key
    d1 = dic.get(k0)
    if d1 is None:
        res = None
    else:
        res = d1.get(k1)

    if res is None and args:
        return get(dic, *args)
    return res


def latex_float(f, n=3):
    float_str = ("{0:." + str(n) + "g}").format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "aes-batch", description="run experiments in batch"
    )
    argparser.add_argument(
        "batch_directories",
        help="directory with batch logs",
        nargs="+",
        type=Path,
    )
    argparser.add_argument(
        "-a",
        dest="aes_tup",
        help="aes tuple; eg 3224. If not specified, output all",
        type=str,
    )
    argparser.add_argument(
        "-m",
        dest="metric",
        default="all",
        help="metric to display",
        # choices=["time/all", "time/gb", "time/preprocess"],
        type=str,
    )
    argparser.add_argument(
        "--pre",
        dest="is_preprocessing",
        action="store_true",
        help="When specified, plot as preprocessing instead of ref",
    )
    args = argparser.parse_args()

    if not all(dir.is_dir() for dir in args.batch_directories):
        print(
            f"Folder(s) {list(filter(lambda dir: not dir.is_dir(), args.batch_directories))} do not exist"
        )
        exit(1)

    def lift_10(x):
        return x if x % 10 else 10

    aes_tup = (
        tuple(map(lambda x: lift_10(int(x)), args.aes_tup))
        if args.aes_tup is not None
        else None
    )

    visualiser = Visualiser(batch_directories=args.batch_directories)
    visualiser.plot(
        metric=args.metric, aes_tup=aes_tup, is_preprocessing=args.is_preprocessing
    )
