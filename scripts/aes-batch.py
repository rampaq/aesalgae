#!/usr/bin/env python3
""" Used to execute series of experiments from batch JSON files and save the logs to a directory """
import argparse
import atexit
import datetime
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from threading import Timer

from termcolor import cprint

# DEFAULT_TIMEOUT = 30  # 2.5 * 3600


class Batch:
    def __init__(self, batch: dict, path: Path, timeout=0):
        self.timeout = timeout
        self.batch_info = batch
        self.current_popen = None
        self.kill_lock = threading.Lock()
        self.path = path

    def run_batch(self, run_number: int = 0):
        batch = self.batch_info
        batch_id = batch["id"].format(run_num=run_number)
        parallel = batch["parallel"]
        self.timeout = batch["timeout"] if "timeout" in batch else self.timeout
        magma = batch["magma"] if "magma" in batch else "magma"
        fpath_format = self.path / batch_id / batch["fname"]
        os.makedirs(self.path / batch_id, exist_ok=True)

        if self.timeout:
            print(f"Timeout set to {self.timeout} s")

        for params in batch["batch"]:
            n, r, c, e = map(int, params["aes"])
            if n == 0:
                n = 10

            gpu = parallel
            if "gpu" in params:
                gpu = params["gpu"]

            parallel_args = (
                f"{'-T24' if parallel else '-T1'} {'--no-gpu' if not gpu else ''}"
            )
            # ["-T24" if parallel else "-T1"] + ["--no-gpu"] if not gpu else []

            preprocessing = params.get("preprocessing", None)
            pre_reduce_mult = params.get("pre-reduce-mult", [None])
            print()
            for pc in params["pc"]:
                for reduc_dim in pre_reduce_mult:
                    if reduc_dim is not None:
                        if reduc_dim > pc and "inflate" not in preprocessing.split(","):
                            continue
                        if preprocessing == "monomelim" and reduc_dim >= pc:
                            continue

                    info_current_batch = f"SR({n},{r},{c},{e}), PC={pc}" + (
                        f", {preprocessing}->{reduc_dim}"
                        if preprocessing is not None
                        else ""
                    )

                    fpath = str(fpath_format).format(
                        n=int(n) % 10,
                        r=r,
                        c=c,
                        e=e,
                        pc=pc,
                        preprocessing=preprocessing,
                        prereduce=reduc_dim,
                    )
                    if os.path.isfile(fpath):
                        print(info_current_batch, end="")
                        print(" | ", end="")
                        continue
                    else:
                        print()
                        cprint(info_current_batch, "yellow", end="")
                        print(": computing...")
                        print(datetime.datetime.now())

                    preprocessing_args = (
                        f"--preprocessing {preprocessing} --pre-reduce-mult {reduc_dim if reduc_dim else 0}"
                        if preprocessing is not None
                        else ""
                    )
                    with subprocess.Popen(
                        f'sage /home/faikltom/scripts/aes.py -n {n} -r {r} -c {c} -e {e} --pc-pairs {pc} --magma {magma} {parallel_args} {preprocessing_args} --batched --logfile "{fpath}" ',
                        start_new_session=True,  # group all child into session to kill them if needed
                        shell=True,
                        # os.killpg(os.getpgid(pro.pid), signal.SIGTERM)  # Send the signal to all the process groups
                    ) as proc:
                        self.current_popen = proc
                        try:
                            proc.communicate(timeout=self.timeout)
                        except subprocess.TimeoutExpired:
                            self.quit_current_experiment(timeout=True)

                        print("Waiting for timeout/kill lock to finish if present")
                        with self.kill_lock:
                            # wait till killing is complete so the next batch can start
                            pass

    def quit_current_experiment(self, timeout=False):
        """quit current iteration"""
        # make sure to start new session in the child if parent should not be killed
        with self.kill_lock:
            proc = self.current_popen
            if proc is None:
                print("No running process")
                return
            pid = proc.pid
            sid = os.getsid(pid)
            if timeout:
                cprint(
                    f"Killing due to timeout {self.timeout if self.timeout is not None else '?'}s...",
                    "red",
                )
                # proc.send_signal(signal.SIGUSR1)  # signal timeout
                # time.sleep(0.5)
                print(f"/bin/kill -SIGUSR1 -- -{sid}")
                subprocess.run(f"/bin/kill -SIGUSR1 -- -{sid}", shell=True)
            else:
                cprint("Interrupting experiment...", "red")
                # proc.send_signal(signal.SIGINT)
                # time.sleep(0.5)
                print(f"/bin/kill -SIGINT -- -{sid}")
                subprocess.run(f"/bin/kill -SIGINT -- -{sid}", shell=True)
            print("Sleeping 8s before next kill, do not press Ctrl-C! ...")
            try:
                time.sleep(8)
            except:
                print(f"/bin/kill -SIGKILL -- -{sid}")
                subprocess.run(
                    f"/bin/kill -SIGKILL -- -{sid}", shell=True
                )  # send SIGINT to all children in session

            # print(f"/bin/kill -SIGINT -- -{sid}")
            # subprocess.run(
            #    f"/bin/kill -SIGINT -- -{sid}", shell=True
            # )  # send SIGINT to all children in session
            ## time.sleep(0.5)
            ## subprocess.run(
            ##    f"/bin/kill -SIGUSR1 -- -{sid}", shell=True
            ## )  # send SIGINT to all children in session
            # time.sleep(5)
            print(f"/bin/kill -SIGKILL -- -{sid}")
            subprocess.run(
                f"/bin/kill -SIGKILL -- -{sid}", shell=True
            )  # send SIGINT to all children in session
            # time.sleep(0.2)
            # proc.terminate()
            # time.sleep(0.2)
            # proc.kill()
            time.sleep(1)
            print("proc.kill")
            proc.kill()

    def signal_handler(
        self,
        sig: int,
        frame,
    ):
        if sig in [signal.SIGINT, signal.SIGTERM]:
            self.quit_current_experiment()
            # elif sig == signal.SIGTERM:
            exit()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "aes-batch", description="run experiments in batch"
    )
    argparser.add_argument(
        "batch_json",
        help="JSON file with batch specification",
        type=Path,
    )
    argparser.add_argument(
        "--logdir",
        help="Location of directory where to store logs",
        default=Path("/scratch/faikltom/logs"),
        type=Path,
    )
    argparser.add_argument(
        "-r",
        dest="run_num",
        default=0,
        type=int,
        help="Run number for batch.",
    )

    argparser.add_argument(
        "--timeout",
        default=2.5 * 3600,
        type=int,
        help="Max time spent on one iteration",
    )
    args = argparser.parse_args()
    if not args.batch_json.is_file():
        print(f"File {args.batch_json} does not exist")
        exit(1)

    with open(args.batch_json, encoding="utf-8") as file_json:
        batch_info = json.load(file_json)

    batch = Batch(batch_info, path=args.logdir, timeout=args.timeout)

    signal.signal(signal.SIGTERM, batch.signal_handler)
    signal.signal(signal.SIGINT, batch.signal_handler)
    batch.run_batch(run_number=args.run_num)
