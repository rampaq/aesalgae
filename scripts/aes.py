""" Run with Sage. A scripting frontend for the aesalgae package. Use this to run single experiments. """
import argparse
import datetime
import json
import os
import pickle
import pprint
import signal
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import aesalgae
from sage.interfaces.magma import Magma

__version__ = "1.0"
DEFAULT_LOGPATH = "logs"


def parse_cmdline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="aesalgae", description="Run experiments")
    parser.set_defaults(sparse=None, batched=False, disable_gpu=False)

    parser.add_argument(
        "--logdir",
        type=str,
        default=DEFAULT_LOGPATH,
        help="Distination for logs. Create the directory first.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        help="Overloads logdir if specified.",
    )

    parser.add_argument(
        "--batched",
        action="store_true",
        default=False,
        help="Used for scripting. Disable collecting of unused statistics and telemetry and autosave logs automatically.",
    )

    parser.add_argument("-n", type=int, default=1, choices=range(1, 11))
    parser.add_argument("-r", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("-c", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("-e", type=int, default=4, choices=[4, 8])

    parser.add_argument("--pc-pairs", type=int, default=1)

    sparse_group = parser.add_mutually_exclusive_group()
    pre_reduce_group = parser.add_mutually_exclusive_group()

    sparse_group.add_argument(
        "--sparse",
        action="store_true",
        help="When neither --sparse or --dense option is not supplied, algorithm will decide automatically. "
        "Sparse version is good when the dense system would run out of memory. "
        "Sparsity refers to Macaulay matrices during F4 computation",
    )
    sparse_group.add_argument(
        "--dense",
        dest="sparse",
        action="store_false",
        help="Supply to enable dense version of Magma F4 + GPU. See --sparse",
    )
    parser.add_argument(
        "--no-gpu",
        dest="disable_gpu",
        action="store_true",
        help="Supply to forcibly disable GPU in case dense Magma F4 is used. See --sparse and --dense.",
    )
    parser.add_argument(
        "-T",
        "--threads",
        type=int,
        default=1,
        dest="num_threads",
        help="Number of threads for Magma. Default 1. Magma does not recommend combining more threads and GPU (in dense mode)",
    )

    # parser.add_argument(
    #    "--timeout",
    #    type=str,
    #    help="Timeout for experiment. After timeout expires, scratch the experiment and save logs. Specify in format: "
    #    "5m -- 5 minutes, 4h -- 4 hours, 356s -- 356 seconds",
    # )
    # parallel_group.add_argument(
    #    "--auto-parallel",
    #    dest="auto_parallel",
    #    default=False,
    #    action="store_true",
    #    help="For Magma, only dense systems are parallelizable. GPU is preferred. When Magma ",
    # )

    parser.add_argument(
        "--gb-method",
        type=str,
        dest="gb_method",
        default="magma",
        help="Method of computing Groebner basis. Possible values: 'magma' (default), 'fgb'.",
    )
    parser.add_argument(
        "--magma-exec",
        default="magma",
        type=str,
        help="Executable of Magma. Defaults to 'magma'.",
    )

    parser.add_argument(
        "--preprocessing",
        type=str,
        help="Basic types of preprocessing are 'inflate', 'lll' and 'monomelim'. Chain them by concatenating with comma ','",
    )
    pre_reduce_group.add_argument(
        "--pre-reduce",
        type=int,
        help="Only valid when --preprocessing is given. Number of polynomials which preprocessing should output."
        " When not given, it is set to number of PC pairs * key bits.",
    )
    pre_reduce_group.add_argument(
        "--pre-reduce-mult",
        type=int,
        help="Same as --pre-reduce  but given in multiples of key bits.",
    )

    args = parser.parse_args()
    return args


class ExperimentRun:
    def __init__(self, args):
        self.args = args
        self.kill_lock = threading.Lock()

        if args.logfile is None:
            self.path = args.logdir
            self.path_is_dir = True
            if not os.path.isdir(args.logdir):
                print(
                    f"The log directory '{args.logdir}' does not exist. Create it first."
                )
                exit(1)
        else:
            self.path = Path(args.logfile)
            self.path_is_dir = False
            if not os.path.isdir(self.path.parent):
                print(
                    f"The parent directory '{self.path.parent}' of log file '{self.path}' does not exist. Create it first."
                )
                exit(1)
            print(f"Logging to {args.logfile}")

        self.batched = args.batched
        n, r, c, e = args.n, args.r, args.c, args.e
        key_bits = r * c * e

        # preprocessing dimensions
        if args.pre_reduce is None:
            args.pre_reduce = args.pc_pairs * key_bits
        if args.pre_reduce_mult is not None:
            args.pre_reduce = args.pre_reduce_mult * key_bits

        self.magma_interface = None
        if args.gb_method == "magma":
            self.magma_interface = Magma(command=args.magma_exec)

        self.aes = aesalgae.AlgebraicAES(
            args.n,
            args.r,
            args.c,
            args.e,
            pc_pairs=args.pc_pairs,
            reduced_dim=args.pre_reduce,
            gb_method=args.gb_method,
            preprocessing=args.preprocessing,
            sparse=args.sparse,
            enable_gpu=not args.disable_gpu,
            num_threads=args.num_threads,
            magma_interface=self.magma_interface,
        )

    def run(self):
        crash = None
        key_contained = None

        try:
            key_contained = self.aes.log.timefn(self.aes.run, name="run")()
        except Exception as e:
            print("Crashed.")
            print(traceback.format_exc())
            crash = e

        with self.kill_lock:
            self.finalize(key_contained, crash)

    def finalize(
        self,
        key_contained=None,
        crash=None,
    ):
        log = self.aes.log
        if self.batched:
            self.save_log(key_contained, crash)
        else:
            inp = input("Logs? [Y(show)/(n)o/(s)ave]> ")
            if inp.lower() in ["n"]:
                exit()
            elif inp.lower() in ["s"]:
                self.save_log(key_contained, crash)
            else:
                logs = log.get_logs()
                pp = pprint.PrettyPrinter(indent=4, compact=True, width=120)
                pp = pp.pprint
                pp(logs)
                inp = input("Save? [y/N]> ")

                if inp.lower() in ["y", "yes"]:
                    self.save_log(key_contained, crash)

    def save_log(self, key_contained: bool | None, crash: Exception | None):
        if self.path_is_dir:
            fpath = self.path / self.filename_builder(args, crash)
        else:
            fpath = self.path
        log = {
            "args": dict(args.__dict__.items()),
            "log": self.aes.log.get_logs(),
            "_crash": str(crash) if crash else False,
            "_key-contained": key_contained,
        }

        with open(str(fpath) + ".pickle", "wb") as f:
            pickle.dump(log, f)

        with open(fpath, "w", encoding="utf8") as f:
            json.dump(log, f, indent=2, sort_keys=True)

    def filename_builder(self, crash):
        args = self.args
        n, r, c, e = args.n, args.r, args.c, args.e
        crash = "CRASH" if crash else ""
        timestamp = datetime.datetime.now().timestamp()
        preprocessing = (
            f"_PRE-{args.preprocessing}_PREDIM{args.pre_reduce}_"
            if args.preprocessing is not None
            else ""
        )
        gb_method = args.magma_exec if args.gb_method == "magma" else "fgb"
        return f"{crash}{n}{r}{c}{e}_PC{args.pc_pairs}{preprocessing}_{gb_method}_v{__version__}_{timestamp}.json"

    def signal_handler(
        self,
        sig: int,
        frame,
    ):
        if sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1]:
            with self.kill_lock:
                self.finalize(
                    key_contained=None,
                    crash="timeout" if sig == signal.SIGUSR1 else "interrupt",
                )
                print("Finalized.")

                if self.magma_interface is not None:
                    print("Killing Magma...")
                    self.magma_interface.interrupt(tries=2, timeout=1)
                    self.magma_interface.quit()
                else:
                    print("Magma not defined!")

                # kill all
                print("Killing self")
                sid = os.getsid(os.getpid())
                subprocess.run(
                    f"/bin/kill -SIGTERM -- -{sid}", shell=True
                )  # send SIGINT to all children in session
                time.sleep(2)
                subprocess.run(
                    f"/bin/kill -SIGINT -- -{sid}", shell=True
                )  # send SIGINT to all children in session

            exit()


if __name__ == "__main__":
    args = parse_cmdline()
    exprun = ExperimentRun(args)

    if args.batched:
        # register interrupts when running automatically
        signal.signal(signal.SIGTERM, exprun.signal_handler)
        signal.signal(signal.SIGINT, exprun.signal_handler)
        signal.signal(signal.SIGUSR1, exprun.signal_handler)
    exprun.run()


# def parse_timeout(timeout: str) -> float:
#    """return timeout in seconds"""
#    num = timeout[:-1]
#    unit = timeout[-1]
#    if not unit in ["s", "m", "h"]:
#        raise ValueError("Time unit in timeout invalid. <h,s,m>")
#    try:
#        num = float(num)
#    except ValueError:
#        raise ValueError("Timeout format invalid")
#
#    if unit == "s":
#        return num
#    elif unit == "m":
#        return 60 * num
#    elif unit == "h":
#        return 3600 * num
