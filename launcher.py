#!/usr/bin/env python3
"""
To run K Means example in multiprocess mode:

$ python3 launcher.py --multiprocess
"""

import argparse
import logging
import os
import logging
from examples.multiprocess_launcher import MultiProcessLauncher


parser = argparse.ArgumentParser(description="CrypTen K Means algorithm")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--epochs", default=1, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--clusters", default=5, type=int, metavar="N", help="Number of cluster to segregate"
)

parser.add_argument(
    "--path", default='Mall_Customers.csv', type=str, metavar="N", help="Input File path"
)
parser.add_argument(
    "--skip_plaintext",
    default=False,
    action="store_true",
    help="skip evaluation for plaintext svm",
)
parser.add_argument(
    "--multiprocess",
    default=True,
    action="store_true",
    help="Run example in multiprocess mode",
)


def _run_experiment(args):
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    )
    from KmeansEncrypted import run_mpc_kmeans

    run_mpc_kmeans(
        args.epochs, args.path, args.clusters, args.skip_plaintext, os.environ["RANK"]
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
