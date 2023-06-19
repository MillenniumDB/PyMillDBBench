import pickle
from memory_profiler import profile
import argparse
from torch_geometric.loader import NeighborLoader
from benchmark_drivers import clear_os
from time import perf_counter_ns
import numpy as np


NUM_SAMPLES = 50
NUM_SEEDS = 64
NUM_NEIGHBORS = [5, 5]


@profile
def sample(pkl_path):
    clear_os()

    print(f"Loading graph {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Sampling {NUM_SAMPLES} times from {pkl_path}...")
    sample_times = list()
    loader = NeighborLoader(
        data=data,
        batch_size=NUM_SEEDS,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=None,
        shuffle=True
    )
    for _ in range(NUM_SAMPLES):
        t0 = perf_counter_ns()
        g = next(iter(loader))
        sample_times.append((perf_counter_ns() - t0) / 1e9)

    print(
        "TIME (in seconds)\n"
        f"  AVG: {np.mean(sample_times)}\n"
        f"  STD: {np.std(sample_times)}\n"
        f"  TOT: {np.sum(sample_times)}\n"
        f"  MIN: {np.min(sample_times)}\n"
        f"  Q25: {np.quantile(sample_times, .25)}\n"
        f"  Q50: {np.quantile(sample_times, .5)}\n"
        f"  Q75: {np.quantile(sample_times, .75)}\n"
        f"  MAX: {np.max(sample_times)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Usage: mprof run benchmark_sample_memory.py <pkl_path>"
    )
    parser.add_argument("pkl_path", type=str, help="The path of the .pkl graph")
    args = parser.parse_args()
    sample(args.pkl_path)
