import pickle
import os
from torch_geometric.loader import NeighborLoader
from benchmark_drivers import clear_os
from time import perf_counter_ns
import numpy as np


graph_names = ["N100000.D10.F3"]

NUM_SAMPLES = 10
NUM_SEEDS = 64
NUM_NEIGHBORS = [5, 5]

for graph_name in graph_names:
    pkl_path = os.path.join("./data/pkl", f"{graph_name}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Sampling {NUM_SAMPLES} times from {graph_name}...")
    sample_times = list()

    clear_os()
    for _ in range(NUM_SAMPLES):
        loader = NeighborLoader(
            data=data,
            batch_size=NUM_SEEDS,
            num_neighbors=NUM_NEIGHBORS,
            input_nodes=None,
        )
        t0 = perf_counter_ns()
        g = next(iter(loader))
        sample_times.append((perf_counter_ns() - t0) / 1e9)

    print(
        "STATS (in seconds)\n"
        f"  AVG: {np.mean(sample_times)}\n"
        f"  STD: {np.std(sample_times)}\n"
        f"  TOT: {np.sum(sample_times)}\n"
        f"  MIN: {np.min(sample_times)}\n"
        f"  Q25: {np.quantile(sample_times, .25)}\n"
        f"  Q50: {np.quantile(sample_times, .5)}\n"
        f"  Q75: {np.quantile(sample_times, .75)}\n"
        f"  MAX: {np.max(sample_times)}"
    )
