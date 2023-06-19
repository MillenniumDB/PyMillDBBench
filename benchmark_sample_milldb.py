from benchmark_drivers import MillenniumDBDriver, stop_process, clear_os
from pymilldb import MDBClient, TensorStore, Sampler
from time import perf_counter_ns
from memory_profiler import profile
import argparse
import numpy as np

NUM_SAMPLES = 10
NUM_SEEDS = 64
NUM_NEIGHBORS = [5, 5]

@profile
def sample(db_path):
    clear_os()
    driver = MillenniumDBDriver(
        data_path="./data/MillenniumDB",
        create_db_path="/home/mdbai/MillenniumDB-Dev/build/Release/bin/create_db",
        server_pymilldb_path="/home/mdbai/MillenniumDB-Dev/build/Release/bin/server_pymilldb",
    )

    print(f"Sampling {NUM_SAMPLES} times from {db_path}...")
    sample_times = list()
    server_process = driver.start_server(db_path)
    with MDBClient("localhost", 8080) as client:
        sampler = Sampler(client)
        store = TensorStore(client, "feat")
        for _ in range(NUM_SAMPLES):
            t0 = perf_counter_ns()
            g = sampler.subgraph(NUM_SEEDS, NUM_NEIGHBORS)
            g.x = store[g.node_ids]
            sample_times.append((perf_counter_ns() - t0) / 1e9)
    stop_process(server_process)

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
        description="Usage: mprof run --include-children benchmark_sample_milldb.py <db_path>"
    )
    parser.add_argument("db_path", type=str, help="The path of the MillenniumDB graph")
    args = parser.parse_args()
    sample(args.db_path)