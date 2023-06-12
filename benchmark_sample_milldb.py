from benchmark_drivers import MillenniumDBDriver, stop_process, clear_os
from pymilldb import MDBClient, TensorStore, Sampler
from time import perf_counter_ns
import numpy as np

graph_names = ["N100000.D10.F3"]
driver = MillenniumDBDriver(
    data_path="./data/MillenniumDB",
    create_db_path="/home/mdbai/MillenniumDB-Dev/build/Release/bin/create_db",
    server_pymilldb_path="/home/mdbai/MillenniumDB-Dev/build/Release/bin/server_pymilldb",
)
NUM_SAMPLES = 10
NUM_SEEDS = 64
NUM_NEIGHBORS = [5, 5]

for graph_name in graph_names:
    print(f"Sampling {NUM_SAMPLES} times from {graph_name}...")
    sample_times = list()

    clear_os()
    server_process = driver.start_server(graph_name)
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
