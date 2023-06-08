from time import perf_counter_ns

import numpy as np
from pymilldb import MDBClient, Sampler, TensorStore
from torch_geometric.datasets import FakeDataset

from benchmark_interface import MillenniumDBBenchmark, clear_os


def random_graph(avg_num_nodes, avg_degree, num_node_features):
    g = FakeDataset(
        num_graphs=1,
        avg_num_nodes=avg_num_nodes,
        avg_degree=avg_degree,
        num_channels=num_node_features,
        edge_dim=10,
        num_classes=1,
        is_undirected=True,
    ).generate_data()
    del g.y
    return g


if __name__ == "__main__":
    AVG_NUM_NODES = 100
    AVG_DEGREE = 10
    NUM_NODE_FEATURES = 5
    PORT = 8080
    NUM_SAMPLES = 10
    NAME = f"FakeDataset_N{AVG_NUM_NODES}_D{AVG_DEGREE}F_{NUM_NODE_FEATURES}"
    GRAPH = random_graph(AVG_NUM_NODES, AVG_DEGREE, NUM_NODE_FEATURES)

    # MillenniumDB
    mdbbench = MillenniumDBBenchmark(
            data_path="./tmp/MillenniumDB",
            create_db_path="/home/zeus/MDB/MillenniumDB-Dev/build/Release/bin/create_db",
            server_pymilldb_path="/home/zeus/MDB/MillenniumDB-Dev/build/Release/bin/server_pymilldb",
        )

    if not mdbbench.database_exists(NAME):
        print("Creating database:", NAME)
        mdbbench.create_database(NAME, GRAPH)
        print("Database created")
    else:
        print("Database already exists:", NAME)

    print("Launching server...")
    clear_os()
    server_process = mdbbench.start_server(name=NAME, port=PORT)
    mdb_times = list()
    with MDBClient("localhost", PORT) as client:
        s = Sampler(client)
        print("Starting sampling...")
        for _ in range(NUM_SAMPLES):
            t0 = perf_counter_ns()
            s.subgraph(10, [5, 5])
            mdb_times.append(perf_counter_ns() - t0)
        print("Finished sampling:")
        print(f"AVG: {np.mean(mdb_times)/1e9}s")
        print(f"STD: {np.std(mdb_times)/1e9}s")


