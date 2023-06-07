# Could be useful:
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FakeDataset.html#torch_geometric.datasets.FakeDataset
from pymilldb import MDBClient, Sampler, TensorStore
from torch_geometric.datasets import CitationFull

from benchmark_interface import MillenniumDBBenchmark, clear_os, stop_process

if __name__ == "__main__":
    dataset = CitationFull(root="./tmp/PubMed", name="PubMed")
    data = dataset[0]

    mdb = MillenniumDBBenchmark(
        data_path="./tmp/MillenniumDB",
        create_db_path="/home/zeus/MDB/MillenniumDB-Dev/build/Release/bin/create_db",
        server_pymilldb_path="/home/zeus/MDB/MillenniumDB-Dev/build/Release/bin/server_pymilldb",
    )
    port = 8080
    db_name = "PubMed"

    if not mdb.database_exists(db_name):
        print("Creating database:", db_name)
        # Create database if it does not exist
        mdb.create_database(name=db_name, graph=data, port=port)
        print("Database created")
    else:
        print("Database already exists:", db_name)

    # Start the server
    print("Starting server...")
    clear_os()
    server_process = mdb.start_server(name=db_name, port=port)

    # Interact with the server from the client
    with MDBClient("localhost", port) as client:
        sampler = Sampler(client)
        g = sampler.subgraph(num_seeds=10, num_neighbors=[5, 5])
        print("Sampled subgraph:", g)

    # Stop the server
    print("Stopping server...")
    stop_process(server_process)
