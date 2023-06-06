# Could be useful:
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FakeDataset.html#torch_geometric.datasets.FakeDataset
import os

from torch_geometric.datasets import CitationFull

from benchmark_interface import MillenniumDBBenchmarkInterface


# Clear both buffer/cache and swap of the OS
def clear_os():
    os.system("sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
    os.system("sudo swapoff -a && sudo swapon -a")


if __name__ == "__main__":
    dataset = CitationFull(root="./tmp/PubMed", name="PubMed")
    data = dataset[0]
    
    mdb = MillenniumDBBenchmarkInterface(
        data_path="./tmp/MillenniumDB",
        create_db_path="/home/zeus/MDB/MillenniumDB-Dev/build/Release/bin/create_db",
        server_pymilldb_path="/home/zeus/MDB/MillenniumDB-Dev/build/Release/bin/server_pymilldb",
        port=8080
    )

    # Remove if exists, else do nothing)
    mdb.delete_database("PubMed")
    # Create database
    db_path = mdb.create_database(name="PubMed", graph=data)
    db_path = "./tmp/MillenniumDB/PubMed"

