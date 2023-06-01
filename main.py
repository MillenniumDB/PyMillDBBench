# Could be useful:
# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FakeDataset.html#torch_geometric.datasets.FakeDataset
import os

from torch_geometric.datasets import CoraFull

from benchmark_interface import MillenniumDBBenchmarkInterface


# Clear both buffer/cache and swap of the OS
def clear_os():
    os.system("sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
    os.system("sudo swapoff -a && sudo swapon -a")


if __name__ == "__main__":
    dataset = CoraFull(root="./tmp/CoraFull")
    data = dataset[0]
    
    MDB = MillenniumDBBenchmarkInterface(
        data_path="./tmp/MillenniumDB",
        create_db_path="/home/zeus/MDB/MillenniumDB-Dev/build/Release/bin/create_db",
    )
    MDB.create_database(name="CoraFull", graph=data)
