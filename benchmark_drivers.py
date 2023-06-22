import os
import shutil
import socket
import subprocess
import time
from abc import ABC, abstractmethod
import pickle
from pymilldb import MDBClient, TensorStore
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset


## Clear both buffer/cache and swap of the OS
def clear_os() -> None:
    os.system("sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
    os.system("sudo swapoff -a && sudo swapon -a")


## Stop a process
def stop_process(process: subprocess.Popen) -> None:
    process.kill()
    process.wait()

## Generate a random graph and store it as a pickle. If it was already generated before
def random_graph(data_path: str, avg_num_nodes: int, avg_degree: int, num_node_features: int) -> Data:
    os.makedirs(data_path, exist_ok=True)
    pkl_path = os.path.join(data_path, f"N{avg_num_nodes}.D{avg_degree}.F{num_node_features}.pkl")
    if not os.path.exists(pkl_path):
        # Generate and dump the graph
        g = FakeDataset(
            num_graphs=1,
            avg_num_nodes=avg_num_nodes,
            avg_degree=avg_degree,
            num_channels=num_node_features,
            edge_dim=10,
            num_classes=1,
            is_undirected=True,
            task="node"
        ).generate_data()
        del g.y
        with open(pkl_path, "wb") as f:
            pickle.dump(g, f)
    else:
        # Load an existing graph dump
        with open(pkl_path, "rb") as f:
            g = pickle.load(f)
    return g


class BenchmarkDriver(ABC):
    @abstractmethod
    ## Return whether a database with the given name exists
    def database_exists(self, name: str) -> bool:
        pass

    @abstractmethod
    ## Create a database given a name and a graph
    def create_database(self, name: str, graph: Data) -> None:
        pass

    @abstractmethod
    ## Delete a database given its name
    def delete_database(self, name: str) -> None:
        pass

    @abstractmethod
    ## Start a server given a database name and a port
    def start_server(self, name: str, port: int) -> subprocess.Popen:
        pass


class MillenniumDBDriver(BenchmarkDriver):
    def __init__(self, data_path: str, create_db_path: str, server_pymilldb_path: str):
        os.makedirs(data_path, exist_ok=True)
        if not os.path.exists(create_db_path):
            raise FileNotFoundError(f"create_db binary not found: {create_db_path}")
        if not os.path.exists(server_pymilldb_path):
            raise FileNotFoundError(
                f"server_pymilldb binary not found: {server_pymilldb_path}"
            )

        # Data storage path
        self.data_path = data_path
        # Executable paths
        self.create_db_path = create_db_path
        self.server_pymilldb_path = server_pymilldb_path
        # TensorStore name to use
        self.store_name = "feat"

    def database_exists(self, name: str) -> bool:
        return os.path.isdir(os.path.join(self.data_path, name))

    def create_database(self, name: str, graph: Data, port: int = 8080) -> str:
        if self.database_exists(name):
            raise FileExistsError(f"Database already exists!")

        db_path = os.path.join(self.data_path, name)
        dump_path = os.path.join(self.data_path, f"{name}.milldb")

        # Write the graph dump
        with open(dump_path, "w") as f:
            # Nodes
            for node_idx in range(graph.num_nodes):
                f.write(f"N{node_idx}\n")
            # Edges
            for edge_idx in range(graph.num_edges):
                f.write(
                    f"N{graph.edge_index[0, edge_idx]}->N{graph.edge_index[1, edge_idx]} :T\n"
                )

        # Create the database
        result = subprocess.run(
            [self.create_db_path, dump_path, db_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(f"create_db {result.stderr.decode('utf-8')}")

        # Store tensors
        server_process = self.start_server(name, port)
        with MDBClient("localhost", port) as client:
            TensorStore.create(client, self.store_name, graph.num_node_features)
            with TensorStore(client, self.store_name) as store:
                for node_idx in range(graph.num_nodes):
                    store[f"N{node_idx}"] = graph.x[node_idx]
        stop_process(server_process)

    def delete_database(self, name: str) -> None:
        shutil.rmtree(os.path.join(self.data_path, name), ignore_errors=True)

    def start_server(self, name: str, port: int = 8080) -> None:
        if socket.socket().connect_ex(("localhost", port)) == 0:
            raise RuntimeError(f"Server already running on port {port}")
        db_path = os.path.join(self.data_path, name)
        buffer_size = 8 * 1024 * 256 # 8GB
        server_process = subprocess.Popen(
            [self.server_pymilldb_path, db_path, "-p", str(port), "-b", buffer_size],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait until server listen to port
        while socket.socket().connect_ex(("localhost", port)) != 0:
            time.sleep(0.5)
        return server_process


class Neo4jDriver(BenchmarkDriver):
    def __init__(self, data_path: str):
        raise NotImplementedError("Not implemented!")

    def database_exists(self, name: str) -> bool:
        raise NotImplementedError("Not implemented!")

    def create_database(self, name: str, graph: Data):
        raise NotImplementedError("Not implemented!")

    def delete_database(self, name: str):
        raise NotImplementedError("Not implemented!")

    def start_server(self, name: str, port: int = 8080):
        raise NotImplementedError("Not implemented!")


class ArangoDBDriver(BenchmarkDriver):
    def __init__(self, data_path: str):
        raise NotImplementedError("Not implemented!")

    def database_exists(self, name: str) -> bool:
        raise NotImplementedError("Not implemented!")

    def create_database(self, name: str, graph: Data):
        raise NotImplementedError("Not implemented!")

    def delete_database(self, name: str):
        raise NotImplementedError("Not implemented!")

    def start_server(self, name: str, port: int = 8080):
        raise NotImplementedError("Not implemented!")
