import os
import shutil
import socket
import subprocess
import time
from abc import ABC, abstractmethod

from pymilldb import MDBClient, TensorStore
from torch_geometric.data import Data


## Clear both buffer/cache and swap of the OS
def clear_os() -> None:
    os.system("sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
    os.system("sudo swapoff -a && sudo swapon -a")


## Stop a process
def stop_process(process: subprocess.Popen) -> None:
    process.kill()
    process.wait()


class BenchmarkInterface(ABC):
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


class MillenniumDBBenchmark(BenchmarkInterface):
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
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(f"create_db: {result.stderr.decode('utf-8')}")

        # Store tensors
        server_process = self.start_server(name, port)
        with MDBClient("localhost", port) as client:
            TensorStore.create(client, self.store_name, graph.num_node_features)
            with TensorStore(client, self.store_name) as store:
                for node_idx in range(graph.num_nodes):
                    store[f"N{node_idx}"] = graph.x[node_idx]
        stop_process(server_process)

    def delete_database(self, name: str) -> None:
        db_path = os.path.join(self.data_path, name)
        shutil.rmtree(db_path, ignore_errors=True)

    def start_server(self, name: str, port: int = 8080) -> None:
        if socket.socket().connect_ex(("localhost", port)) == 0:
            raise RuntimeError(f"Server already running on port {port}")
        db_path = os.path.join(self.data_path, name)
        server_process = subprocess.Popen(
            [self.server_pymilldb_path, db_path, "-p", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait until server listen to port
        while socket.socket().connect_ex(("localhost", port)) != 0:
            time.sleep(0.5)
        return server_process


class Neo4jBenchmark(BenchmarkInterface):
    def __init__(self, data_path: str, neo4j_admin_path: str):
        os.makedirs(data_path, exist_ok=True)
        if not os.path.exists(neo4j_admin_path):
            raise FileNotFoundError(
                f"neo4j_admin_path binary not found: {neo4j_admin_path}"
            )

        # Data storage path
        self.data_path = data_path
        # Executable paths
        self.neo4j_admin_path = neo4j_admin_path

    def database_exists(self, name:str)->bool:
        # TODO: Implement this
        return False

    def create_database(self, name: str, graph: Data) -> None:
        if self.database_exists(name):
            raise FileExistsError(f"Database already exists!")

        db_path = os.path.join(self.data_path, name)
        nodes_dump_path = os.path.join(self.data_path, f"nodes_{name}.csv")
        edges_dump_path = os.path.join(self.data_path, f"edges_{name}.csv")

        # Write the graph dump
        with open(nodes_dump_path, "w") as f:
            # Nodes
            f.write(":id,feat\n")
            for node_idx in range(graph.num_nodes):
                f.write(f"{node_idx},{graph.x[node_idx].tolist()}\n")
        with open(edges_dump_path, "w") as f:
            # Edges
            f.write(":START_ID,:END_ID,relation_type")
            for edge_idx in range(graph.num_edges):
                f.write(f"{graph.edge_index[0, edge_idx]},{graph.edge_index[1, edge_idx]},T\n")
        # Create the database
        # result = subprocess.run(
        #     [self.neo4j_admin_path, "create", name],
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL,
        # )
        # if result.returncode != 0:
        #     raise RuntimeError(f"neo4j-admin create: {result.stderr.decode('utf-8')}")

    def delete_database(self):
        # TODO: Implement this
        pass

    def start_server(self):
        # TODO: Implement this
        pass

from torch_geometric.datasets import FakeDataset


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

n4j = Neo4jBenchmark(data_path="./tmp/Neo4j", neo4j_admin_path="/usr/bin/neo4j-admin")
n4j.create_database("test", random_graph(10, 2, 5))
