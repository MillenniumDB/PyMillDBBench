import os
import shutil
import socket
import subprocess
import time
from abc import ABC, abstractmethod

from pymilldb import MDBClient, TensorStore
from torch_geometric.data import Data


class BenchmarkInterface(ABC):
    @abstractmethod
    def create_database(self, name: str, graph: Data) -> str:
        pass

    @abstractmethod
    def delete_database(self, name: str):
        pass

    @abstractmethod
    def start_server(self) -> None:
        pass

    @abstractmethod
    def stop_server(self) -> None:
        pass


class MillenniumDBBenchmarkInterface(BenchmarkInterface):
    def __init__(
        self, data_path: str, create_db_path: str, server_pymilldb_path: str, port: int
    ):
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
        # Port
        self.port = port
        # Server process
        self.server_process = None
        # TensorStore name to use
        self.store_name = "feat"

    def create_database(self, name: str, graph: Data) -> str:
        db_path = os.path.join(self.data_path, name)
        dump_path = os.path.join(self.data_path, f"{name}.milldb")

        if os.path.isdir(db_path):
            raise FileExistsError(f"Database directory already exists: {db_path}")

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
        self.start_server(db_path)
        with MDBClient("localhost", self.port) as client:
            TensorStore.create(client, self.store_name, graph.num_node_features)
            with TensorStore(client, self.store_name) as store:
                # TODO: store tensors
                print("TODO: store tensors")
                pass
        self.stop_server()

        return db_path

    def delete_database(self, name: str) -> None:
        db_path = os.path.join(self.data_path, name)
        shutil.rmtree(db_path, ignore_errors=True)

    def start_server(self, db_path: str) -> None:
        if self.server_process is not None:
            raise RuntimeError("Server process has not been stopped.")

        self.server_process = subprocess.Popen(
            [self.server_pymilldb_path, db_path, "-p", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait until server listen to port
        while socket.socket().connect_ex(("localhost", self.port)) != 0:
            time.sleep(0.5)

    def stop_server(self):
        self.server_process.kill()
        self.server_process.wait()
