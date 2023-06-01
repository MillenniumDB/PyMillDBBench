import os
import subprocess
from abc import ABC, abstractmethod

from torch_geometric.data import Data


class BenchmarkInterface(ABC):
    @abstractmethod
    def create_database(self, name: str, graph: Data):
        pass

    # @abstractmethod
    # def delete_database(self, name: str):
    #    pass

    # @abstractmethod
    # def start_server(self):
    #    pass

    # @abstractmethod
    # def stop_server(self):
    #     pass


class MillenniumDBBenchmarkInterface(BenchmarkInterface):
    def __init__(self, data_path: str, create_db_path: str):
        os.makedirs(data_path, exist_ok=True)
        if not os.path.exists(create_db_path):
            raise FileNotFoundError(f"create_db binary not found: {create_db_path}")

        # Data storage path
        self.data_path = data_path
        # Executable paths
        self.create_db_path = create_db_path

    def create_database(self, name: str, graph: Data) -> str:
        # TODO: Prevent creation if already exists?
        dump_path = os.path.join(self.data_path, f"{name}.milldb")
        if os.path.isfile(dump_path):
            # Delete the dump if it already exists
            os.remove(dump_path)
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

        # TODO: Create the database
        # TODO: Store tensors
        #       - start_server
        #       - connect client
        #       - create TensorStore
        #       - store tensors
        #       - stop client
        #       - stop_server
        # TODO: Return the database path
        """
        db_path = os.path.join(self.data_path, name)
        if os.path.isdir(db_path):
            # Delete the database if it already exists
            os.rmtree(db_path)
        os.makedirs(db_path)
        subprocess.run([
            self.create_db_path,
            "--db-path", db_path,

        ]
        """
