# PyMillDBBench

Benchmark resources for graph sampling

## Installation

### MillenniumDB

- Compile the `MillenniumDB-Dev/MillenniumAI` branch.
- Install `pymilldb` python library for the client/sampler.
- Set both `create_db` and `pymilldb_server` executables path in the `MilleniumDBBenchmark` class instance.

### Neo4j

- Install Neo4j.

### ArangoDB

## Create databases

Run the necessary cells from `benchmark_createdbs.ipynb`

## Run benchmarks

For tracking the memory, run the `benchmark_sample_<database_name>.py` scripts. You can see the details for running each one with the `--help` command.