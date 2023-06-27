#!/bin/bash

graph_names=("N100000.D10.F10" "N1000000.D10.F10" "N10000000.D10.F10")

# In memory
for graph_name in ${graph_names[@]}; do
    mprof run benchmark_sample_memory.py "/home/mdbai/PyMillDBBench/data/pkl/$graph_name.pkl" > "./$graph_name.mem.out"
done

# MillenniumDB
for graph_name in ${graph_names[@]}; do
    mprof run --include-children benchmark_sample_milldb.py "/home/mdbai/PyMillDBBench/data/MillenniumDB/$graph_name" > "./$graph_name.milldb.out"
done
