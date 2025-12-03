import numpy as np
import Orange.distance as odist

from src.nj_orange import neighbor_joining_orange
from src.nj_visualization import interactive_tree_view

from Orange.data import Table
from Orange.misc import DistMatrix
from Orange.distance import Euclidean, Manhattan, Hamming


# ----------------------------------------------------------------------
# Unified distance-computation wrapper (from previous demo logic)
# ----------------------------------------------------------------------

def compute_distance_matrix(table, metric="euclidean"):
    if metric == "euclidean":
        return odist.Euclidean(table)
    elif metric == "manhattan":
        return odist.Manhattan(table)
    elif metric == "hamming":
        return odist.Hamming(table)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ----------------------------------------------------------------------
# 1. Iris (Euclidean)
# ----------------------------------------------------------------------

def demo_iris_euclidean():
    print("Demo 1: Iris (Euclidean)")

    table = Table("iris")

    s0 = [i for i in range(150) if table[i].get_class() == "Iris-setosa"][:15]
    s1 = [i for i in range(150) if table[i].get_class() == "Iris-versicolor"][:15]
    s2 = [i for i in range(150) if table[i].get_class() == "Iris-virginica"][:15]

    idx = s0 + s1 + s2
    table = table[idx]

    dm = compute_distance_matrix(table, metric="euclidean")

    labels = [str(inst.get_class()) for inst in table]

    root = neighbor_joining_orange(dm, labels=labels)
    interactive_tree_view(root)


# ----------------------------------------------------------------------
# 2. Zoo (Manhattan)
# ----------------------------------------------------------------------

def demo_zoo_manhattan():
    print("Demo 2: Zoo (Manhattan)")

    table = Table("zoo")
    dm = compute_distance_matrix(table, metric="manhattan")

    name_var = None
    for var in table.domain.metas:
        if var.name.lower() == "name":
            name_var = var
            break

    if name_var is None:
        raise RuntimeError("Zoo dataset: could not find meta variable 'name'.")

    labels = [str(inst[name_var]) for inst in table]

    root = neighbor_joining_orange(dm, labels=labels)
    interactive_tree_view(root)


# ----------------------------------------------------------------------
# 3. Boston Housing (Euclidean)
# ----------------------------------------------------------------------

def demo_housing_euclidean():
    print("Demo 3: Housing (Euclidean)")

    table = Table("housing")
    table = table[:60]

    dm = compute_distance_matrix(table, metric="euclidean")

    labels = [f"h{i}" for i in range(len(table))]

    root = neighbor_joining_orange(dm, labels=labels)
    interactive_tree_view(root)


# ----------------------------------------------------------------------
# Run all demos
# ----------------------------------------------------------------------

def main():
    demo_iris_euclidean()
    demo_zoo_manhattan()
    demo_housing_euclidean()


if __name__ == "__main__":
    main()