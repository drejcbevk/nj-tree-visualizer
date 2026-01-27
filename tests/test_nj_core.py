import unittest
import numpy as np

from skbio import DistanceMatrix, TreeNode as SkbioTree
from skbio.tree import nj as skbio_nj
from skbio.tree import path_dists

from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from Bio import Phylo
from io import StringIO

from src.nj_core import neighbor_joining_core, to_newick

# ----------------------------------------------------------------------
# Test matrix generator
# ----------------------------------------------------------------------

def generate_test_matrices():

    labels = ["A", "B", "C", "D"]
    D = np.array([
        [0, 5, 9, 9],
        [5, 0, 10, 10],
        [9, 10, 0, 8],
        [9, 10, 8, 0],
    ], dtype=float)
    yield D, labels

    labels = ["A", "B", "C", "D", "E", "F"]
    D = np.array([
        [0, 2, 4, 4, 7, 7],
        [2, 0, 4, 4, 7, 7],
        [4, 4, 0, 2, 7, 7],
        [4, 4, 2, 0, 7, 7],
        [7, 7, 7, 7, 0, 4],
        [7, 7, 7, 7, 4, 0],
    ], dtype=float)
    yield D, labels

    labels = [f"T{i}" for i in range(8)]
    D = np.array([
        [0, 2, 4, 6, 6, 8, 8, 8],
        [2, 0, 4, 6, 6, 8, 8, 8],
        [4, 4, 0, 6, 6, 8, 8, 8],
        [6, 6, 6, 0, 2, 8, 8, 8],
        [6, 6, 6, 2, 0, 8, 8, 8],
        [8, 8, 8, 8, 8, 0, 4, 4],
        [8, 8, 8, 8, 8, 4, 0, 2],
        [8, 8, 8, 8, 8, 4, 2, 0],
    ], dtype=float)
    yield D, labels

    labels = [f"L{i}" for i in range(10)]
    D = np.array([
        [0,2,4,4,6,6,8,8,8,8],
        [2,0,4,4,6,6,8,8,8,8],
        [4,4,0,2,6,6,8,8,8,8],
        [4,4,2,0,6,6,8,8,8,8],
        [6,6,6,6,0,2,8,8,8,8],
        [6,6,6,6,2,0,8,8,8,8],
        [8,8,8,8,8,8,0,2,4,4],
        [8,8,8,8,8,8,2,0,4,4],
        [8,8,8,8,8,8,4,4,0,2],
        [8,8,8,8,8,8,4,4,2,0],
    ], dtype=float)
    yield D, labels

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def my_tree_to_skbio(D, labels):
    """
    Run our NJ and convert to scikit-bio TreeNode via Newick.
    """
    root = neighbor_joining_core(D, labels)
    newick = to_newick(root)
    return SkbioTree.read([newick])

def assert_trees_equivalent(tree1, tree2, places=6):
    """
    Two trees are equivalent iff their patristic distance matrices match.
    """
    dist = path_dists(
        trees=[tree1, tree2],
        use_length=True,
        metric="euclidean"
    )
    assert abs(dist[0, 1]) < 10 ** (-places)

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

class TestNeighborJoining(unittest.TestCase):

    def test_against_scikit_bio(self):
        """
        Compare against scikit-bio NJ
        """
        for D, labels in generate_test_matrices():
            with self.subTest(labels=labels):
                dm = DistanceMatrix(D, labels)
                ref_tree = skbio_nj(dm)
                my_tree = my_tree_to_skbio(D, labels)
                assert_trees_equivalent(ref_tree, my_tree)

    def test_against_biopython(self):
        """
        Compare against Biopython NJ
        """
        constructor = DistanceTreeConstructor()
        for D, labels in generate_test_matrices():
            with self.subTest(labels=labels):
                lower = [list(D[i, :i + 1]) for i in range(len(D))]
                dm = _DistanceMatrix(labels, lower)
                ref_tree = constructor.nj(dm)
                handle = StringIO()
                Phylo.write(ref_tree, handle, "newick")
                ref_skbio = SkbioTree.read([handle.getvalue()])
                my_tree = my_tree_to_skbio(D, labels)
                assert_trees_equivalent(ref_skbio, my_tree)

    def test_leaf_count_preserved(self):
        """
        NJ must preserve all input taxa.
        """
        for D, labels in generate_test_matrices():
            root = neighbor_joining_core(D, labels)
            leaves = []
            def walk(n):
                if n.is_leaf():
                    leaves.append(n.name)
                for c in n.children:
                    walk(c)
            walk(root)
            self.assertCountEqual(leaves, labels)

    def test_invalid_input_raises(self):
        D = np.array([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            neighbor_joining_core(D, ["A"])

    def test_two_taxa(self):
        D = np.array([[0, 4], [4, 0]], float)
        labels = ["A", "B"]
        root = neighbor_joining_core(D, labels)
        self.assertEqual(len(root.children), 2)

    def test_all_equal_distances(self):
        labels = ["A", "B", "C", "D"]
        D = np.ones((4, 4)) - np.eye(4)
        root = neighbor_joining_core(D, labels)
        self.assertEqual(len([n for n in root.children]), 2)

    def test_deterministic(self):
        D, labels = next(generate_test_matrices())
        t1 = to_newick(neighbor_joining_core(D, labels))
        t2 = to_newick(neighbor_joining_core(D, labels))
        self.assertEqual(t1, t2)


if __name__ == "__main__":
    unittest.main()