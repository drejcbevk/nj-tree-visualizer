import numpy as np

from Orange.misc import DistMatrix
from .nj_core import TreeNode, neighbor_joining_core


# ----------------------------------------------------------------------
# Wrapper for Orange DistMatrix
# ----------------------------------------------------------------------

def neighbor_joining_orange(dm: DistMatrix, labels=None) -> TreeNode:
    """
    Run Neighbor-Joining on an Orange.misc.DistMatrix.

    Parameters
    ----------
    dm : DistMatrix
        Orange distance matrix (e.g. from Orange.distance.euclidean(table)).
    labels : sequence of str, optional
        Taxon labels. If None, labels are taken from dm.row_items
        if available, otherwise generated as T0, T1, ...

    Returns
    -------
    root : TreeNode
        Root of the resulting NJ tree.
    """
    D = np.asarray(dm, dtype=float)

    if labels is not None:
        labels = [str(l) for l in labels]
    elif getattr(dm, "row_items", None) is not None:
        labels = [str(item) for item in dm.row_items]
    else:
        labels = [f"T{i}" for i in range(D.shape[0])]

    return neighbor_joining_core(D, labels)