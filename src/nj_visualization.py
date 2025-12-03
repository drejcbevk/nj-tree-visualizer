"""
Interactive Neighbor-Joining tree visualization with a distance-threshold slider.

- Slider cuts at a distance from the root.
- A vertical line at this cut defines clusters: one cluster per subtree to the
  right of each cut edge.
- All nodes/branches in a cluster subtree share the same color.
- Branches go from LEFT to RIGHT into the child node; splits are at nodes.
- At each cut, cluster colors are unique and assigned top-to-bottom.
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List
from matplotlib.widgets import Slider
import matplotlib.cm as cm
from matplotlib import patheffects as pe
from .nj_core import TreeNode


# ----------------------------------------------------------------------
# Helpers: parent map, depths, layout
# ----------------------------------------------------------------------

def build_parent_map(root: TreeNode) -> Dict[TreeNode, TreeNode]:
    """Return a dict child -> parent (root maps to None)."""
    parent: Dict[TreeNode, TreeNode] = {root: None}
    stack = [root]
    while stack:
        node = stack.pop()
        for ch in node.children:
            parent[ch] = node
            stack.append(ch)
    return parent


def compute_depths(root: TreeNode,
                   parent: Dict[TreeNode, TreeNode]) -> Dict[TreeNode, float]:
    """
    Cumulative distance from root to each node (root depth = 0).
    """
    depth: Dict[TreeNode, float] = {root: 0.0}
    stack = [root]
    while stack:
        node = stack.pop()
        d = depth[node]
        for ch in node.children:
            depth[ch] = d + ch.length
            stack.append(ch)
    return depth


def compute_layout(
    root: TreeNode
) -> Tuple[Dict[TreeNode, float], Dict[TreeNode, float], Dict[TreeNode, float], float]:
    """
    Compute plotting layout and depths.

    x_pos[node] = distance from root (horizontal).
    y_pos[node] : leaves in consecutive rows; internal nodes = mean of children.
    """
    parent = build_parent_map(root)
    depth = compute_depths(root, parent)

    max_depth = max(depth.values()) if depth else 0.0
    x_pos = {n: d for n, d in depth.items()}

    y_pos: Dict[TreeNode, float] = {}
    counter = [0.0]

    def assign_y(node: TreeNode):
        if not node.children:
            y_pos[node] = counter[0]
            counter[0] += 1.0
        else:
            for ch in node.children:
                assign_y(ch)
            ys = [y_pos[ch] for ch in node.children]
            y_pos[node] = sum(ys) / len(ys)

    assign_y(root)

    return x_pos, y_pos, depth, max_depth


# ----------------------------------------------------------------------
# Clustering by cut distance from root
# ----------------------------------------------------------------------

def compute_clusters_by_cut(
        root: TreeNode, 
        depth: Dict[TreeNode, float], 
        parent: Dict[TreeNode, TreeNode],
        cut_distance: float) -> Tuple[List[TreeNode], Dict[TreeNode, int]]:
    """
    Clusters induced by a vertical cut at `cut_distance` from the root.

    Rule:
      - For each edge (parent -> child) with parent_depth < cut <= child_depth,
        the child subtree is a cluster.
      - Leaves not in such subtrees form singleton clusters.
      - If no edge is cut, the whole tree is one cluster.
    """
    eps = 1e-9
    cluster_roots_set = set()

    # Edges crossed by the cut
    for node, par in parent.items():
        if par is None:
            continue
        dp = depth[par]
        dn = depth[node]
        if dp < cut_distance <= dn + eps:
            cluster_roots_set.add(node)

    cluster_for_node: Dict[TreeNode, int] = {}

    # No edge cut -> single cluster (whole tree)
    if not cluster_roots_set:
        cluster_roots = [root]
        stack = [root]
        cid = 0
        while stack:
            n = stack.pop()
            cluster_for_node[n] = cid
            for ch in n.children:
                stack.append(ch)
        return cluster_roots, cluster_for_node

    # Subtrees to the right of the cut
    cluster_roots = list(cluster_roots_set)

    for cid, croot in enumerate(cluster_roots):
        stack = [croot]
        while stack:
            n = stack.pop()
            if n in cluster_for_node:
                continue
            cluster_for_node[n] = cid
            for ch in n.children:
                stack.append(ch)

    # Leaves above the cut as singleton clusters
    all_nodes = list(depth.keys())
    leaves = [n for n in all_nodes if not n.children]

    next_cid = len(cluster_roots)
    for leaf in leaves:
        if leaf not in cluster_for_node:
            cluster_for_node[leaf] = next_cid
            cluster_roots.append(leaf)
            next_cid += 1

    return cluster_roots, cluster_for_node


# ----------------------------------------------------------------------
# Interactive plotting
# ----------------------------------------------------------------------

def interactive_tree_view(root: TreeNode, max_distance_from_data: float | None = None, figsize=(14, 8)):
    """
    Interactive tree viewer.

    - X-axis distance is aligned to data if max_distance_from_data is given.
    - Slider is in the same units as the distance matrix.
    """
    parent = build_parent_map(root)
    x_depth, y_pos, depth, max_depth = compute_layout(root)

    all_nodes = list(depth.keys())
    leaves = [n for n in all_nodes if not n.children]

    # Scaling: from tree-depth units -> data distance units
    if max_distance_from_data is not None and max_depth > 0:
        scale = max_distance_from_data / max_depth
        max_plot_distance = max_distance_from_data
    else:
        scale = 1.0
        max_plot_distance = max_depth

    # Plot coordinates
    x_plot = {n: d * scale for n, d in depth.items()}

    # parent -> child edges
    edges = []
    for node in all_nodes:
        for ch in node.children:
            edges.append((node, ch))

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0, right=0.98, bottom=0.10, top=0.98)

    branch_lw = 1
    leaf_size = 3

    edge_artists = []

    for parent_node, child_node in edges:
        xp, yp = x_plot[parent_node], y_pos[parent_node]
        xc, yc = x_plot[child_node], y_pos[child_node]

        vline, = ax.plot([xp, xp], [yp, yc], lw=branch_lw, color='0.6')
        hline, = ax.plot([xp, xc], [yc, yc], lw=branch_lw, color='0.6')

        edge_artists.append({
            'parent': parent_node,
            'child': child_node,
            'hline': hline,
            'vline': vline,
        })

    # Leaf markers
    node_artists: Dict[TreeNode, plt.Line2D] = {}
    for node in leaves:
        xn, yn = x_plot[node], y_pos[node]
        marker, = ax.plot(
            xn, yn, marker='o', linestyle='',
            markersize=leaf_size, color='0.3'
        )
        node_artists[node] = marker

    # Leaf labels
    label_artists: Dict[TreeNode, plt.Text] = {}
    for node in leaves:
        xn, yn = x_plot[node], y_pos[node]
        name = node.name if node.name is not None else ""
        text_offset = max_plot_distance * 0.0075
        txt = ax.text(
            xn + text_offset, yn, name,
            va='center', ha='left',
            fontsize=8,
            clip_on=False,
        )
        label_artists[node] = txt

    
    x_min = max_plot_distance * -0.05
    x_max = max_plot_distance * 1.05

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, max(y_pos.values()) + 0.5)
    ax.set_yticks([])

    # Remove outline box
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Slider and cut line
    slider_max = max_plot_distance if max_plot_distance > 0 else 1.0
    cut_init = slider_max / 2.0 if slider_max > 0 else 0.0

    cut_line = ax.axvline(x=cut_init, color='gray', linestyle='-', linewidth=1.0)

    # Cluster count text
    cluster_text = fig.text(
        0.98, 0.98, "",
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=1)
    )

    slider_ax = fig.add_axes([0.20, 0.03, 0.60, 0.02])
    cut_slider = Slider(
        slider_ax,
        "Cut distance",
        0.0,
        slider_max,
        valinit=cut_init
    )
    
    base_colors = cm.get_cmap("tab20").colors
    n_base = len(base_colors)

    cluster_for_node: Dict[TreeNode, int] = {}
    current_highlight_cluster = None

    # caches for colors
    root_color: Dict[TreeNode, np.ndarray] = {}
    cid_to_root: Dict[int, TreeNode] = {}


    def get_cluster_color(i: int):
        # if more clusters than colors, change lightness slightly
        base = np.array(base_colors[i % n_base])
        k = i // n_base
        factor = 1.0 - 0.15 * min(k, 2)
        rgb = np.clip(base[:3] * factor, 0, 1)
        return rgb

    current_highlight_cluster = None

    def highlight_cluster(cid: int):
        """Outline cluster cid in black while keeping its color."""
        # Path effect: black stroke under normal line/text
        outline_effect = [
            pe.Stroke(linewidth=1.25, foreground="black"),
            pe.Normal(),
        ]

        # Nodes (markers)
        for node, c in cluster_for_node.items():
            if c != cid:
                continue
            if node in node_artists:
                m = node_artists[node]
                m.set_markeredgecolor("black")
                m.set_markeredgewidth(1.2)


        # Labels (text)
        for node, c in cluster_for_node.items():
            if c != cid:
                continue
            if node in label_artists:
                txt = label_artists[node]
                txt.set_color("black")
                txt.set_path_effects([])


        # Edges (lines)
        for item in edge_artists:
            child = item["child"]
            if cluster_for_node.get(child) == cid:
                for line in (item["hline"], item["vline"]):
                    line.set_path_effects(outline_effect)


    def unhighlight_all():
        """Remove hover effects and restore base cluster colors."""

        # Remove edge path effects
        for item in edge_artists:
            for line in (item["hline"], item["vline"]):
                line.set_path_effects([])

        # Reset markers
        for node, m in node_artists.items():
            m.set_markeredgewidth(0.0)
            m.set_markeredgecolor("none")

            cid = cluster_for_node.get(node, None)
            if cid is None or not cid_to_root:
                col = "0.5"
            else:
                root_node = cid_to_root[cid]
                col = root_color[root_node]
            m.set_color(col)

        # Reset labels
        for node, txt in label_artists.items():
            cid = cluster_for_node.get(node, None)
            if cid is None or not cid_to_root:
                col = "0.5"
            else:
                root_node = cid_to_root[cid]
                col = root_color[root_node]
            txt.set_color(col)
            txt.set_path_effects([])


    def update(_):
        nonlocal cluster_for_node, current_highlight_cluster, root_color, cid_to_root

        unhighlight_all()

        cut_distance = cut_slider.val
        cut_line.set_xdata([cut_distance, cut_distance])

        # recompute clusters for this cut
        cluster_roots, cluster_for_node = compute_clusters_by_cut(
            root, depth, parent, cut_distance
        )
        n_clusters = len(cluster_roots)

        cluster_roots_sorted = sorted(cluster_roots, key=lambda n: y_pos[n])
        root_color = {}

        if n_clusters > 0:
            for i, r in enumerate(cluster_roots_sorted):
                root_color[r] = get_cluster_color(i)
        else:
            root_color[cluster_roots_sorted[0]] = get_cluster_color(0)

        cid_to_root = {cid: cluster_roots[cid] for cid in range(len(cluster_roots))}

        # Leaf colors (markers + labels)
        for node, marker in node_artists.items():
            cid = cluster_for_node.get(node, None)
            if cid is None:
                col = "0.5"
            else:
                root_node = cid_to_root[cid]
                col = root_color.get(root_node, "tab:blue")
            marker.set_color(col)
            label_artists[node].set_color(col)

        # Edge colors by child cluster
        for item in edge_artists:
            child = item["child"]
            cid = cluster_for_node.get(child, None)
            if cid is None:
                col = "0.75"
                lw = 1.0
            else:
                root_node = cid_to_root[cid]
                col = root_color.get(root_node, "tab:blue")
                lw = 1.0
            item["hline"].set_color(col)
            item["hline"].set_linewidth(lw)
            item["vline"].set_color(col)
            item["vline"].set_linewidth(lw)

        # reset any active highlight because cluster assignments changed
        current_highlight_cluster = None

        cluster_text.set_text(f"Clusters: {n_clusters}")
        fig.canvas.draw_idle()


    def on_move(event):
        nonlocal current_highlight_cluster

        # Mouse outside main axes -> clear highlight
        if event.inaxes is not ax:
            if current_highlight_cluster is not None:
                current_highlight_cluster = None
                unhighlight_all()
                fig.canvas.draw_idle()
            return

        # Detect if hovering over any label
        hovered_node = None
        renderer = fig.canvas.get_renderer()
        for node, txt in label_artists.items():
            bbox = txt.get_window_extent(renderer=renderer)
            if bbox.contains(event.x, event.y):
                hovered_node = node
                break

        # If not over a label, remove highlight
        if hovered_node is None:
            if current_highlight_cluster is not None:
                current_highlight_cluster = None
                unhighlight_all()
                fig.canvas.draw_idle()
            return

        # Hovering over a label -> get its cluster
        cid = cluster_for_node.get(hovered_node, None)
        if cid is None:
            return

        # If it is a new cluster, reset then highlight that cluster
        if cid != current_highlight_cluster:
            current_highlight_cluster = cid
            unhighlight_all()
            highlight_cluster(cid)
            fig.canvas.draw_idle()


    # initial draw and wiring up callbacks
    update(None)
    cut_slider.on_changed(update)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.show()