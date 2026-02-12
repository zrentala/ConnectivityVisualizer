import numpy as np
import networkx as nx
import community.community_louvain as community_louvain  # python-louvain

def get_weight_matrix(conn_mat: np.ndarray, mat_idx=0, agg=None) -> np.ndarray:
    """
    Returns a 2D (n_nodes, n_nodes) matrix. Supports:
      - 2D conn_mat
      - 3D conn_mat with single/mean/sum/median aggregation
    """
    conn = np.asarray(conn_mat)

    if conn.ndim == 2:
        return conn

    if agg is None or agg == "single":
        return conn[int(mat_idx)]

    agg = agg.lower()
    if agg == "mean":
        return np.nanmean(conn, axis=0)
    if agg == "sum":
        return np.nansum(conn, axis=0)
    if agg == "median":
        return np.nanmedian(conn, axis=0)

    raise ValueError(f"Invalid agg='{agg}'")


class GraphAnalysis:
    """
    Stores connectivity + metadata, builds the graph ONCE, and
    provides a lightweight API to access the cached graph.
    """
    ### WE SHOULDNT STORE THESE FIELDS. ONLY THE GRAPH. WE SHOULD STORE ALL GRAPHS IN THE CONN MATS
    # UI MANAGER SHOULD CONTROL THE MAT IDX
    def __init__(
        self,
        conn_mat: np.ndarray,
        elec_names: list[str],
        directed: bool = False,
        mat_idx: int = 0,
        agg: str | None = None,
        threshold: float | None = None,
    ):
        self.conn_mat = conn_mat
        self.elec_names = elec_names
        self.directed = directed
        self.mat_idx = mat_idx
        self.agg = agg
        self.threshold = threshold

        self._graph = None
        self._build_graph_once()

    # -----------------------------
    # build/store graph
    # -----------------------------
    def _build_graph_once(self):
        W = get_weight_matrix(self.conn_mat, self.mat_idx, self.agg)

        if self.threshold is not None:
            W = W.copy()
            W[np.abs(W) < self.threshold] = 0.0

        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(self.elec_names)

        for i, src in enumerate(self.elec_names):
            for j, dst in enumerate(self.elec_names):
                if i == j:
                    continue
                w = W[i, j]
                if not np.isnan(w) and w != 0:
                    G.add_edge(src, dst, weight=float(w))

        self._graph = G

    # -----------------------------
    # expose cached graph
    # -----------------------------
    @property
    def graph(self) -> nx.Graph:
        return self._graph

def connection_density(G: nx.Graph) -> float:
    return nx.density(G)

def num_edges(G: nx.Graph) -> int:
    return G.number_of_edges()

def num_nodes(G: nx.Graph) -> int:
    return G.number_of_nodes()

def node_in_out_bidirectional_counts(G: nx.Graph, directed: bool):
    data = {}

    for node in G.nodes():
        if directed:
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            bi_deg = sum(1 for nbr in G.successors(node) if G.has_edge(nbr, node))
        else:
            in_deg = out_deg = bi_deg = G.degree(node)

        data[node] = {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "bidirectional": bi_deg,
        }
    return data

def node_connection_strengths(G: nx.Graph, directed: bool):
    strengths = {}

    for node in G.nodes():
        if directed:
            in_s = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
            out_s = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
            bi_s = sum(
                G[node][nbr]["weight"]
                for nbr in G.successors(node)
                if G.has_edge(nbr, node)
            )
        else:
            in_s = out_s = bi_s = sum(d["weight"] for _, _, d in G.edges(node, data=True))

        strengths[node] = dict(
            in_strength=in_s,
            out_strength=out_s,
            bidirectional_strength=bi_s,
        )

    return strengths

def global_efficiency(G: nx.Graph) -> float:
    return nx.global_efficiency(G)

def local_efficiency(G: nx.Graph) -> float:
    return nx.local_efficiency(G)

def modularity(G: nx.Graph, directed=False):
    G_und = G.to_undirected() if directed else G
    if G_und.number_of_edges() == 0:
        # No edges: modularity is undefined, return 0.0 and empty partition
        return 0.0, {}
    part = community_louvain.best_partition(G_und, weight="weight")

    comms = {}
    for node, cid in part.items():
        comms.setdefault(cid, set()).add(node)
    partition = list(comms.values())

    mod = nx.community.modularity(G_und, partition, weight="weight")
    return mod, part

def graph_summary(ga: GraphAnalysis):
    G = ga.graph

    dens = connection_density(G)
    g_eff = global_efficiency(G)
    l_eff = local_efficiency(G)
    mod, _ = modularity(G, directed=ga.directed)

    print("Graph Summary:")
    print(f"  Nodes: {num_nodes(G)}")
    print(f"  Edges: {num_edges(G)}")
    print(f"  Density: {dens:.4f}")
    print(f"  Global Efficiency: {g_eff:.4f}")
    print(f"  Local Efficiency: {l_eff:.4f}")
    print(f"  Modularity: {mod:.4f}")

# need to implement threhsolding
# identify most efficient/node conn strength/modularity of each node
# identify most _ of each edge


# import networkx as nx
# import numpy as np
# import community as community_louvain  # pip install python-louvain
# from utils.braindata import BrainData


# class GraphAnalysis:
#     """
#     High-level graph analysis class leveraging NetworkX for BrainData with possibly
#     multiple connectivity matrices.

#     - brain_data.conn_mat can be 2D: (n_nodes, n_nodes)
#       or 3D: (n_mat, n_nodes, n_nodes).

#     You can:
#       - choose a specific mat_idx
#       - or aggregate across mats via agg={"mean","sum","median"}.

#     All metrics accept `mat_idx` and `agg` arguments.
#     """

#     def __init__(self):

#     # -----------------------------
#     # Internal helpers
#     # -----------------------------
#     def _get_weight_matrix(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#     ) -> np.ndarray:
#         """
#         Return a 2D (n_nodes, n_nodes) weight matrix.

#         If conn_mat is 3D (n_mat, n_nodes, n_nodes):
#           - agg is None / "single": use brain_data.conn_mat[mat_idx]
#           - agg == "mean": average over axis 0
#           - agg == "sum": sum over axis 0
#           - agg == "median": median over axis 0

#         If conn_mat is 2D, returns it directly (mat_idx/agg ignored).
#         """
#         conn = self.brain_data.conn_mat
#         if conn.ndim == 2:
#             return conn

#         # 3D: (n_mat, n_nodes, n_nodes)
#         if agg is None or agg == "single":
#             if mat_idx is None:
#                 mat_idx = 0
#             return conn[int(mat_idx), :, :]

#         agg = agg.lower()
#         if agg == "mean":
#             return np.nanmean(conn, axis=0)
#         elif agg == "sum":
#             return np.nansum(conn, axis=0)
#         elif agg == "median":
#             return np.nanmedian(conn, axis=0)
#         else:
#             raise ValueError(f"Unknown agg='{agg}'. Use None/'single', 'mean', 'sum', or 'median'.")

#     def _build_graph(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ) -> nx.Graph:
#         """
#         Build a NetworkX graph from the chosen/aggregated connectivity matrix.

#         threshold (optional): if provided, zeroes out edges with |weight| < threshold.
#         """
#         W = self._get_weight_matrix(mat_idx=mat_idx, agg=agg)

#         if threshold is not None:
#             W = W.copy()
#             W[np.abs(W) < threshold] = 0.0

#         G = nx.DiGraph() if self.directed else nx.Graph()
#         G.add_nodes_from(self.elec_names)

#         for i, src in enumerate(self.elec_names):
#             for j, dst in enumerate(self.elec_names):
#                 if i == j:
#                     continue
#                 weight = W[i, j]
#                 if not np.isnan(weight) and weight != 0:
#                     G.add_edge(src, dst, weight=float(weight))
#         return G

#     # -----------------------------
#     # Basic graph stats
#     # -----------------------------
#     def connection_density(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ) -> float:
#         """Fraction of actual edges to possible edges."""
#         G = self._build_graph(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         return nx.density(G)

#     def num_edges(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ) -> int:
#         """Number of edges in the graph."""
#         G = self._build_graph(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         return G.number_of_edges()

#     def num_nodes(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#     ) -> int:
#         """Number of nodes in the graph (usually constant across mats)."""
#         G = self._build_graph(mat_idx=mat_idx, agg=agg)
#         return G.number_of_nodes()

#     # -----------------------------
#     # Node-level connection counts
#     # -----------------------------
#     def node_in_out_bidirectional_counts(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ):
#         """
#         Returns per-node in-degree, out-degree, and bidirectional connection counts.
#         For undirected graphs, in == out == bidirectional == degree.
#         """
#         G = self._build_graph(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         data = {}
#         for node in G.nodes():
#             if self.directed:
#                 in_deg = G.in_degree(node)
#                 out_deg = G.out_degree(node)
#                 bi_deg = sum(1 for nbr in G.successors(node) if G.has_edge(nbr, node))
#             else:
#                 in_deg = out_deg = bi_deg = G.degree(node)
#             data[node] = dict(in_degree=in_deg, out_degree=out_deg, bidirectional=bi_deg)
#         return data

#     # -----------------------------
#     # Node connection strengths
#     # -----------------------------
#     def node_connection_strengths(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ):
#         """
#         Computes weighted in/out/bidirectional connection strengths for each node.
#         """
#         G = self._build_graph(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         strengths = {}
#         for node in G.nodes():
#             if self.directed:
#                 in_strength = sum(d["weight"] for _, _, d in G.in_edges(node, data=True))
#                 out_strength = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
#                 bi_strength = sum(
#                     G[node][nbr]["weight"]
#                     for nbr in G.successors(node)
#                     if G.has_edge(nbr, node)
#                 )
#             else:
#                 in_strength = out_strength = bi_strength = sum(
#                     d["weight"] for _, _, d in G.edges(node, data=True)
#                 )
#             strengths[node] = dict(
#                 in_strength=in_strength,
#                 out_strength=out_strength,
#                 bidirectional_strength=bi_strength,
#             )
#         return strengths

#     # -----------------------------
#     # Efficiency metrics
#     # -----------------------------
#     def global_efficiency(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ) -> float:
#         """Global efficiency = average inverse shortest path length."""
#         G = self._build_graph(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         return nx.global_efficiency(G)

#     def local_efficiency(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ) -> float:
#         """Local efficiency = mean of node neighborhood efficiencies."""
#         G = self._build_graph(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         return nx.local_efficiency(G)

#     # -----------------------------
#     # Modularity (Louvain)
#     # -----------------------------
#     def modularity(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ):
#         """
#         Computes Louvain modularity and community partition for the chosen/aggregated matrix.

#         Returns:
#             modularity (float), partition (dict[node -> community_id])
#         """
#         G = self._build_graph(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         if self.directed:
#             G_undirected = G.to_undirected()
#         else:
#             G_undirected = G

#         # Using python-louvain for partition (works with weighted graphs)
#         part_dict = community_louvain.best_partition(G_undirected, weight="weight")
#         # Convert to list-of-sets partition for modularity
#         communities = {}
#         for node, cid in part_dict.items():
#             communities.setdefault(cid, set()).add(node)
#         partition = list(communities.values())
#         mod = nx.community.modularity(G_undirected, partition, weight="weight")
#         return mod, part_dict

#     # -----------------------------
#     # Convenience summary
#     # -----------------------------
#     def summary(
#         self,
#         mat_idx: int | None = 0,
#         agg: str | None = None,
#         threshold: float | None = None,
#     ):
#         density = self.connection_density(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         eff_global = self.global_efficiency(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         eff_local = self.local_efficiency(mat_idx=mat_idx, agg=agg, threshold=threshold)
#         mod, _ = self.modularity(mat_idx=mat_idx, agg=agg, threshold=threshold)

#         print("Graph Summary:")
#         print(f"  Matrix: {'agg='+str(agg) if agg not in (None,'single') else mat_idx}")
#         print(f"  Nodes: {self.num_nodes(mat_idx=mat_idx, agg=agg)}")
#         print(f"  Edges: {self.num_edges(mat_idx=mat_idx, agg=agg, threshold=threshold)}")
#         print(f"  Density: {density:.4f}")
#         print(f"  Global Efficiency: {eff_global:.4f}")
#         print(f"  Local Efficiency: {eff_local:.4f}")
#         print(f"  Modularity: {mod:.4f}")
