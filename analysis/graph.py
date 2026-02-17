
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain  # python-louvain



class GraphAnalysis:
    """
    Stores connectivity, metadata, builds the graph ONCE, and computes/caches all graph metrics as fields.
    """
    def __init__(
        self,
        conn_mat: np.ndarray,
        elec_names: list[str],
        directed: bool = False,
        mat_idx: int = 0,
        agg: str | None = None,
        threshold: float | None = None,
        mask: np.ndarray | None = None,
    ):
        self.conn_mat = conn_mat
        self.elec_names = elec_names
        self.directed = directed
        self.mat_idx = mat_idx
        self.agg = agg
        self.threshold = threshold

        self.mask = mask

        self._graph = None
        self.density = None
        self.total_num_edges = None
        self.visible_num_edges = None
        self.num_nodes = None
        self.node_degrees = None
        self.node_strengths = None
        self.global_eff = None
        self.local_eff = None
        self.modularity = None
        self.partition = None

        self._build_graph_and_metrics()



    def _get_weight_matrix(self) -> np.ndarray:
        conn = np.asarray(self.conn_mat)
        if conn.ndim == 2:
            return conn
        if self.agg is None or self.agg == "single":
            return conn[int(self.mat_idx)]
        agg = self.agg.lower()
        if agg == "mean":
            return np.nanmean(conn, axis=0)
        if agg == "sum":
            return np.nansum(conn, axis=0)
        if agg == "median":
            return np.nanmedian(conn, axis=0)
        raise ValueError(f"Invalid agg='{self.agg}'")

    def _build_graph_and_metrics(self):
        W = self._get_weight_matrix()
        if self.threshold is not None:
            W = W.copy()
            W[np.abs(W) < self.threshold] = 0.0
            print(f"Applied threshold: {self.threshold}, resulting in {np.sum(W != 0)} edges")
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
        self._compute_metrics()

    def _compute_metrics(self):
        G = self._graph
        self.density = nx.density(G)
        self.num_nodes = G.number_of_nodes()
        self.total_num_edges = self._get_total_number_of_edges(self.num_nodes, self.directed)
        # Use mask if provided, else fallback to G.number_of_edges()
        if self.mask is not None:
            self.visible_num_edges = self._get_visible_number_of_edges(self.mask, self.directed)
        else:
            self.visible_num_edges = G.number_of_edges()
        self.node_degrees = self._node_in_out_bidirectional_counts(G)
        self.node_strengths = self._node_connection_strengths(G)
        self.global_eff = self._weighted_global_efficiency(G)
        self.local_eff = self._weighted_local_efficiency(G)
        self.modularity, self.partition = self._modularity(G)
    
    def _prepare_distance_graph(self, G: nx.Graph) -> nx.Graph:
        G_dist = G.copy()
        for u, v, d in G_dist.edges(data=True):
            w = d.get("weight", 0.0)

            if w == 0:
                d["weight"] = np.inf
            else:
                d["weight"] = 1.0 / abs(w)

        return G_dist

    def _weighted_global_efficiency(self, G: nx.Graph) -> float:
        if G.number_of_edges() == 0:
            return 0.0

        G_eff = G.to_undirected() if self.directed else G
        G_dist = self._prepare_distance_graph(G_eff)

        n = G_dist.number_of_nodes()
        if n <= 1:
            return 0.0

        path_lengths = dict(nx.all_pairs_dijkstra_path_length(G_dist, weight="weight"))

        total = 0.0
        for u in G_dist.nodes():
            for v in G_dist.nodes():
                if u != v and v in path_lengths[u]:
                    d = path_lengths[u][v]
                    if d > 0:
                        total += 1.0 / d

        return total / (n * (n - 1))


    def _weighted_local_efficiency(self, G: nx.Graph) -> float:
        G_eff = G.to_undirected() if self.directed else G

        local_vals = []

        for node in G_eff.nodes():
            neighbors = list(G_eff.neighbors(node))

            if len(neighbors) < 2:
                local_vals.append(0.0)
                continue

            subgraph = G_eff.subgraph(neighbors)
            ge = self._weighted_global_efficiency(subgraph)
            local_vals.append(ge)

        return np.mean(local_vals) if local_vals else 0.0


    def _node_in_out_bidirectional_counts(self, G: nx.Graph):
        data = {}
        for node in G.nodes():
            if self.directed:
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

    def _node_connection_strengths(self, G: nx.Graph):
        strengths = {}
        for node in G.nodes():
            if self.directed:
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

    def _get_total_number_of_edges(self, n: int, directed: bool = False) -> int:
        if n < 0:
            raise ValueError("Number of nodes must be non-negative")
            
        if directed:
            return n * (n - 1)
        else:
            return n * (n - 1) // 2
        
    def _get_visible_number_of_edges(self, mask: np.ndarray, directed: bool = False) -> int:
        if directed:
            return int(mask.sum())
        else:
            return int(np.triu(mask, k=1).sum())


    def _modularity(self, G: nx.Graph):
        G_und = G.to_undirected() if self.directed else G
        if G_und.number_of_edges() == 0:
            return 0.0, {}
        part = community_louvain.best_partition(G_und, weight="weight")
        comms = {}
        for node, cid in part.items():
            comms.setdefault(cid, set()).add(node)
        partition = list(comms.values())
        mod = nx.community.modularity(G_und, partition, weight="weight")
        return mod, part

    def summary(self):
        print("Graph Summary:")
        print(f"  Nodes: {self.num_nodes}")
        print(f"  Total Edges: {self.total_num_edges}")
        print(f"  Edges: {self.visible_num_edges}")
        print(f"  Density: {self.density:.4f}")
        print(f"  Global Efficiency: {self.global_eff:.4f}")
        print(f"  Local Efficiency: {self.local_eff:.4f}")
        print(f"  Modularity: {self.modularity:.4f}")

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
