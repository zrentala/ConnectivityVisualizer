import networkx as nx
import numpy as np
import community as community_louvain  # pip install python-louvain

class GraphAnalysis:
    """
    High-level graph analysis class leveraging NetworkX.

    Supports:
      - Directed or undirected weighted graphs
      - Connection density
      - Global & local efficiency
      - Modularity (Louvain)
      - In/out/bidirectional degree counts
      - Node connection strengths (in/out/bidirectional)
    """

    def __init__(self, conn_matrix, elec_names=None, directed=False):
        self.directed = directed
        self.n = conn_matrix.shape[0]
        self.elec_names = elec_names or [f"Elec{i}" for i in range(self.n)]

        # Build graph
        self.G = nx.DiGraph() if directed else nx.Graph()
        for i, src in enumerate(self.elec_names):
            for j, dst in enumerate(self.elec_names):
                weight = conn_matrix[i, j]
                if i != j and not np.isnan(weight) and weight != 0:
                    self.G.add_edge(src, dst, weight=weight)

    # -----------------------------
    # Basic graph stats
    # -----------------------------
    def connection_density(self) -> float:
        """Fraction of actual edges to possible edges."""
        return nx.density(self.G)

    def num_edges(self):
        """Number of edges in the graph."""
        return self.G.number_of_edges()

    def num_nodes(self):
        """Number of nodes in the graph."""
        return self.G.number_of_nodes()

    # -----------------------------
    # Node-level connection counts
    # -----------------------------
    def node_in_out_bidirectional_counts(self):
        """
        Returns per-node in-degree, out-degree, and bidirectional connection counts.
        For undirected graphs, in == out == bidirectional == degree.
        """
        data = {}
        for node in self.G.nodes():
            if self.directed:
                in_deg = self.G.in_degree(node)
                out_deg = self.G.out_degree(node)
                # Count bidirectional edges
                bi_deg = sum(1 for nbr in self.G.successors(node)
                             if self.G.has_edge(nbr, node))
            else:
                in_deg = out_deg = bi_deg = self.G.degree(node)
            data[node] = dict(in_degree=in_deg, out_degree=out_deg, bidirectional=bi_deg)
        return data

    # -----------------------------
    # Node connection strengths
    # -----------------------------
    def node_connection_strengths(self):
        """
        Computes weighted in/out/bidirectional connection strengths for each node.
        """
        strengths = {}
        for node in self.G.nodes():
            if self.directed:
                in_strength = sum(d["weight"] for _, _, d in self.G.in_edges(node, data=True))
                out_strength = sum(d["weight"] for _, _, d in self.G.out_edges(node, data=True))
                bi_strength = sum(self.G[node][nbr]["weight"]
                                  for nbr in self.G.successors(node)
                                  if self.G.has_edge(nbr, node))
            else:
                in_strength = out_strength = bi_strength = sum(
                    d["weight"] for _, _, d in self.G.edges(node, data=True)
                )
            strengths[node] = dict(in_strength=in_strength,
                                   out_strength=out_strength,
                                   bidirectional_strength=bi_strength)
        return strengths

    # -----------------------------
    # Efficiency metrics
    # -----------------------------
    def global_efficiency(self):
        """Global efficiency = average inverse shortest path length."""
        return nx.global_efficiency(self.G)

    def local_efficiency(self):
        """Local efficiency = mean of node neighborhood efficiencies."""
        return nx.local_efficiency(self.G)

    # -----------------------------
    # Modularity (Louvain)
    # -----------------------------
    def modularity(self):
        """
        Computes Louvain modularity and community partition.
        Returns:
            modularity (float), partition (dict[node -> community_id])
        """
        if self.directed:
            # Louvain only supports undirected graphs, so use symmetrized version
            G_undirected = self.G.to_undirected()
        else:
            G_undirected = self.G

        partition = nx.community.louvain_communities(G_undirected)
        mod = nx.community.modularity(G_undirected, partition)
        return mod, partition

    # -----------------------------
    # Convenience summary
    # -----------------------------
    def summary(self):
        density = self.connection_density()
        eff_global = self.global_efficiency()
        eff_local = self.local_efficiency()
        mod, _ = self.modularity()

        print(f"Graph Summary:")
        print(f"  Nodes: {self.num_nodes()}")
        print(f"  Edges: {self.num_edges()}")
        print(f"  Density: {density:.4f}")
        print(f"  Global Efficiency: {eff_global:.4f}")
        print(f"  Local Efficiency: {eff_local:.4f}")
        print(f"  Modularity: {mod:.4f}")
