import os
import pickle
import joblib
import dgl
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    resource_allocation_index,
    preferential_attachment,
    adamic_adar_index
)


def build_nx_graph(graph: dgl.DGLGraph) -> nx.Graph:
    """Convert DGLGraph to undirected NetworkX graph"""
    # 1. First convert DGLGraph to a directed/multigraph of NetworkX
    multi = dgl.to_networkx(graph)
    # 2. Then create an undirected Graph and add all nodes and edges
    g = nx.Graph()
    g.add_nodes_from(multi.nodes())
    g.add_edges_from(multi.edges())
    return g


class LinkPredictor:
    """
    Automatically map and load based on the provided dataset name ("small" or "large"):
      - DGL graph file (graph_path)
      - Trained model directory (model_dir)
      - Cache directory (cache_dir)
    Supports caching NetworkX graphs and similarity dictionaries to disk, and directly loading them in subsequent usage to avoid recomputation.
    """

    # Predefined dataset path mapping
    DATASET_CONFIG = {
        "small": {
            "graph_path": "/hy-tmp/cot-prompt-words/preprocessing/small_disease.bin",
            "model_dir": "/hy-tmp/cot-prompt-words/rag/graph_model/small",
            "cache_dir": "/hy-tmp/cot-prompt-words/rag/graph_model/small/cache"
        },
        "large": {
            "graph_path": "/hy-tmp/cot-prompt-words/preprocessing/large_disease.bin",
            "model_dir": "/hy-tmp/cot-prompt-words/rag/graph_model/large",
            "cache_dir": "/hy-tmp/cot-prompt-words/rag/graph_model/large/cache"
        }
    }

    def __init__(self, dataset_name: str):
        """
        Constructor only requires passing dataset_name: "small" or "large".
        Automatically sets graph_path, model_dir, cache_dir based on mapping, and tries to load cache from cache_dir.
        """
        # 1. Check dataset_name validity
        if dataset_name not in self.DATASET_CONFIG:
            raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Choose from {list(self.DATASET_CONFIG.keys())}.")

        cfg = self.DATASET_CONFIG[dataset_name]
        graph_path = cfg["graph_path"]
        model_dir = cfg["model_dir"]
        cache_dir = cfg["cache_dir"]

        # 2. First load DGL graph from disk to ensure self.graph always exists
        graphs_list, _ = dgl.load_graphs(graph_path)
        self.graph = graphs_list[0]

        # 3. Try to load from cache_dir: if directory exists and contains required cache files, load directly
        self.cache_dir = cache_dir
        if os.path.isdir(cache_dir) and self._check_cache_files(cache_dir):
            self._load_cache(cache_dir)
        else:
            # Cache does not exist or is incomplete, then construct NetworkX graph from self.graph, delay similarity computation
            self.nx_graph = build_nx_graph(self.graph)
            self.j_scores = None
            self.ra_scores = None
            self.pa_scores = None
            self.aa_scores = None

        # 4. Load trained model and threshold dictionary
        self.model, self.thresholds = self._load_model_and_thresholds(model_dir)

        # 5. Define feature column names same as during training
        self.columns = [
            'jaccard_gt_mean', 'jaccard_gt_median', 'jaccard_between_25_75',
            'resource_allocation_gt_mean', 'resource_allocation_gt_median', 'resource_allocation_between_25_75',
            'preferential_attachment_gt_mean', 'preferential_attachment_gt_median', 'preferential_attachment_between_25_75',
            'adamic_adar_gt_mean', 'adamic_adar_gt_median', 'adamic_adar_between_25_75'
        ]

    @staticmethod
    def _load_model_and_thresholds(model_dir: str):
        """Load sklearn/LightGBM models and threshold dictionary"""
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl model files found in {model_dir}")
        model = joblib.load(os.path.join(model_dir, pkl_files[0]))
        thresholds = np.load(os.path.join(model_dir, 'thresholds.npy'), allow_pickle=True).item()
        return model, thresholds

    def _check_cache_files(self, cache_dir: str) -> bool:
        """
        Check if all required cache files exist in cache_dir:
          - nx_graph.pkl
          - j_scores.pkl
          - ra_scores.pkl
          - pa_scores.pkl
          - aa_scores.pkl
        """
        required = [
            "nx_graph.pkl",
            "j_scores.pkl",
            "ra_scores.pkl",
            "pa_scores.pkl",
            "aa_scores.pkl"
        ]
        return all(os.path.isfile(os.path.join(cache_dir, fn)) for fn in required)

    def cache_precompute(self):
        """
        1. If cache_dir does not exist, create it automatically;
        2. Serialize NetworkX graph and four similarity dictionaries to cache_dir,
           so they can be directly loaded later to avoid recomputation.
        """
        cache_dir = self.cache_dir
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Ensure self.nx_graph has been constructed
        if not hasattr(self, "nx_graph") or self.nx_graph is None:
            self.nx_graph = build_nx_graph(self.graph)

        # Compute four similarities (if not computed yet)
        if self.j_scores is None or self.ra_scores is None or self.pa_scores is None or self.aa_scores is None:
            self.precompute_similarity()

        # 1) Save NetworkX graph to pickle file
        nx_path = os.path.join(cache_dir, "nx_graph.pkl")
        with open(nx_path, "wb") as f:
            pickle.dump(self.nx_graph, f)

        # 2) Save four similarity dictionaries
        joblib.dump(self.j_scores, os.path.join(cache_dir, "j_scores.pkl"))
        joblib.dump(self.ra_scores, os.path.join(cache_dir, "ra_scores.pkl"))
        joblib.dump(self.pa_scores, os.path.join(cache_dir, "pa_scores.pkl"))
        joblib.dump(self.aa_scores, os.path.join(cache_dir, "aa_scores.pkl"))

        print(f"NetworkX graph and similarity dictionaries have been cached to: {cache_dir}")

    def _load_cache(self, cache_dir: str):
        """
        Load previously cached NetworkX graph and similarity dictionaries from cache_dir,
        assign them to instance attributes to avoid recomputation.
        """
        # 1) Read NetworkX graph (using pickle)
        nx_path = os.path.join(cache_dir, "nx_graph.pkl")
        if not os.path.isfile(nx_path):
            raise FileNotFoundError(f"{nx_path} does not exist, cannot load NetworkX graph")
        with open(nx_path, "rb") as f:
            self.nx_graph = pickle.load(f)

        # 2) Read four similarity dictionaries
        self.j_scores = joblib.load(os.path.join(cache_dir, "j_scores.pkl"))
        self.ra_scores = joblib.load(os.path.join(cache_dir, "ra_scores.pkl"))
        self.pa_scores = joblib.load(os.path.join(cache_dir, "pa_scores.pkl"))
        self.aa_scores = joblib.load(os.path.join(cache_dir, "aa_scores.pkl"))

        print(f"NetworkX graph and similarity dictionaries have been loaded from {cache_dir}.")

    def precompute_similarity(self):
        """
        Compute and cache four similarity scores (Jaccard, RA, PA, AA) for all edges in the graph,
        results are stored in self.j_scores, self.ra_scores, self.pa_scores, self.aa_scores.
        """
        # Get all edges from DGL graph
        src_list, dst_list = self.graph.edges()
        edges = list(zip(src_list.tolist(), dst_list.tolist()))

        # Compute and store in dictionaries
        self.j_scores  = { (u, v): p for u, v, p in jaccard_coefficient(self.nx_graph, edges) }
        self.ra_scores = { (u, v): p for u, v, p in resource_allocation_index(self.nx_graph, edges) }
        self.pa_scores = { (u, v): p for u, v, p in preferential_attachment(self.nx_graph, edges) }
        self.aa_scores = { (u, v): p for u, v, p in adamic_adar_index(self.nx_graph, edges) }

    def _make_feature_vector(self, src: int, dst: int) -> pd.DataFrame:
        """
        Retrieve scores from cache and generate DataFrame feature vector based on two nodes,
        if similarity scores have not been computed yet, call precompute_similarity() first.
        """
        if self.j_scores is None:
            self.precompute_similarity()

        raw = {
            'jaccard':               self.j_scores.get((src, dst), 0.0),
            'resource_allocation':   self.ra_scores.get((src, dst), 0.0),
            'preferential_attachment': self.pa_scores.get((src, dst), 0.0),
            'adamic_adar':           self.aa_scores.get((src, dst), 0.0)
        }
        feat = []
        for metric in ['jaccard', 'resource_allocation', 'preferential_attachment', 'adamic_adar']:
            mean_, med_, p25_, p75_ = self.thresholds[metric]
            v = raw[metric]
            feat.append(int(v > mean_))
            feat.append(int(v > med_))
            feat.append(int(p25_ <= v <= p75_))

        return pd.DataFrame([feat], columns=self.columns)

    def predict_edge(self, src: int, dst: int) -> tuple:
        """
        Predict single edge: returns (positive_probability, predicted_label)
        """
        X = self._make_feature_vector(src, dst)
        prob  = self.model.predict_proba(X)[0, 1]
        label = int(self.model.predict(X)[0])
        return prob, label


# ----------------------------------------------
# Example usage
# ----------------------------------------------
if __name__ == "__main__":
    # Case1: First use of "small" dataset, needs to cache first
    predictor_small = LinkPredictor("small")
    predictor_small.cache_precompute()

    # Case2: Subsequent use of "small" dataset, will automatically load from cache directory without recomputation
    predictor_small2 = LinkPredictor("small")
    src_id, dst_id = 98, 274
    prob, label = predictor_small2.predict_edge(src_id, dst_id)
    print(f"[small] Edge ({src_id},{dst_id}) -> Prob: {prob:.4f}, Label: {label}")

    # Case3: Use "large" dataset, same logic
    predictor_large = LinkPredictor("large")
    predictor_large.cache_precompute()

    predictor_large2 = LinkPredictor("large")
    prob_l, label_l = predictor_large2.predict_edge(src_id, dst_id)
    print(f"[large] Edge ({src_id},{dst_id}) -> Prob: {prob_l:.4f}, Label: {label_l}")