#!/usr/bin/env python3
"""
Exercises memory_map_optimizer.py's logic with synthetic data, standing in
for the real ~/.graphify graph JSON and ~/.claude/projects/*/memory files
this sandbox doesn't have access to. Everything here is network-free --
_embed_texts (the one function that calls out to Hugging Face via
sentence-transformers) is never invoked; embeddings below are hand-built
vectors chosen to have known cosine similarities, so hybrid_search/
find_duplicate_groups correctness can be checked exactly, not just "ran
without crashing."
"""
import json
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import memory_map_optimizer as mmo


def test_load_graph_json_dict_shape():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "graph.json"
        p.write_text(json.dumps({
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}],
        }))
        nodes, edges, dangling = mmo._load_graph_json(p, "nodes", "edges", "source", "target", "id")
        assert set(nodes) == {"a", "b", "c"}
        assert edges == [("a", "b"), ("b", "c")]
        assert dangling == []
    print("test_load_graph_json_dict_shape OK")


def test_load_graph_json_dangling_edge_endpoint():
    """Regression test for the real graphify data: an edge referencing a
    node id ('Bazartalks_Py2Cplus::json'-shaped) with no matching entry in
    "nodes" -- must not KeyError building the igraph object, must surface
    the endpoint as "dangling" instead of silently dropping the edge."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "graph.json"
        p.write_text(json.dumps({
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"},
                      {"source": "b", "target": "Bazartalks_Py2Cplus::json"}],
        }))
        nodes, edges, dangling = mmo._load_graph_json(p, "nodes", "edges", "source", "target", "id")
        assert set(nodes) == {"a", "b", "Bazartalks_Py2Cplus::json"}
        assert dangling == ["Bazartalks_Py2Cplus::json"]
        # must not raise -- this is the exact KeyError hit on real data
        result = mmo.run_leiden(nodes, edges, seed=0)
        assert result.num_nodes == 3
    print("test_load_graph_json_dangling_edge_endpoint OK")


def test_load_graph_json_edge_list_shape():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "graph.json"
        p.write_text(json.dumps({"edges": [["x", "y"], ["y", "z"]]}))
        nodes, edges, dangling = mmo._load_graph_json(p, "nodes", "edges", "source", "target", "id")
        assert set(nodes) == {"x", "y", "z"}
        assert edges == [("x", "y"), ("y", "z")]
        assert dangling == ["x", "y", "z"]  # no "nodes" key at all -- every endpoint is "dangling"
    print("test_load_graph_json_edge_list_shape OK")


def _synthetic_graph(n_clusters=4, cluster_size=30, cross_edges=15, seed=7):
    """Build a graph with real community structure: dense within-cluster
    edges, sparse cross-cluster edges -- so a correct Leiden run MUST find
    close to n_clusters communities, not 0 and not n_nodes."""
    rnd = random.Random(seed)
    nodes = [f"n{c}_{i}" for c in range(n_clusters) for i in range(cluster_size)]
    edges = []
    clusters = [[f"n{c}_{i}" for i in range(cluster_size)] for c in range(n_clusters)]
    for members in clusters:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if rnd.random() < 0.3:
                    edges.append((members[i], members[j]))
    for _ in range(cross_edges):
        ca, cb = rnd.sample(range(n_clusters), 2)
        edges.append((rnd.choice(clusters[ca]), rnd.choice(clusters[cb])))
    return nodes, edges, n_clusters


def test_leiden_finds_real_communities():
    nodes, edges, expected_clusters = _synthetic_graph()
    result = mmo.run_leiden(nodes, edges, resolution=1.0, seed=0)

    assert len(result.node_to_cluster) == len(nodes)
    assert result.num_nodes == len(nodes)
    assert result.num_edges == len(edges)
    # THE graphify bug this fixes: community detection "returns 0". Assert
    # we never do that, and that we find something close to the planted
    # structure (allow some slack -- Leiden resolution isn't exact).
    assert len(result.cluster_sizes) > 0, "must not return 0 communities"
    assert 2 <= len(result.cluster_sizes) <= expected_clusters * 3, (
        f"expected roughly {expected_clusters} communities, got {len(result.cluster_sizes)}")
    assert result.modularity > 0.2, "modularity should reflect real cluster structure, not noise"
    print(f"test_leiden_finds_real_communities OK "
          f"({len(result.cluster_sizes)} communities found, planted {expected_clusters}, "
          f"modularity {result.modularity:.3f})")


def test_leiden_no_edges_still_returns_result():
    """Isolated nodes (no edges at all) shouldn't crash -- each node its own community."""
    result = mmo.run_leiden(["a", "b", "c"], [], seed=0)
    assert len(result.cluster_sizes) == 3
    print("test_leiden_no_edges_still_returns_result OK")


def _unit_vec(n, dims=8):
    """Deterministic pseudo-random unit vector for doc index n."""
    rnd = random.Random(n)
    v = [rnd.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def test_cosine_known_values():
    assert abs(mmo.cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(mmo.cosine([1, 0], [0, 1]) - 0.0) < 1e-9
    assert abs(mmo.cosine([1, 0], [-1, 0]) - (-1.0)) < 1e-9
    assert mmo.cosine([0, 0], [1, 1]) == 0.0  # degenerate, must not divide by zero
    print("test_cosine_known_values OK")


def test_build_index_threads_max_seq_length_and_batch_size():
    """Regression test for the real crash: an unbounded-length memory file
    made Qwen3-Embedding-0.6B's attention mask try to allocate 17.23 GiB.
    The fix caps max_seq_length/batch_size in _embed_texts -- this checks
    build_index actually passes non-default values through to embed_fn
    rather than silently dropping them (a mocked embed_fn stands in for
    the real HF call, which this sandbox can't make)."""
    seen_calls = []

    def fake_embed_fn(texts, model_name, max_seq_length, batch_size):
        seen_calls.append((max_seq_length, batch_size))
        return [[float(len(t))] for t in texts]

    docs = [("a.md", "short"), ("b.md", "also short")]
    indexed = mmo.build_index(docs, "fake-model", max_seq_length=999, batch_size=3,
                              embed_fn=fake_embed_fn)
    assert seen_calls == [(999, 3)]
    assert len(indexed) == 2
    assert indexed[0].embedding == [5.0]
    print("test_build_index_threads_max_seq_length_and_batch_size OK")


def test_hybrid_search_ranks_relevant_doc_first():
    docs = [
        mmo.MemoryDoc(path="a.md", text="token optimizer routing policy for model selection",
                      text_hash="h1", embedding=[1.0, 0.0, 0.0]),
        mmo.MemoryDoc(path="b.md", text="masaladeutsch blog post about dosa recipes",
                      text_hash="h2", embedding=[0.0, 1.0, 0.0]),
        mmo.MemoryDoc(path="c.md", text="india ministry site access credentials note",
                      text_hash="h3", embedding=[0.0, 0.0, 1.0]),
    ]
    # Query embedding aligned with doc a's embedding -- a.md should win on
    # cosine; query text also shares "token optimizer" words -- a.md should
    # ALSO win on BM25. Fused rank must put it first.
    query_vec = [1.0, 0.01, 0.0]
    results = mmo.hybrid_search("token optimizer routing", docs, query_vec, k=3)
    assert results[0][0].path == "a.md", f"expected a.md first, got {results[0][0].path}"
    assert len(results) == 3
    print("test_hybrid_search_ranks_relevant_doc_first OK")


def test_find_duplicate_groups_merges_transitively():
    # 4 near-identical "snapshot" docs (pairwise similarity ~1.0 via a shared
    # base vector + tiny noise) + 1 unrelated doc -- mirrors the audit's
    # real "4 sequential token-optimizer snapshots -> merged into 1" case.
    base = _unit_vec(0)
    def near(base, jitter, dims=8):
        rnd = random.Random(jitter)
        v = [b + rnd.gauss(0, 0.01) for b in base]
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]

    docs = [mmo.MemoryDoc(path=f"snap{i}.md", text=f"snapshot {i}", text_hash=f"s{i}",
                          embedding=near(base, i)) for i in range(4)]
    docs.append(mmo.MemoryDoc(path="unrelated.md", text="unrelated", text_hash="u",
                              embedding=_unit_vec(999)))

    groups = mmo.find_duplicate_groups(docs, threshold=0.95)
    assert len(groups) == 1, f"expected 1 group, got {len(groups)}: {groups}"
    assert set(groups[0]) == {"snap0.md", "snap1.md", "snap2.md", "snap3.md"}
    print("test_find_duplicate_groups_merges_transitively OK")


def test_find_duplicate_groups_exact_hash_match_always_grouped():
    # Identical text (same hash) must group even if embedding cosine is
    # computed as slightly under threshold due to float noise.
    docs = [
        mmo.MemoryDoc(path="x.md", text="same", text_hash="deadbeef", embedding=[1.0, 0.0]),
        mmo.MemoryDoc(path="y.md", text="same", text_hash="deadbeef", embedding=[0.0, 1.0]),
    ]
    groups = mmo.find_duplicate_groups(docs, threshold=0.99)
    assert groups == [["x.md", "y.md"]]
    print("test_find_duplicate_groups_exact_hash_match_always_grouped OK")


def test_report_runs_without_index_or_clusters():
    """report must degrade gracefully when embed/cluster haven't run yet --
    this is the first thing a new user will hit."""
    import argparse
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.html"
        args = argparse.Namespace(index="/nonexistent/index.json",
                                  clusters="/nonexistent/clusters.json",
                                  dedup="/nonexistent/dedup.json",
                                  out=str(out))
        rc = mmo.cmd_report(args)
        assert rc == 0
        html = out.read_text()
        assert "Memory Map v2" in html
        assert "run `cluster` first" in html
    print("test_report_runs_without_index_or_clusters OK")


if __name__ == "__main__":
    test_load_graph_json_dict_shape()
    test_load_graph_json_dangling_edge_endpoint()
    test_load_graph_json_edge_list_shape()
    test_leiden_finds_real_communities()
    test_leiden_no_edges_still_returns_result()
    test_cosine_known_values()
    test_build_index_threads_max_seq_length_and_batch_size()
    test_hybrid_search_ranks_relevant_doc_first()
    test_find_duplicate_groups_merges_transitively()
    test_find_duplicate_groups_exact_hash_match_always_grouped()
    test_report_runs_without_index_or_clusters()
    print("\nALL TESTS PASSED")
