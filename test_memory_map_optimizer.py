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


def test_rerank_threads_max_length_and_batch_size_and_reorders():
    """Regression test for the real crash: --rerank hit an MPS out-of-memory
    error the same way embed() hit a CPU one, same root cause (no length
    cap on the CrossEncoder call). Checks both that the cap values reach
    predict_fn (mocked -- this sandbox can't make the real HF call either)
    and that rerank() actually reorders by the returned scores, not just
    passes them through."""
    seen_calls = []

    def fake_predict_fn(pairs, model_name, max_length, batch_size):
        seen_calls.append((max_length, batch_size))
        # score = length of the doc side of the pair -- deliberately
        # inverts the input order so a passthrough bug would be caught
        return [float(len(p[1])) for p in pairs]

    a = mmo.MemoryDoc(path="short.md", text="hi", text_hash="h1", embedding=[])
    b = mmo.MemoryDoc(path="long.md", text="a much longer document body here",
                      text_hash="h2", embedding=[])
    candidates = [(a, 0.9), (b, 0.1)]  # a ranked first going in

    reranked = mmo.rerank("query", candidates, "fake-model",
                          max_length=777, batch_size=2, predict_fn=fake_predict_fn)

    assert seen_calls == [(777, 2)]
    assert [d.path for d, _ in reranked] == ["long.md", "short.md"]  # b now first
    print("test_rerank_threads_max_length_and_batch_size_and_reorders OK")


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


def test_group_pairwise_similarities_flags_transitive_chain():
    """Regression test for the real Piotroski false-positive: a group of 2
    where find_duplicate_groups says "grouped" but the diff showed almost
    no real overlap. This is the tool that would have caught it directly --
    a 3-doc chain (A-B high, B-C high, A-C low) must report A-C's actual
    (low) similarity, not hide it behind the group's binary membership."""
    a = mmo.MemoryDoc(path="a.md", text="a", text_hash="ha", embedding=[1.0, 0.0, 0.0])
    b = mmo.MemoryDoc(path="b.md", text="b", text_hash="hb", embedding=[0.9, 0.1, 0.0])  # close to a
    c = mmo.MemoryDoc(path="c.md", text="c", text_hash="hc", embedding=[0.0, 0.1, 1.0])  # close to b, not a
    by_path = {d.path: d for d in (a, b, c)}

    pairs = mmo.group_pairwise_similarities(by_path, ["a.md", "b.md", "c.md"])
    assert len(pairs) == 3
    # descending order
    assert pairs[0][2] >= pairs[1][2] >= pairs[2][2]
    # a-c must be the weakest pair -- exactly the transitive-chain signal
    weakest = min(pairs, key=lambda p: p[2])
    assert {weakest[0], weakest[1]} == {"a.md", "c.md"}
    print("test_group_pairwise_similarities_flags_transitive_chain OK")


def test_parse_sweep_arg():
    assert mmo._parse_sweep_arg("0.80:0.90:0.05") == [0.8, 0.85, 0.9]
    assert mmo._parse_sweep_arg("0.9:0.9:0.05") == [0.9]  # single-point range

    try:
        mmo._parse_sweep_arg("not-a-spec")
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        mmo._parse_sweep_arg("0.8:0.9:0")
        assert False, "expected ValueError for non-positive step"
    except ValueError:
        pass
    print("test_parse_sweep_arg OK")


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


# ── added 2026-10-25 (sdlc test-hardening) ────────────────────────────────

def test_tokenize_strips_trailing_punctuation_so_words_match():
    """Memory markdown is prose: "routing." / "(qmd)" / "optimizer," must
    tokenize to the same term a query uses, or BM25 never matches them."""
    toks = mmo._tokenize("Use qmd for routing. See (graphify), optimizer: done")
    for word in ("routing", "graphify", "optimizer", "qmd"):
        assert word in toks, f"{word!r} missing from {toks}"


def test_tokenize_lowercases_and_keeps_short_alnum_words():
    assert mmo._tokenize("A Token OPTIMIZER") == ["a", "token", "optimizer"]


def test_tokenize_drops_pure_punctuation():
    assert mmo._tokenize("-- ... ## |") == []


def test_parse_sweep_arg_single_point_range():
    assert mmo._parse_sweep_arg("0.9:0.9:0.05") == [0.9]


def test_parse_sweep_arg_reversed_range_is_rejected():
    """START > STOP produced an empty list, and cmd_dedup then crashed on
    thresholds[0] with an IndexError instead of a usage error."""
    try:
        mmo._parse_sweep_arg("0.95:0.80:0.05")
    except ValueError:
        return
    assert False, "expected ValueError for START > STOP"


def test_cmd_dedup_reversed_sweep_returns_error_code():
    import argparse
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / "index.json"
        idx.write_text(json.dumps({"model": "m", "docs": [
            {"path": "a.md", "text": "x", "text_hash": "h1", "embedding": [1.0, 0.0]},
            {"path": "b.md", "text": "y", "text_hash": "h2", "embedding": [0.0, 1.0]},
        ]}))
        args = argparse.Namespace(index=str(idx), sweep="0.95:0.80:0.05", threshold=0.85, out=None)
        assert mmo.cmd_dedup(args) == 1


def test_cmd_dedup_writes_groups_json():
    import argparse
    with tempfile.TemporaryDirectory() as d:
        idx = Path(d) / "index.json"
        out = Path(d) / "dedup.json"
        idx.write_text(json.dumps({"model": "m", "docs": [
            {"path": "a.md", "text": "x", "text_hash": "h1", "embedding": [1.0, 0.0]},
            {"path": "b.md", "text": "y", "text_hash": "h2", "embedding": [0.99, 0.01]},
            {"path": "c.md", "text": "z", "text_hash": "h3", "embedding": [0.0, 1.0]},
        ]}))
        args = argparse.Namespace(index=str(idx), sweep=None, threshold=0.9, out=str(out))
        assert mmo.cmd_dedup(args) == 0
        data = json.loads(out.read_text())
        assert data["threshold"] == 0.9
        assert [sorted(g) for g in data["groups"]] == [["a.md", "b.md"]]


def test_find_duplicate_groups_threshold_is_inclusive_and_singletons_dropped():
    # cosine([1,0],[0.6,0.8]) == 0.6 exactly
    docs = [mmo.MemoryDoc("a.md", "x", "h1", [1.0, 0.0]),
            mmo.MemoryDoc("b.md", "y", "h2", [0.6, 0.8]),
            mmo.MemoryDoc("c.md", "z", "h3", [0.0, -1.0])]
    assert [sorted(g) for g in mmo.find_duplicate_groups(docs, threshold=0.6)] == [["a.md", "b.md"]]
    assert mmo.find_duplicate_groups(docs, threshold=0.61) == []
    assert mmo.find_duplicate_groups([], threshold=0.5) == []


def test_cosine_zero_vector_and_opposite():
    assert mmo.cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert math.isclose(mmo.cosine([1.0, 2.0], [-1.0, -2.0]), -1.0)


def test_load_index_roundtrip_and_default_model():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "i.json"
        p.write_text(json.dumps({"docs": [{"path": "a.md", "text": "t", "text_hash": "h", "embedding": [1.0]}]}))
        model, docs = mmo.load_index(p)
        assert model == mmo.DEFAULT_EMBED_MODEL
        assert docs[0].path == "a.md" and docs[0].embedding == [1.0]


def test_read_memory_files_only_markdown_from_directory():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "sub").mkdir()
        (Path(d) / "a.md").write_text("alpha")
        (Path(d) / "sub" / "b.md").write_text("beta")
        (Path(d) / "c.txt").write_text("ignored")
        docs = mmo._read_memory_files(d)
        assert sorted(Path(p).name for p, _ in docs) == ["a.md", "b.md"]
        assert dict((Path(p).name, t) for p, t in docs)["b.md"] == "beta"


def test_build_index_hashes_text_and_handles_empty():
    assert mmo.build_index([], "m", embed_fn=lambda *a, **k: 1 / 0) == []
    out = mmo.build_index([("a.md", "abc")], "m", embed_fn=lambda texts, m, **k: [[0.5]] * len(texts))
    import hashlib
    assert out[0].text_hash == hashlib.sha256(b"abc").hexdigest()[:16]
    assert out[0].embedding == [0.5]


def test_report_escapes_file_paths_in_html():
    """Dedup group paths are file names on disk; they must be HTML-escaped,
    not injected raw into the report page."""
    import argparse
    with tempfile.TemporaryDirectory() as d:
        dd = Path(d) / "dedup.json"
        bad = "<img src=x onerror=alert(1)>.md"
        dd.write_text(json.dumps({"groups": [[bad, "b&c.md"]]}))
        out = Path(d) / "r.html"
        args = argparse.Namespace(index=None, clusters=None, dedup=str(dd), out=str(out))
        assert mmo.cmd_report(args) == 0
        html = out.read_text()
        assert bad not in html
        assert "&lt;img src=x onerror=alert(1)&gt;.md" in html
        assert "b&amp;c.md" in html


def test_report_renders_cluster_counts():
    import argparse
    with tempfile.TemporaryDirectory() as d:
        c = Path(d) / "c.json"
        c.write_text(json.dumps({"num_clusters": 2, "modularity": 0.4123,
                                 "cluster_sizes": {"0": 3, "1": 5}}))
        out = Path(d) / "r.html"
        args = argparse.Namespace(index=None, clusters=str(c), dedup=None, out=str(out))
        assert mmo.cmd_report(args) == 0
        html = out.read_text()
        assert "<b>2</b><span>real communities" in html
        assert html.index("<td>1</td><td>5</td>") < html.index("<td>0</td><td>3</td>")


def test_cmd_cluster_write_back_sets_community_on_nodes():
    pytest = __import__("pytest")
    pytest.importorskip("leidenalg")
    import argparse
    with tempfile.TemporaryDirectory() as d:
        g = Path(d) / "g.json"
        g.write_text(json.dumps({"nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                                 "edges": [{"source": "a", "target": "b"}]}))
        args = argparse.Namespace(graph=str(g), nodes_key="nodes", edges_key="edges",
                                  source_key="source", target_key="target", id_key="id",
                                  resolution=1.0, seed=0, write_back=True, out=str(Path(d) / "c.json"))
        assert mmo.cmd_cluster(args) == 0
        nodes = json.loads(g.read_text())["nodes"]
        comm = {n["id"]: n["community"] for n in nodes}
        assert comm["a"] == comm["b"] != comm["c"]


if __name__ == "__main__":
    test_load_graph_json_dict_shape()
    test_load_graph_json_dangling_edge_endpoint()
    test_load_graph_json_edge_list_shape()
    test_leiden_finds_real_communities()
    test_leiden_no_edges_still_returns_result()
    test_cosine_known_values()
    test_build_index_threads_max_seq_length_and_batch_size()
    test_rerank_threads_max_length_and_batch_size_and_reorders()
    test_hybrid_search_ranks_relevant_doc_first()
    test_find_duplicate_groups_merges_transitively()
    test_find_duplicate_groups_exact_hash_match_always_grouped()
    test_group_pairwise_similarities_flags_transitive_chain()
    test_parse_sweep_arg()
    test_report_runs_without_index_or_clusters()
    print("\nALL TESTS PASSED")
