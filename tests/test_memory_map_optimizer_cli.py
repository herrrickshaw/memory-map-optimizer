"""Offline tests for memory_map_optimizer: loaders, Leiden wrapper, embed/search/dedup/report commands, CLI.

sentence-transformers is replaced by tiny fakes in sys.modules (no model download); igraph, leidenalg and
rank-bm25 are real. Every path lives under tmp_path.
"""
import json
import math
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import memory_map_optimizer as mmo  # noqa: E402


# ---- helpers ----------------------------------------------------------------------------------------

def doc(path, text, emb, h=None):
    return mmo.MemoryDoc(path=path, text=text, text_hash=h or f"h-{path}", embedding=list(emb))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


class Vec(list):
    """Stands in for a numpy row: encode() results expose .tolist()."""
    def tolist(self):
        return list(self)


@pytest.fixture
def fake_st(monkeypatch):
    """Fake `sentence_transformers` recording how the model classes are used."""
    calls = {"st_init": [], "encode": [], "ce_init": [], "predict": []}

    class SentenceTransformer:
        def __init__(self, name):
            calls["st_init"].append(name)
            self.max_seq_length = calls.get("initial_max")

        def encode(self, texts, normalize_embeddings, batch_size, show_progress_bar):
            calls["encode"].append(dict(texts=list(texts), normalize=normalize_embeddings, batch=batch_size,
                                        progress=show_progress_bar, max_seq=self.max_seq_length))
            return [Vec([float(len(t)), 1.0]) for t in texts]

    class CrossEncoder:
        def __init__(self, name, max_length):
            calls["ce_init"].append((name, max_length))

        def predict(self, pairs, batch_size):
            calls["predict"].append((pairs, batch_size))
            return [float(len(p[1])) for p in pairs]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer, mod.CrossEncoder = SentenceTransformer, CrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return calls


# ---- _load_graph_json ---------------------------------------------------------------------------------

def load(path, nk="nodes", ek="edges", sk="source", tk="target", ik="id"):
    return mmo._load_graph_json(path, nk, ek, sk, tk, ik)


def test_load_graph_plain_string_nodes_and_list_edges_with_extra_items(tmp_path):
    p = write_json(tmp_path / "g.json", {"nodes": ["a", "b"], "edges": [["a", "b", 0.5], ("b", "a")]})
    assert load(p) == (["a", "b"], [("a", "b"), ("b", "a")], [])


def test_load_graph_custom_keys(tmp_path):
    p = write_json(tmp_path / "g.json", {"V": [{"name": "x"}, {"name": "y"}], "E": [{"from": "x", "to": "y"}]})
    assert load(p, "V", "E", "from", "to", "name") == (["x", "y"], [("x", "y")], [])


def test_load_graph_without_node_list_derives_nodes_in_edge_order(tmp_path):
    p = write_json(tmp_path / "g.json", {"edges": [["b", "a"], ["a", "c"]]})
    nodes, edges, dangling = load(p)
    assert nodes == ["b", "a", "c"] and dangling == ["b", "a", "c"] and edges == [("b", "a"), ("a", "c")]


def test_load_graph_reports_each_dangling_endpoint_once(tmp_path):
    p = write_json(tmp_path / "g.json", {"nodes": [{"id": "a"}], "edges": [{"source": "a", "target": "z"},
                                                                          {"source": "z", "target": "y"}]})
    nodes, _, dangling = load(p)
    assert nodes == ["a", "z", "y"] and dangling == ["z", "y"]


def test_load_graph_numeric_ids_become_strings(tmp_path):
    p = write_json(tmp_path / "g.json", {"nodes": [{"id": 1}, {"id": 2}], "edges": [[1, 2]]})
    assert load(p) == (["1", "2"], [("1", "2")], [])


@pytest.mark.parametrize("bad", ["abc", 7, ["only-one"], None])
def test_load_graph_unrecognised_edge_shape_raises(tmp_path, bad):
    p = write_json(tmp_path / "g.json", {"nodes": ["a"], "edges": [bad]})
    with pytest.raises(ValueError, match="unrecognized edge shape"):
        load(p)


def test_load_graph_dict_edge_missing_key_raises_keyerror(tmp_path):
    p = write_json(tmp_path / "g.json", {"edges": [{"source": "a"}]})
    with pytest.raises(KeyError):
        load(p)


# ---- run_leiden / ClusterResult --------------------------------------------------------------------------

TWO_CLIQUES = (list("abcdef"), [("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f"), ("c", "d")])


def test_run_leiden_splits_two_cliques():
    r = mmo.run_leiden(*TWO_CLIQUES, seed=1)
    assert r.num_nodes == 6 and r.num_edges == 7 and len(r.cluster_sizes) == 2
    assert r.node_to_cluster["a"] == r.node_to_cluster["b"] == r.node_to_cluster["c"]
    assert r.node_to_cluster["d"] == r.node_to_cluster["e"] == r.node_to_cluster["f"]
    assert r.node_to_cluster["a"] != r.node_to_cluster["d"]
    assert sorted(r.cluster_sizes.values()) == [3, 3] and r.modularity > 0.3


def test_run_leiden_is_deterministic_for_a_seed_and_resolution_changes_granularity():
    assert mmo.run_leiden(*TWO_CLIQUES, seed=5).node_to_cluster == mmo.run_leiden(*TWO_CLIQUES, seed=5).node_to_cluster
    coarse = mmo.run_leiden(*TWO_CLIQUES, resolution=0.01, seed=0)
    fine = mmo.run_leiden(*TWO_CLIQUES, resolution=5.0, seed=0)
    assert len(coarse.cluster_sizes) < len(fine.cluster_sizes)


def test_run_leiden_isolated_nodes_are_their_own_communities():
    r = mmo.run_leiden(["x", "y", "z"], [], seed=0)
    assert len(r.cluster_sizes) == 3 and sorted(r.cluster_sizes.values()) == [1, 1, 1]


def test_run_leiden_without_native_libs_raises_a_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "igraph", None)
    with pytest.raises(RuntimeError, match="python-igraph \\+ leidenalg"):
        mmo.run_leiden(["a"], [])


def test_cluster_result_json_is_sorted_rounded_and_string_keyed():
    r = mmo.ClusterResult(node_to_cluster={"a": 0, "b": 1, "c": 1}, cluster_sizes={0: 1, 1: 2},
                          modularity=0.123456789, num_nodes=3, num_edges=2)
    out = r.to_json()
    assert out["modularity"] == 0.1235 and out["num_clusters"] == 2 and out["num_nodes"] == 3 and out["num_edges"] == 2
    assert list(out["cluster_sizes"].items()) == [("1", 2), ("0", 1)]
    assert out["node_to_cluster"] == {"a": 0, "b": 1, "c": 1}


# ---- cmd_cluster ---------------------------------------------------------------------------------------------

def cluster_args(tmp_path, graph, **kw):
    ns = dict(graph=str(graph), nodes_key="nodes", edges_key="edges", source_key="source", target_key="target",
              id_key="id", resolution=1.0, seed=0, write_back=False, out=str(tmp_path / "out" / "clusters.json"))
    ns.update(kw)
    return types.SimpleNamespace(**ns)


def graph_file(tmp_path):
    nodes, edges = TWO_CLIQUES
    return write_json(tmp_path / "g.json", {"nodes": [{"id": n} for n in nodes],
                                            "edges": [{"source": a, "target": b} for a, b in edges]})


def test_cmd_cluster_missing_graph_and_empty_graph(tmp_path, capsys):
    assert mmo.cmd_cluster(cluster_args(tmp_path, tmp_path / "nope.json")) == 1
    assert "graph file not found" in capsys.readouterr().err
    empty = write_json(tmp_path / "e.json", {"nodes": [], "edges": []})
    assert mmo.cmd_cluster(cluster_args(tmp_path, empty)) == 1
    assert "no nodes found" in capsys.readouterr().err


def test_cmd_cluster_writes_result_and_prints_summary(tmp_path, capsys):
    args = cluster_args(tmp_path, graph_file(tmp_path))
    assert mmo.cmd_cluster(args) == 0
    out = capsys.readouterr().out
    assert "6 nodes, 7 edges -> 2 communities (modularity " in out and "  cluster " in out and f"wrote {args.out}" in out
    data = json.loads(Path(args.out).read_text())
    assert data["num_nodes"] == 6 and data["num_clusters"] == 2 and set(data["node_to_cluster"]) == set("abcdef")


def test_cmd_cluster_prints_only_the_ten_largest_clusters(tmp_path, capsys):
    g = write_json(tmp_path / "g.json", {"nodes": [f"n{i}" for i in range(12)], "edges": []})
    assert mmo.cmd_cluster(cluster_args(tmp_path, g)) == 0
    assert capsys.readouterr().out.count("  cluster ") == 10


def test_cmd_cluster_notes_dangling_endpoints(tmp_path, capsys):
    g = write_json(tmp_path / "g.json", {"nodes": [{"id": "a"}], "edges": [{"source": "a", "target": "ghost"}]})
    assert mmo.cmd_cluster(cluster_args(tmp_path, g)) == 0
    assert "note: 1 edge endpoint(s) had no matching node entry (e.g. 'ghost')" in capsys.readouterr().out


def test_cmd_cluster_write_back_sets_community_only_on_dict_nodes(tmp_path, capsys):
    g = write_json(tmp_path / "g.json", {"nodes": [{"id": "a"}, "b"], "edges": [["a", "b"]]})
    assert mmo.cmd_cluster(cluster_args(tmp_path, g, write_back=True)) == 0
    data = json.loads(g.read_text())
    assert data["nodes"][0]["community"] in (0, 1) and data["nodes"][1] == "b"
    assert "wrote community assignments back" in capsys.readouterr().out


def test_cmd_cluster_without_write_back_leaves_graph_untouched(tmp_path):
    g = graph_file(tmp_path)
    before = g.read_text()
    mmo.cmd_cluster(cluster_args(tmp_path, g))
    assert g.read_text() == before


# ---- _read_memory_files --------------------------------------------------------------------------------------

def make_memory(tmp_path):
    d = tmp_path / "mem"
    (d / "sub").mkdir(parents=True)
    (d / "b.md").write_text("bee")
    (d / "a.md").write_text("ay")
    (d / "sub" / "c.md").write_text("sea")
    (d / "notes.txt").write_text("skip")
    return d


def test_read_memory_files_glob_filters_md_and_sorts(tmp_path):
    d = make_memory(tmp_path)
    got = mmo._read_memory_files(str(d / "**"))
    assert [Path(p).name for p, _ in got] == ["a.md", "b.md", "c.md"]
    assert [t for _, t in got] == ["ay", "bee", "sea"]
    assert [Path(p).name for p, _ in mmo._read_memory_files(str(d / "*.md"))] == ["a.md", "b.md"]


def test_read_memory_files_directory_argument_falls_back_to_recursive_scan(tmp_path):
    d = make_memory(tmp_path)
    assert [Path(p).name for p, _ in mmo._read_memory_files(str(d))] == ["a.md", "b.md", "c.md"]


def test_read_memory_files_nothing_found(tmp_path):
    assert mmo._read_memory_files(str(tmp_path / "missing" / "**")) == []
    (tmp_path / "empty").mkdir()
    assert mmo._read_memory_files(str(tmp_path / "empty")) == []


def test_read_memory_files_skips_unreadable_and_tolerates_bad_bytes(tmp_path, monkeypatch):
    d = make_memory(tmp_path)
    (d / "bad.md").write_bytes(b"ok \xff\xfe bytes")
    real = Path.read_text

    def flaky(self, *a, **k):
        if self.name == "b.md":
            raise OSError("denied")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky)
    got = dict((Path(p).name, t) for p, t in mmo._read_memory_files(str(d / "**")))
    assert "b.md" not in got and got["a.md"] == "ay" and got["bad.md"].startswith("ok ")


def test_read_memory_files_expands_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "x.md").write_text("x")
    assert [Path(p).name for p, _ in mmo._read_memory_files("~/m/*")] == ["x.md"]


# ---- embedding -------------------------------------------------------------------------------------------------

def test_embed_texts_caps_sequence_length_and_batches(fake_st):
    out = mmo._embed_texts(["ab", "abc"], "model-x", max_seq_length=128, batch_size=3)
    assert out == [[2.0, 1.0], [3.0, 1.0]] and all(type(v) is list for v in out)
    c = fake_st["encode"][0]
    assert fake_st["st_init"] == ["model-x"]
    assert (c["normalize"], c["batch"], c["progress"]) == (True, 3, False)
    assert c["max_seq"] == 128                                              # model default None -> capped


@pytest.mark.parametrize("initial,expected", [(None, 100), (512, 100), (100, 100), (64, 64)])
def test_embed_texts_only_ever_lowers_the_models_own_limit(fake_st, initial, expected):
    fake_st["initial_max"] = initial
    mmo._embed_texts(["t"], "m", max_seq_length=100)
    assert fake_st["encode"][0]["max_seq"] == expected


def test_embed_texts_defaults_and_progress_bar_threshold(fake_st):
    mmo._embed_texts(["t"] * 20, "m")
    mmo._embed_texts(["t"] * 21, "m")
    assert [c["progress"] for c in fake_st["encode"]] == [False, True]
    assert fake_st["encode"][0]["batch"] == 8 and fake_st["encode"][0]["max_seq"] == 4096


def test_build_index_hashes_and_pairs_vectors_in_order():
    seen = {}

    def embed(texts, model, max_seq_length, batch_size):
        seen.update(texts=texts, model=model, msl=max_seq_length, bs=batch_size)
        return [[1.0], [2.0]]

    docs = mmo.build_index([("p1", "alpha"), ("p2", "beta")], "m", max_seq_length=7, batch_size=3, embed_fn=embed)
    assert seen == {"texts": ["alpha", "beta"], "model": "m", "msl": 7, "bs": 3}
    assert [d.path for d in docs] == ["p1", "p2"] and [d.embedding for d in docs] == [[1.0], [2.0]]
    import hashlib
    assert docs[0].text_hash == hashlib.sha256(b"alpha").hexdigest()[:16] and len(docs[0].text_hash) == 16
    assert mmo.build_index([], "m", embed_fn=lambda *a, **k: pytest.fail("no docs -> no model call")) == []


def embed_args(tmp_path, memory_dir, **kw):
    ns = dict(memory_dir=str(memory_dir), model="m-test", max_seq_length=64, batch_size=2,
              index=str(tmp_path / ".memopt" / "index.json"))
    ns.update(kw)
    return types.SimpleNamespace(**ns)


def test_cmd_embed_writes_index_and_reports(tmp_path, fake_st, capsys):
    args = embed_args(tmp_path, make_memory(tmp_path) / "**")
    assert mmo.cmd_embed(args) == 0
    out = capsys.readouterr().out
    assert "embedding 3 memory files with m-test (max_seq_length=64, batch_size=2) ..." in out
    assert f"wrote {args.index} (3 docs)" in out
    data = json.loads(Path(args.index).read_text())
    assert data["model"] == "m-test" and [Path(d["path"]).name for d in data["docs"]] == ["a.md", "b.md", "c.md"]
    assert data["docs"][0]["embedding"] == [2.0, 1.0] and set(data["docs"][0]) == {"path", "text", "text_hash", "embedding"}
    assert fake_st["encode"][0]["batch"] == 2 and fake_st["encode"][0]["max_seq"] == 64


def test_cmd_embed_without_files_is_an_error(tmp_path, fake_st, capsys):
    assert mmo.cmd_embed(embed_args(tmp_path, tmp_path / "nothing" / "**")) == 1
    assert "no .md files found" in capsys.readouterr().err and fake_st["st_init"] == []


# ---- index I/O and vector math -----------------------------------------------------------------------------

def test_load_index_roundtrip_and_default_model(tmp_path):
    d = {"path": "p", "text": "t", "text_hash": "h", "embedding": [1.0, 0.0]}
    model, docs = mmo.load_index(write_json(tmp_path / "i.json", {"model": "mm", "docs": [d]}))
    assert model == "mm" and docs == [mmo.MemoryDoc(**d)]
    model, docs = mmo.load_index(write_json(tmp_path / "j.json", {"docs": []}))
    assert model == mmo.DEFAULT_EMBED_MODEL and docs == []


@pytest.mark.parametrize("a,b,expected", [
    ([1, 0], [1, 0], 1.0), ([1, 0], [0, 1], 0.0), ([1, 0], [-1, 0], -1.0), ([0, 0], [1, 1], 0.0),
    ([1, 1], [0, 0], 0.0), ([3, 4], [6, 8], 1.0), ([1, 1], [1, 0], math.sqrt(0.5)),
])
def test_cosine(a, b, expected):
    assert mmo.cosine(a, b) == pytest.approx(expected)


def test_tokenize_keeps_alnum_and_longer_tokens_lowercased():
    assert mmo._tokenize("Hello, World a-b -- x 42 ok!") == ["hello,", "world", "a-b", "x", "42", "ok!"]
    assert mmo._tokenize("") == []


# ---- hybrid search / rerank ------------------------------------------------------------------------------------

def test_hybrid_search_fuses_ranks_with_rrf_k_60():
    docs = [doc("both", "apple pie", [1, 0]), doc("bm25-only", "apple apple tart", [0, 1]), doc("cos-only", "zzz yyy", [1, 0.1])]
    # query mentions apple; its embedding points along [1, 0] so `both` and `cos-only` rank high by cosine
    res = mmo.hybrid_search("apple", docs, [1.0, 0.0], k=3)
    scores = {d.path: s for d, s in res}
    assert res[0][0].path == "both"
    ranks = {"both": (1, 0), "bm25-only": (0, 2), "cos-only": (2, 1)}                 # (bm25 rank, cosine rank)
    for path, (br, cr) in ranks.items():
        assert scores[path] == pytest.approx(1 / (60 + br) + 1 / (60 + cr))
    assert [d.path for d, _ in res] == sorted(scores, key=lambda p: -scores[p])
    assert len(mmo.hybrid_search("apple", docs, [1.0, 0.0], k=2)) == 2
    assert len(mmo.hybrid_search("apple", docs, [1.0, 0.0])) == 3


def test_hybrid_search_without_bm25_library_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    with pytest.raises(RuntimeError, match="rank-bm25"):
        mmo.hybrid_search("q", [doc("p", "t", [1])], [1.0])


def test_rerank_orders_by_model_score_and_forwards_arguments():
    cands = [(doc("a", "short", [1]), 0.9), (doc("b", "a much longer text", [1]), 0.1)]
    seen = {}

    def predict(pairs, model, max_length, batch_size):
        seen.update(pairs=pairs, model=model, max_length=max_length, batch_size=batch_size)
        return [0.2, 0.8]

    out = mmo.rerank("q", cands, "rr", max_length=50, batch_size=4, predict_fn=predict)
    assert [d.path for d, _ in out] == ["b", "a"] and [s for _, s in out] == [0.8, 0.2]
    assert seen == {"pairs": [["q", "short"], ["q", "a much longer text"]], "model": "rr", "max_length": 50, "batch_size": 4}


def test_predict_rerank_scores_builds_cross_encoder_with_length_cap(fake_st):
    scores = mmo._predict_rerank_scores([["q", "abc"], ["q", "de"]], "ce-model", max_length=77, batch_size=5)
    assert scores == [3.0, 2.0] and fake_st["ce_init"] == [("ce-model", 77)] and fake_st["predict"][0][1] == 5


# ---- cmd_search ---------------------------------------------------------------------------------------------------

def make_index(tmp_path, docs, model="m-idx"):
    return write_json(tmp_path / "idx.json", {"model": model, "docs": [vars(d) for d in docs]})


def search_args(index, **kw):
    ns = dict(query="apple", index=str(index), k=8, rerank=False, rerank_model="rr", max_seq_length=64, batch_size=2)
    ns.update(kw)
    return types.SimpleNamespace(**ns)


def test_cmd_search_missing_index(tmp_path, capsys):
    assert mmo.cmd_search(search_args(tmp_path / "nope.json")) == 1
    assert "no index at" in capsys.readouterr().err


def test_cmd_search_prints_scores_paths_and_trimmed_snippets(tmp_path, monkeypatch, capsys):
    long_text = "apple " + "word " * 100
    idx = make_index(tmp_path, [doc("p1", "apple\npie", [1, 0]), doc("p2", long_text, [0, 1])])
    seen = {}
    monkeypatch.setattr(mmo, "_embed_texts", lambda texts, model: seen.update(texts=texts, model=model) or [[1.0, 0.0]])
    assert mmo.cmd_search(search_args(idx, k=2)) == 0
    assert seen == {"texts": ["apple"], "model": "m-idx"}                        # query embedded with the INDEX's model
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith("  p1") and out[0].split()[0].replace(".", "").isdigit()
    assert out[1] == "         apple pie"
    assert len(out[3].strip()) == 140 and out[3].strip().startswith("apple word")


def test_cmd_search_k_limits_results_and_rerank_reorders(tmp_path, monkeypatch, fake_st, capsys):
    idx = make_index(tmp_path, [doc("p1", "apple", [1, 0]), doc("p2", "apple apple apple pie", [0, 1]), doc("p3", "x", [0, 1])])
    monkeypatch.setattr(mmo, "_embed_texts", lambda texts, model: [[1.0, 0.0]])
    assert mmo.cmd_search(search_args(idx, k=2, rerank=True)) == 0
    out = capsys.readouterr().out
    assert fake_st["ce_init"] == [("rr", 64)] and fake_st["predict"][0][1] == 2
    import re
    paths = [m.group(1) for m in (re.match(r"^\s*[-\d.]+  (\S+)$", ln) for ln in out.splitlines()) if m]
    assert paths == ["p2", "p1"]                                                  # the longer doc scores higher in the fake CE; k=2 kept two


# ---- dedup -------------------------------------------------------------------------------------------------------

def test_find_duplicate_groups_threshold_is_inclusive_and_groups_are_transitive():
    a, b, c, d = doc("a", "", [1, 0]), doc("b", "", [1, 0]), doc("c", "", [0.8, 0.6]), doc("d", "", [0, 1])
    assert mmo.find_duplicate_groups([a, b, c, d], threshold=1.0) == [["a", "b"]]            # cos == threshold counts
    groups = mmo.find_duplicate_groups([a, b, c, d], threshold=0.8)
    assert groups == [["a", "b", "c"]]                                                      # c joins via cosine 0.8 exactly
    assert mmo.find_duplicate_groups([a, d], threshold=0.5) == []
    chain = [doc("x", "", [1, 0]), doc("y", "", [math.cos(0.5), math.sin(0.5)]), doc("z", "", [math.cos(1.0), math.sin(1.0)])]
    assert mmo.find_duplicate_groups(chain, threshold=0.85) == [["x", "y", "z"]]             # x-z alone would be below 0.85
    assert mmo.cosine(chain[0].embedding, chain[2].embedding) < 0.85


def test_find_duplicate_groups_same_hash_groups_regardless_of_embedding_and_sorts_by_size():
    docs = [doc("a", "", [0, 1], h="same"), doc("b", "", [0, -1], h="same"), doc("c", "", [1, 0]), doc("d", "", [1, 0]),
            doc("e", "", [1, 0]), doc("f", "", [-1, 0])]
    assert mmo.cosine([0, 1], [0, -1]) == -1.0                                              # a and b are opposites ...
    groups = mmo.find_duplicate_groups(docs, threshold=0.99)
    assert groups == [["c", "d", "e"], ["a", "b"]]                                          # ... yet grouped by identical text hash, biggest first
    assert mmo.find_duplicate_groups([], 0.85) == [] and mmo.find_duplicate_groups([doc("a", "", [1])], 0.1) == []


def test_find_duplicate_groups_default_threshold_is_point_eight_five():
    close = [doc("a", "", [1, 0]), doc("b", "", [math.cos(math.acos(0.86)), math.sin(math.acos(0.86))])]
    far = [doc("a", "", [1, 0]), doc("b", "", [math.cos(math.acos(0.84)), math.sin(math.acos(0.84))])]
    assert len(mmo.find_duplicate_groups(close)) == 1 and mmo.find_duplicate_groups(far) == []


def test_group_pairwise_similarities_sorted_descending():
    by = {p: doc(p, "", e) for p, e in {"a": [1, 0], "b": [1, 0], "c": [0, 1]}.items()}
    pairs = mmo.group_pairwise_similarities(by, ["a", "b", "c"])
    assert [(x, y) for x, y, _ in pairs][0] == ("a", "b") and pairs[0][2] == pytest.approx(1.0)
    assert len(pairs) == 3 and [round(s, 6) for _, _, s in pairs] == [1.0, 0.0, 0.0]
    assert mmo.group_pairwise_similarities(by, ["a"]) == []


@pytest.mark.parametrize("spec,expected", [
    ("0.80:0.95:0.05", [0.8, 0.85, 0.9, 0.95]), ("0.9:0.9:0.1", [0.9]), ("0.5:0.6:0.5", [0.5]), ("0.9:0.8:0.1", []),
])
def test_parse_sweep_arg(spec, expected):
    assert mmo._parse_sweep_arg(spec) == expected


@pytest.mark.parametrize("spec", ["0.8:0.9", "a:b:c", "", "0.8:0.9:0.1:0.2"])
def test_parse_sweep_arg_bad_format(spec):
    with pytest.raises(ValueError, match="START:STOP:STEP"):
        mmo._parse_sweep_arg(spec)


@pytest.mark.parametrize("step", ["0", "-0.1"])
def test_parse_sweep_arg_non_positive_step(step):
    with pytest.raises(ValueError, match="STEP must be positive"):
        mmo._parse_sweep_arg(f"0.8:0.9:{step}")


def dedup_args(index, **kw):
    ns = dict(index=str(index), threshold=0.85, sweep=None, out=None)
    ns.update(kw)
    return types.SimpleNamespace(**ns)


DEDUP_DOCS = [doc("a", "t", [1, 0]), doc("b", "t", [0.9, math.sqrt(1 - 0.81)]), doc("c", "t", [0, 1])]


def test_cmd_dedup_missing_index(tmp_path, capsys):
    assert mmo.cmd_dedup(dedup_args(tmp_path / "nope.json")) == 1
    assert "no index at" in capsys.readouterr().err


def test_cmd_dedup_reports_groups_and_writes_out(tmp_path, capsys):
    idx = make_index(tmp_path, DEDUP_DOCS)
    out = tmp_path / "o" / "dedup.json"
    assert mmo.cmd_dedup(dedup_args(idx, threshold=0.85, out=str(out))) == 0
    text = capsys.readouterr().out
    assert "1 merge-candidate group(s) (threshold 0.85):" in text and "  group of 2:" in text and "    - a" in text and "    - b" in text
    assert json.loads(out.read_text()) == {"threshold": 0.85, "groups": [["a", "b"]]}
    assert f"wrote {out}" in text


def test_cmd_dedup_no_groups_message_and_no_file(tmp_path, capsys):
    idx = make_index(tmp_path, DEDUP_DOCS)
    out = tmp_path / "dedup.json"
    assert mmo.cmd_dedup(dedup_args(idx, threshold=0.95, out=str(out))) == 0
    assert "no groups found above similarity 0.95 -- nothing to merge" in capsys.readouterr().out
    assert not out.exists()


def test_cmd_dedup_sweep_lists_each_threshold_and_flags_fragile_pairs(tmp_path, capsys):
    idx = make_index(tmp_path, DEDUP_DOCS)
    out = tmp_path / "sweep.json"
    assert mmo.cmd_dedup(dedup_args(idx, sweep="0.80:0.95:0.05", out=str(out))) == 0
    text = capsys.readouterr().out
    assert "sweeping 4 threshold(s) [0.800..0.950]:" in text
    assert "  0.800: 1 group(s), sizes=[2]" in text and "  0.900: 1 group(s), sizes=[2]" in text
    assert "  0.950: 0 group(s)" in text and "sizes=[]" not in text
    assert "0.9000  a" in text
    data = json.loads(out.read_text())
    assert data["thresholds"] == [0.8, 0.85, 0.9, 0.95]
    assert data["groups_by_threshold"]["0.8000"] == [["a", "b"]] and data["groups_by_threshold"]["0.9500"] == []


def test_cmd_dedup_sweep_marks_pairs_below_the_strictest_threshold(tmp_path, capsys):
    idx = make_index(tmp_path, DEDUP_DOCS)                                                # cos(a, b) = 0.9
    mmo.cmd_dedup(dedup_args(idx, sweep="0.80:0.95:0.05"))
    text = capsys.readouterr().out
    assert "0.9000  a\n             b  <-- below strictest threshold tested" in text
    mmo.cmd_dedup(dedup_args(idx, sweep="0.80:0.85:0.05"))                                # 0.9 clears a 0.85 strictest end
    text = capsys.readouterr().out
    assert "0.9000  a\n             b\n" in text and "<-- below" not in text


def test_cmd_dedup_sweep_bad_spec(tmp_path, capsys):
    idx = make_index(tmp_path, DEDUP_DOCS)
    assert mmo.cmd_dedup(dedup_args(idx, sweep="nonsense")) == 1
    assert "--sweep must be START:STOP:STEP" in capsys.readouterr().err


# ---- report ---------------------------------------------------------------------------------------------------------

def report_args(tmp_path, **kw):
    ns = dict(index=None, clusters=None, dedup=None, out=str(tmp_path / "map.html"))
    ns.update(kw)
    return types.SimpleNamespace(**ns)


def test_cmd_report_with_no_inputs_uses_placeholders(tmp_path, capsys):
    args = report_args(tmp_path)
    assert mmo.cmd_report(args) == 0
    html = Path(args.out).read_text()
    assert "run `cluster` first" in html and "none found (or `dedup` not run yet)" in html
    assert "<b>0</b><span>indexed files</span>" in html and "<b>0</b><span>merge candidates found</span>" in html
    assert f"wrote {args.out}" in capsys.readouterr().out


def test_cmd_report_fills_counts_clusters_and_groups(tmp_path):
    idx = make_index(tmp_path, DEDUP_DOCS)
    clusters = write_json(tmp_path / "c.json", {"num_clusters": 3, "modularity": 0.4321,
                                                "cluster_sizes": {str(i): i + 1 for i in range(25)}})
    dedup = write_json(tmp_path / "d.json", {"groups": [["x.md", "y.md"], ["p.md", "q.md", "r.md"]]})
    args = report_args(tmp_path, index=str(idx), clusters=str(clusters), dedup=str(dedup))
    mmo.cmd_report(args)
    html = Path(args.out).read_text()
    assert "<b>3</b><span>indexed files</span>" in html and "<b>3</b><span>real communities (was 0)</span>" in html
    assert "<b>0.4321</b><span>modularity</span>" in html and "<b>2</b><span>merge candidates found</span>" in html
    rows = [ln for ln in html.splitlines() if ln.startswith("<tr><td>")]
    assert len(rows) == 20 and rows[0] == "<tr><td>24</td><td>25</td></tr>"                    # largest first, top 20 only
    assert '<div class="group"><b>group of 3</b><br><code>p.md</code><br><code>q.md</code><br><code>r.md</code></div>' in html
    assert "<h2>Merge candidates (2 groups)</h2>" in html
    import datetime
    assert f"generated {datetime.date.today().isoformat()}" in html


def test_cmd_report_empty_cluster_sizes_keep_the_placeholder_row(tmp_path):
    clusters = write_json(tmp_path / "c.json", {"num_clusters": 0, "modularity": 0, "cluster_sizes": {}})
    mmo.cmd_report(report_args(tmp_path, clusters=str(clusters)))
    assert "run `cluster` first" in (tmp_path / "map.html").read_text()


# ---- main --------------------------------------------------------------------------------------------------------------

def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["memory_map_optimizer", *argv])
    return mmo.main()


def test_main_requires_a_subcommand(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch)
    assert exc.value.code == 2


def test_main_cluster_end_to_end_and_defaults(monkeypatch, tmp_path, capsys):
    g = graph_file(tmp_path)
    out = tmp_path / "c.json"
    assert run_main(monkeypatch, "cluster", "--graph", str(g), "--out", str(out), "--seed", "3", "--resolution", "1.5") == 0
    assert json.loads(out.read_text())["num_nodes"] == 6
    seen = {}
    monkeypatch.setattr(mmo, "cmd_cluster", lambda a: seen.update(vars(a)) or 0)
    run_main(monkeypatch, "cluster")
    assert seen["graph"] == "~/.graphify/global-graph.json" and seen["out"] == mmo.DEFAULT_CLUSTERS_PATH
    assert (seen["nodes_key"], seen["edges_key"], seen["source_key"], seen["target_key"], seen["id_key"]) == (
        "nodes", "edges", "source", "target", "id")
    assert (seen["resolution"], seen["seed"], seen["write_back"]) == (1.0, 0, False)


def test_main_embed_search_dedup_report_defaults(monkeypatch):
    got = {}
    for name in ("embed", "search", "dedup", "report"):
        monkeypatch.setattr(mmo, f"cmd_{name}", lambda a, n=name: got.update({n: vars(a)}) or 0)
    run_main(monkeypatch, "embed")
    run_main(monkeypatch, "search", "my query")
    run_main(monkeypatch, "dedup")
    run_main(monkeypatch, "report")
    assert got["embed"]["memory_dir"] == "~/.claude/projects/*/memory/**" and got["embed"]["model"] == mmo.DEFAULT_EMBED_MODEL
    assert (got["embed"]["max_seq_length"], got["embed"]["batch_size"], got["embed"]["index"]) == (4096, 8, ".memopt/index.json")
    assert got["search"]["query"] == "my query" and got["search"]["k"] == 8 and got["search"]["rerank"] is False
    assert got["search"]["rerank_model"] == mmo.DEFAULT_RERANK_MODEL
    assert got["dedup"]["threshold"] == 0.85 and got["dedup"]["sweep"] is None and got["dedup"]["out"] == ".memopt/dedup.json"
    assert got["report"]["out"] == "memory_map_v2.html" and got["report"]["clusters"] == ".memopt/clusters.json"


def test_main_returns_the_command_exit_code(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, "dedup", "--index", str(tmp_path / "missing.json")) == 1
