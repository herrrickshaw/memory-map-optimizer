"""Network-free CLI and pure-function tests (embeddings are hand-built vectors)."""
from __future__ import annotations

import json

import pytest

import memory_map_optimizer as mmo

VECS = [[1, 0, 0], [0.99, 0.1, 0], [0, 1, 0]]


def _write_index(path, vecs=VECS):
    docs = [{"path": f"/m/f{i}.md", "text": f"doc {i} token memory", "text_hash": f"h{i}", "embedding": v}
            for i, v in enumerate(vecs)]
    path.write_text(json.dumps({"model": "stub", "docs": docs}))


def _write_graph(path):
    nodes = [{"id": f"n{i}"} for i in range(6)]
    edges = [{"source": a, "target": b} for a, b in
             [("n0", "n1"), ("n1", "n2"), ("n0", "n2"), ("n3", "n4"), ("n4", "n5"), ("n3", "n5"), ("n5", "ghost")]]
    path.write_text(json.dumps({"nodes": nodes, "edges": edges}))


def test_cluster_cli_writes_clusters_and_write_back(tmp_path, capsys):
    pytest.importorskip("leidenalg")
    g, out = tmp_path / "g.json", tmp_path / "out" / "clusters.json"
    _write_graph(g)
    assert mmo.main(["cluster", "--graph", str(g), "--out", str(out), "--write-back"]) == 0
    assert "had no matching node entry" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["num_nodes"] == 7 and result["num_clusters"] >= 2
    assert all("community" in n for n in json.loads(g.read_text())["nodes"])


def test_cluster_cli_missing_graph_returns_1(tmp_path, capsys):
    assert mmo.main(["cluster", "--graph", str(tmp_path / "nope.json"), "--out", str(tmp_path / "c.json")]) == 1
    assert "graph file not found" in capsys.readouterr().err


def test_dedup_cli_groups_and_writes_json(tmp_path, capsys):
    idx, out = tmp_path / "idx.json", tmp_path / "d.json"
    _write_index(idx)
    assert mmo.main(["dedup", "--index", str(idx), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["groups"] == [["/m/f0.md", "/m/f1.md"]]
    assert "1 merge-candidate group(s)" in capsys.readouterr().out


def test_dedup_cli_sweep_and_bad_spec(tmp_path, capsys):
    idx = tmp_path / "idx.json"
    _write_index(idx)
    out = tmp_path / "sw.json"
    assert mmo.main(["dedup", "--index", str(idx), "--sweep", "0.80:0.90:0.05", "--out", str(out)]) == 0
    assert sorted(json.loads(out.read_text())["groups_by_threshold"]) == ["0.8000", "0.8500", "0.9000"]
    assert mmo.main(["dedup", "--index", str(idx), "--sweep", "bad"]) == 1
    assert "--sweep must be START:STOP:STEP" in capsys.readouterr().err


def test_missing_index_returns_1_for_search_and_dedup(tmp_path, capsys):
    missing = str(tmp_path / "none.json")
    assert mmo.main(["search", "q", "--index", missing]) == 1
    assert mmo.main(["dedup", "--index", missing]) == 1
    assert capsys.readouterr().err.count("run `embed` first") == 2


def test_embed_cli_no_files_returns_1(tmp_path, capsys):
    assert mmo.main(["embed", "--memory-dir", str(tmp_path / "empty" / "*"), "--index", str(tmp_path / "i.json")]) == 1
    assert "no .md files found" in capsys.readouterr().err


def test_search_index_with_injected_embedder_and_reranker():
    docs = [mmo.MemoryDoc(f"/m/{i}.md", t, str(i), v) for i, (t, v) in enumerate([
        ("alpha beta", [1.0, 0.0]), ("gamma delta", [0.0, 1.0])])]
    hits = mmo.search_index("alpha", "stub", docs, k=2, embed_fn=lambda texts, m, **kw: [[1.0, 0.0]])
    assert hits[0][0].path == "/m/0.md"
    reranked = mmo.search_index("alpha", "stub", docs, k=2, rerank_model="rr",
                                embed_fn=lambda texts, m, **kw: [[1.0, 0.0]],
                                predict_fn=lambda pairs, m, **kw: [0.1, 0.9])
    assert reranked[0][0].path == "/m/1.md"


def test_format_result_truncates_snippet():
    line = mmo.format_result(mmo.MemoryDoc("/p.md", "x\n" * 200, "h"), 0.5)
    assert line.startswith(" 0.5000  /p.md\n")
    assert len(line.split("\n")[1]) == 9 + mmo.SNIPPET_CHARS


def test_render_report_placeholders_and_data():
    empty = mmo.render_report(0, 0, 0, {}, [], "2026-01-01")
    assert "run `cluster` first" in empty and "none found" in empty and "2026-01-01" in empty
    full = mmo.render_report(3, 2, 0.5, {"0": 5, "1": 9}, [["/a.md", "/b.md"]], "d")
    assert full.index("<td>1</td><td>9</td>") < full.index("<td>0</td><td>5</td>")
    assert "group of 2" in full and "<code>/a.md</code>" in full


def test_report_cli_roundtrip(tmp_path):
    idx, out = tmp_path / "idx.json", tmp_path / "r.html"
    _write_index(idx)
    assert mmo.main(["report", "--index", str(idx), "--clusters", str(tmp_path / "x.json"),
                     "--dedup", str(tmp_path / "y.json"), "--out", str(out)]) == 0
    assert "<b>3</b><span>indexed files</span>" in out.read_text()


def test_read_memory_files_dir_and_glob(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("A")
    (tmp_path / "sub" / "b.md").write_text("B")
    (tmp_path / "c.txt").write_text("C")
    by_dir = mmo._read_memory_files(str(tmp_path))
    by_glob = mmo._read_memory_files(str(tmp_path / "**"))
    assert [t for _, t in by_dir] == ["A", "B"] and [t for _, t in by_glob] == ["A", "B"]
