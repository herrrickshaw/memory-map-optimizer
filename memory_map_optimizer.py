#!/usr/bin/env python3
"""
memory_map_optimizer.py
========================
Standalone optimizer for a Claude Code memory system shaped like the one
audited in "Memory Map -- Tiers, Clusters & Resource Routing" (2026-08-12):

  - graphify: a knowledge graph over memory/code (~/.graphify/global-graph.json),
    whose community detection is broken (advertised, but returns 0 communities).
  - qmd: BM25 + semantic search over the memory markdown files, embedding
    model unspecified in the audit -- likely dated.
  - Tier-2 memory: typed markdown files (user/feedback/project/reference)
    under ~/.claude/projects/<project>/memory/.

This does NOT reimplement graphify or qmd. It fixes the two concrete gaps
the audit surfaced, using the Hugging Face models discussed alongside it:

  1. Real community detection. graphify's is broken -- this runs actual
     Leiden clustering (python-igraph + leidenalg) against the existing
     graph JSON and writes real cluster assignments back out.
  2. A modern embedding model for qmd's semantic half. Default is
     Qwen/Qwen3-Embedding-0.6B (Apache-2.0, MTEB 70.7, ~1.5GB); pass
     --model BAAI/bge-m3 for the hybrid dense+sparse+multi-vector model
     instead. Optional --rerank second stage via Qwen/Qwen3-Reranker-0.6B.

It also adds one capability neither existing tool has: NEAR-DUPLICATE
DETECTION. The audit found "4 sequential token-optimizer snapshots" that a
human had to notice and merge by hand -- `dedup` finds those automatically
via embedding cosine similarity, before another 100+ files pile up unsorted.

Nothing here talks to graphify's MCP port or qmd's cron job directly -- it
reads/writes the same files those tools already use (graph JSON, memory
markdown), so it's a drop-in augmentation, not a replacement or a rewrite.

RUN THIS ON THE MACHINE THAT HAS THE REAL FILES. It was built and tested in
a sandbox with no access to ~/.graphify or ~/.claude/projects/*/memory, and
with huggingface.co itself blocked by that sandbox's network egress -- so
graph clustering was verified against synthetic graphs (pure algorithm, no
network needed) and the indexing/dedup/search logic was verified with
mocked embedding vectors. The actual embed() HF model call is standard
sentence-transformers usage and hasn't been run against a live download in
this environment. Point --graph / --memory-dir at your real paths and run
`embed` first before trusting `search`/`dedup` output.

Install (macOS Homebrew Python blocks global pip -- use a venv):
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt

Usage (with the venv active):
  python memory_map_optimizer.py cluster --graph ~/.graphify/global-graph.json
  python memory_map_optimizer.py embed --memory-dir "~/.claude/projects/*/memory"
  python memory_map_optimizer.py search "token optimizer routing" --index .memopt/index.json
  python memory_map_optimizer.py dedup --index .memopt/index.json --threshold 0.85
  python memory_map_optimizer.py dedup --sweep 0.80:0.95:0.05   # see pairwise sim, not just in/out
  python memory_map_optimizer.py report --index .memopt/index.json --clusters .memopt/clusters.json --out memory_map_v2.html
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INDEX_PATH = ".memopt/index.json"
DEFAULT_CLUSTERS_PATH = ".memopt/clusters.json"


# ══════════════════════════════════════════════════════════════════════════
# cluster -- real Leiden community detection for graphify's graph JSON
# ══════════════════════════════════════════════════════════════════════════

def _load_graph_json(path: Path, nodes_key: str, edges_key: str,
                      source_key: str, target_key: str, id_key: str
                      ) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """
    Tolerant loader for graphify-style graph JSON. Two shapes handled:
      {"nodes": [{"id": "..."}], "edges": [{"source": "...", "target": "..."}]}
      {"edges": [["a", "b"], ...]}                          (nodes implied)
    Override key names via CLI flags if your graph JSON differs.

    Real graphify data (2026-08) has edges referencing node ids that never
    appear in the declared "nodes" list -- e.g. an edge endpoint like
    "Bazartalks_Py2Cplus::json" with no matching node entry, presumably a
    file/dependency that got pruned from the node list without its edges
    being pruned too. Rather than crash building the igraph object (a bare
    KeyError with no indication of WHY), any such dangling endpoint is
    added to the node set automatically -- Leiden treats it like any other
    node, just one with no declared metadata. Returns the third element as
    the list of endpoints that had to be added this way, so callers can
    report how much of the graph's declared structure doesn't match its
    actual edges.
    """
    data = json.loads(path.read_text())

    raw_edges = data.get(edges_key, [])
    edges: list[tuple[str, str]] = []
    for e in raw_edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            edges.append((str(e[0]), str(e[1])))
        elif isinstance(e, dict):
            edges.append((str(e[source_key]), str(e[target_key])))
        else:
            raise ValueError(f"unrecognized edge shape: {e!r}")

    raw_nodes = data.get(nodes_key)
    if raw_nodes is not None:
        nodes = [str(n[id_key]) if isinstance(n, dict) else str(n) for n in raw_nodes]
    else:
        nodes = []

    declared = set(nodes)
    dangling: list[str] = []
    for a, b in edges:
        for endpoint in (a, b):
            if endpoint not in declared:
                declared.add(endpoint)
                nodes.append(endpoint)
                dangling.append(endpoint)

    return nodes, edges, dangling


@dataclass
class ClusterResult:
    node_to_cluster: dict[str, int]
    cluster_sizes: dict[int, int]
    modularity: float
    num_nodes: int
    num_edges: int

    def to_json(self) -> dict:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_clusters": len(self.cluster_sizes),
            "modularity": round(self.modularity, 4),
            "cluster_sizes": {str(k): v for k, v in sorted(self.cluster_sizes.items(), key=lambda kv: -kv[1])},
            "node_to_cluster": self.node_to_cluster,
        }


def run_leiden(nodes: list[str], edges: list[tuple[str, str]],
               resolution: float = 1.0, seed: int = 0) -> ClusterResult:
    """
    Leiden community detection (Traag, Waltman & van Eck 2019) -- strictly
    better than Louvain at avoiding disconnected communities, and the
    standard replacement when a graph tool's "community detection" has gone
    stale or silently broken (graphify's currently returns 0).
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:
        raise RuntimeError(
            "cluster needs python-igraph + leidenalg -- see README Install (use a venv on macOS)"
        ) from e

    index = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(index[a], index[b]) for a, b in edges])

    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )

    node_to_cluster = {n: partition.membership[index[n]] for n in nodes}
    sizes: dict[int, int] = {}
    for c in node_to_cluster.values():
        sizes[c] = sizes.get(c, 0) + 1

    return ClusterResult(
        node_to_cluster=node_to_cluster,
        cluster_sizes=sizes,
        modularity=partition.modularity,
        num_nodes=len(nodes),
        num_edges=len(edges),
    )


def cmd_cluster(args: argparse.Namespace) -> int:
    graph_path = Path(args.graph).expanduser()
    if not graph_path.exists():
        print(f"error: graph file not found: {graph_path}", file=sys.stderr)
        return 1

    nodes, edges, dangling = _load_graph_json(
        graph_path, args.nodes_key, args.edges_key, args.source_key, args.target_key, args.id_key,
    )
    if not nodes:
        print("error: no nodes found -- check --nodes-key/--edges-key against your graph JSON", file=sys.stderr)
        return 1
    if dangling:
        print(f"note: {len(dangling)} edge endpoint(s) had no matching node entry "
              f"(e.g. {dangling[0]!r}) -- added them as nodes so clustering doesn't crash on them")

    result = run_leiden(nodes, edges, resolution=args.resolution, seed=args.seed)

    print(f"{result.num_nodes} nodes, {result.num_edges} edges "
          f"-> {len(result.cluster_sizes)} communities (modularity {result.modularity:.4f})")
    top = sorted(result.cluster_sizes.items(), key=lambda kv: -kv[1])[:10]
    for cid, size in top:
        print(f"  cluster {cid}: {size} nodes")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_json(), indent=2))
    print(f"\nwrote {out_path}")

    if args.write_back:
        data = json.loads(graph_path.read_text())
        by_id = {str(n[args.id_key]) if isinstance(n, dict) else str(n): n for n in data.get(args.nodes_key, [])}
        for node_id, cluster in result.node_to_cluster.items():
            n = by_id.get(node_id)
            if isinstance(n, dict):
                n["community"] = cluster
        graph_path.write_text(json.dumps(data, indent=2))
        print(f"wrote community assignments back into {graph_path}")

    return 0


# ══════════════════════════════════════════════════════════════════════════
# embed -- index memory markdown files with a modern small embedding model
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryDoc:
    path: str
    text: str
    text_hash: str
    embedding: list[float] = field(default_factory=list)


def _read_memory_files(memory_dir_glob: str) -> list[tuple[str, str]]:
    paths = sorted(set(glob.glob(str(Path(memory_dir_glob).expanduser()), recursive=True)))
    md_paths = [p for p in paths if p.endswith(".md")]
    if not md_paths and Path(memory_dir_glob).expanduser().is_dir():
        md_paths = sorted(str(p) for p in Path(memory_dir_glob).expanduser().rglob("*.md"))
    docs = []
    for p in md_paths:
        try:
            docs.append((p, Path(p).read_text(errors="ignore")))
        except OSError:
            continue
    return docs


DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_EMBED_BATCH_SIZE = 8


def _embed_texts(texts: list[str], model_name: str,
                  max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
                  batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> list[list[float]]:
    """Isolated so the rest of the pipeline is testable without a real
    model download (sentence-transformers pulls weights from HF).

    max_seq_length and batch_size both bound worst-case memory, not just
    speed -- hit for real on a live memory directory: one unusually long
    memory file, uncapped, made Qwen3-Embedding-0.6B's attention-mask
    construction try to allocate 17.23 GiB and crash the whole `embed` run
    over a single outlier document. A batch pads every doc in it to the
    longest one, so keeping batch_size small bounds that blowup too,
    independent of max_seq_length.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    if model.max_seq_length is None or model.max_seq_length > max_seq_length:
        model.max_seq_length = max_seq_length
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                            show_progress_bar=len(texts) > 20)
    return [v.tolist() for v in vectors]


def build_index(docs: list[tuple[str, str]], model_name: str,
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
                 batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
                 embed_fn=_embed_texts) -> list[MemoryDoc]:
    if not docs:
        return []
    texts = [t for _, t in docs]
    vectors = embed_fn(texts, model_name, max_seq_length=max_seq_length, batch_size=batch_size)
    return [
        MemoryDoc(path=p, text=t, text_hash=hashlib.sha256(t.encode()).hexdigest()[:16], embedding=v)
        for (p, t), v in zip(docs, vectors)
    ]


def cmd_embed(args: argparse.Namespace) -> int:
    docs = _read_memory_files(args.memory_dir)
    if not docs:
        print(f"error: no .md files found under {args.memory_dir}", file=sys.stderr)
        return 1

    print(f"embedding {len(docs)} memory files with {args.model} "
          f"(max_seq_length={args.max_seq_length}, batch_size={args.batch_size}) ...")
    indexed = build_index(docs, args.model, max_seq_length=args.max_seq_length, batch_size=args.batch_size)

    out_path = Path(args.index).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model,
        "docs": [{"path": d.path, "text": d.text, "text_hash": d.text_hash, "embedding": d.embedding}
                 for d in indexed],
    }))
    print(f"wrote {out_path} ({len(indexed)} docs)")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# shared: index I/O + vector math (pure, no network -- fully unit-testable)
# ══════════════════════════════════════════════════════════════════════════

def load_index(path: Path) -> tuple[str, list[MemoryDoc]]:
    data = json.loads(path.read_text())
    docs = [MemoryDoc(**d) for d in data["docs"]]
    return data.get("model", DEFAULT_EMBED_MODEL), docs


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _tokenize(text: str) -> list[str]:
    return [w for w in text.lower().split() if w.isalnum() or len(w) > 2]


# ══════════════════════════════════════════════════════════════════════════
# search -- hybrid BM25 + embedding, optional cross-encoder rerank
# ══════════════════════════════════════════════════════════════════════════

def hybrid_search(query: str, docs: list[MemoryDoc], query_embedding: list[float],
                   k: int = 8) -> list[tuple[MemoryDoc, float]]:
    """
    Reciprocal-rank fusion of BM25 and embedding cosine similarity -- avoids
    having to tune a weight between two differently-scaled scores, which is
    the usual failure mode of naive hybrid search.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise RuntimeError("search needs rank-bm25 -- see README Install (use a venv on macOS)") from e

    corpus = [_tokenize(d.text) for d in docs]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_rank = {i: r for r, i in enumerate(sorted(range(len(docs)), key=lambda i: -bm25_scores[i]))}

    cos_scores = [cosine(query_embedding, d.embedding) for d in docs]
    cos_rank = {i: r for r, i in enumerate(sorted(range(len(docs)), key=lambda i: -cos_scores[i]))}

    RRF_K = 60
    fused = [
        (docs[i], 1.0 / (RRF_K + bm25_rank[i]) + 1.0 / (RRF_K + cos_rank[i]))
        for i in range(len(docs))
    ]
    fused.sort(key=lambda x: -x[1])
    return fused[:k]


def _predict_rerank_scores(pairs: list[list[str]], model_name: str,
                            max_length: int = DEFAULT_MAX_SEQ_LENGTH,
                            batch_size: int = DEFAULT_EMBED_BATCH_SIZE) -> list[float]:
    """Isolated so rerank()'s ordering logic is testable without a real
    model download, same pattern as _embed_texts/build_index.

    max_length/batch_size exist for the same reason as _embed_texts's: a
    CrossEncoder concatenates [query, doc] and runs full attention over
    that combined sequence, so an uncapped long doc blows up here too --
    hit for real as an MPS (Apple GPU) out-of-memory crash rather than the
    CPU one _embed_texts hit ("MPS backend out of memory... Tried to
    allocate 237.33 MiB"), same root cause: no length cap.
    """
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, max_length=max_length)
    return list(ce.predict(pairs, batch_size=batch_size))


def rerank(query: str, candidates: list[tuple[MemoryDoc, float]], model_name: str,
           max_length: int = DEFAULT_MAX_SEQ_LENGTH,
           batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
           predict_fn=_predict_rerank_scores) -> list[tuple[MemoryDoc, float]]:
    pairs = [[query, d.text] for d, _ in candidates]
    scores = predict_fn(pairs, model_name, max_length=max_length, batch_size=batch_size)
    reranked = sorted(zip([d for d, _ in candidates], scores), key=lambda x: -x[1])
    return reranked


def cmd_search(args: argparse.Namespace) -> int:
    index_path = Path(args.index).expanduser()
    if not index_path.exists():
        print(f"error: no index at {index_path} -- run `embed` first", file=sys.stderr)
        return 1
    model_name, docs = load_index(index_path)

    query_vec = _embed_texts([args.query], model_name)[0]
    results = hybrid_search(args.query, docs, query_vec, k=args.k)

    if args.rerank:
        results = rerank(args.query, results, args.rerank_model,
                          max_length=args.max_seq_length, batch_size=args.batch_size)

    for doc, score in results:
        snippet = doc.text.strip().replace("\n", " ")[:140]
        print(f"{score:>7.4f}  {doc.path}\n         {snippet}")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# dedup -- near-duplicate / mergeable memory file detection
# ══════════════════════════════════════════════════════════════════════════

def find_duplicate_groups(docs: list[MemoryDoc], threshold: float = 0.85) -> list[list[str]]:
    """
    Groups files whose embeddings are mutually similar above *threshold*
    into merge candidates -- connected components over the similarity graph,
    so e.g. 4 sequential snapshots of the same topic surface as ONE group
    of 4, not 6 separate pairwise hits (matches the "4 -> 1" merge the
    original audit did by hand).

    Default was 0.92 originally; lowered to 0.85 after a real run against
    111 actual memory files found 0 groups at 0.92 -- too strict to be
    useful as a default, even though some of those files are clearly
    related (e.g. the audit's own "4 sequential token-optimizer snapshots"
    case). 0.85 is still well above "vaguely similar topic" territory for
    normalized embeddings; --threshold remains available to tune either
    direction per directory.
    """
    n = len(docs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if docs[i].text_hash == docs[j].text_hash or cosine(docs[i].embedding, docs[j].embedding) >= threshold:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(docs[i].path)

    return sorted([g for g in groups.values() if len(g) > 1], key=lambda g: -len(g))


def group_pairwise_similarities(docs_by_path: dict[str, "MemoryDoc"],
                                 group_paths: list[str]) -> list[tuple[str, str, float]]:
    """
    All pairwise cosine similarities among the docs in one merge-candidate
    group, descending. A group with a low pair inside it is held together
    only by a TRANSITIVE chain (A-B and B-C both cleared the threshold, so
    the union-find in find_duplicate_groups puts A and C in one group even
    if cosine(A, C) is much lower) -- that's the fragile, least trustworthy
    kind of match, and this is the only way to see it: the group listing
    alone just says "in" or "out", it doesn't say how.
    """
    pairs = []
    for i in range(len(group_paths)):
        for j in range(i + 1, len(group_paths)):
            a, b = group_paths[i], group_paths[j]
            sim = cosine(docs_by_path[a].embedding, docs_by_path[b].embedding)
            pairs.append((a, b, sim))
    return sorted(pairs, key=lambda x: -x[2])


def _parse_sweep_arg(spec: str) -> list[float]:
    """Parses "START:STOP:STEP" (e.g. "0.80:0.95:0.05") into a threshold
    list, ascending, rounded to 4dp to avoid float-accumulation drift."""
    try:
        lo, hi, step = (float(x) for x in spec.split(":"))
    except ValueError as e:
        raise ValueError(f'--sweep must be START:STOP:STEP, e.g. "0.80:0.95:0.05" (got {spec!r})') from e
    if step <= 0:
        raise ValueError(f"--sweep STEP must be positive (got {step})")
    thresholds = []
    t = lo
    while t <= hi + 1e-9:
        thresholds.append(round(t, 4))
        t += step
    return thresholds


def cmd_dedup(args: argparse.Namespace) -> int:
    index_path = Path(args.index).expanduser()
    if not index_path.exists():
        print(f"error: no index at {index_path} -- run `embed` first", file=sys.stderr)
        return 1
    _, docs = load_index(index_path)

    if args.sweep:
        try:
            thresholds = _parse_sweep_arg(args.sweep)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        print(f"sweeping {len(thresholds)} threshold(s) [{thresholds[0]:.3f}..{thresholds[-1]:.3f}]:\n")
        by_path = {d.path: d for d in docs}
        sweep_results = {}
        for t in thresholds:
            groups = find_duplicate_groups(docs, threshold=t)
            sweep_results[t] = groups
            sizes = f", sizes={[len(g) for g in groups]}" if groups else ""
            print(f"  {t:.3f}: {len(groups)} group(s){sizes}")

        loosest_groups = sweep_results[thresholds[0]]
        if loosest_groups:
            print(f"\nPairwise similarity within each group at the loosest threshold tested "
                  f"({thresholds[0]:.3f}) -- a pair below {thresholds[-1]:.3f} (the strictest "
                  f"threshold tested) means that pair alone wouldn't clear your strict end; "
                  f"the group only holds together via a chain through a third doc:\n")
            for g in loosest_groups:
                print(f"  group of {len(g)}:")
                for a, b, sim in group_pairwise_similarities(by_path, g):
                    flag = "" if sim >= thresholds[-1] else "  <-- below strictest threshold tested"
                    print(f"    {sim:.4f}  {a}\n             {b}{flag}")
                print()

        if args.out:
            out_path = Path(args.out).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"thresholds": thresholds,
                 "groups_by_threshold": {f"{t:.4f}": g for t, g in sweep_results.items()}}, indent=2))
            print(f"wrote {args.out}")
        return 0

    groups = find_duplicate_groups(docs, threshold=args.threshold)
    if not groups:
        print(f"no groups found above similarity {args.threshold} -- nothing to merge")
        return 0

    print(f"{len(groups)} merge-candidate group(s) (threshold {args.threshold}):\n")
    for g in groups:
        print(f"  group of {len(g)}:")
        for p in g:
            print(f"    - {p}")
        print()

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"threshold": args.threshold, "groups": groups}, indent=2))
        print(f"wrote {args.out}")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# report -- regenerate an HTML memory map from live cluster + dedup data
# ══════════════════════════════════════════════════════════════════════════

_REPORT_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Memory Map v2</title>
<style>
 body{{font:15px -apple-system,sans-serif;max-width:900px;margin:2rem auto;padding:0 1.5rem;color:#16181B;background:#F3F4F1;}}
 h1{{font-size:1.9rem;}}
 .stat-row{{display:flex;gap:.6rem;flex-wrap:wrap;margin:1rem 0 2rem;}}
 .stat{{background:#fff;border:1px solid #DBDDD6;border-radius:8px;padding:.5rem .9rem;}}
 .stat b{{font-family:ui-monospace,monospace;font-size:1.1rem;color:#1F6F78;}}
 .stat span{{font-size:.78rem;color:#5B6169;margin-left:.4rem;}}
 table{{width:100%;border-collapse:collapse;margin:1rem 0;}}
 th,td{{text-align:left;padding:.5rem .7rem;border-bottom:1px solid #DBDDD6;font-size:.88rem;}}
 th{{background:#EAEBE6;font-family:ui-monospace,monospace;text-transform:uppercase;font-size:.68rem;}}
 .group{{background:#fff;border:1px solid #DBDDD6;border-radius:8px;padding:.7rem 1rem;margin-bottom:.6rem;}}
 code{{font-family:ui-monospace,monospace;font-size:.85em;}}
</style></head><body>
<div class="eyebrow" style="font-family:ui-monospace,monospace;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:#1F6F78;">memory-map-optimizer · generated {generated}</div>
<h1>Memory Map v2</h1>
<p>Regenerated from live Leiden clustering and embedding-based dedup, replacing the hand-audited version.</p>
<div class="stat-row">
  <div class="stat"><b>{num_docs}</b><span>indexed files</span></div>
  <div class="stat"><b>{num_clusters}</b><span>real communities (was 0)</span></div>
  <div class="stat"><b>{modularity}</b><span>modularity</span></div>
  <div class="stat"><b>{num_dupe_groups}</b><span>merge candidates found</span></div>
</div>
<h2>Communities (Leiden, real output)</h2>
<table><thead><tr><th>cluster</th><th>size</th></tr></thead><tbody>
{cluster_rows}
</tbody></table>
<h2>Merge candidates ({num_dupe_groups} groups)</h2>
{dupe_html}
</body></html>
"""


def cmd_report(args: argparse.Namespace) -> int:
    import datetime as dt

    num_docs = 0
    if args.index and Path(args.index).expanduser().exists():
        _, docs = load_index(Path(args.index).expanduser())
        num_docs = len(docs)

    num_clusters = modularity = 0
    cluster_rows = "<tr><td colspan=2>run `cluster` first</td></tr>"
    if args.clusters and Path(args.clusters).expanduser().exists():
        cdata = json.loads(Path(args.clusters).expanduser().read_text())
        num_clusters = cdata.get("num_clusters", 0)
        modularity = cdata.get("modularity", 0)
        rows = sorted(cdata.get("cluster_sizes", {}).items(), key=lambda kv: -kv[1])[:20]
        cluster_rows = "\n".join(f"<tr><td>{cid}</td><td>{size}</td></tr>" for cid, size in rows) or cluster_rows

    dupe_groups: list[list[str]] = []
    if args.dedup and Path(args.dedup).expanduser().exists():
        ddata = json.loads(Path(args.dedup).expanduser().read_text())
        dupe_groups = ddata.get("groups", [])
    dupe_html = "\n".join(
        '<div class="group"><b>group of {}</b><br>{}</div>'.format(
            len(g), "<br>".join(f"<code>{p}</code>" for p in g))
        for g in dupe_groups
    ) or "<p>none found (or `dedup` not run yet)</p>"

    html = _REPORT_TEMPLATE.format(
        generated=dt.date.today().isoformat(),
        num_docs=num_docs, num_clusters=num_clusters, modularity=modularity,
        num_dupe_groups=len(dupe_groups),
        cluster_rows=cluster_rows, dupe_html=dupe_html,
    )
    out_path = Path(args.out).expanduser()
    out_path.write_text(html)
    print(f"wrote {out_path}")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cluster", help="Leiden community detection on graphify's graph JSON")
    p.add_argument("--graph", default="~/.graphify/global-graph.json")
    p.add_argument("--nodes-key", default="nodes")
    p.add_argument("--edges-key", default="edges")
    p.add_argument("--source-key", default="source")
    p.add_argument("--target-key", default="target")
    p.add_argument("--id-key", default="id")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write-back", action="store_true", help="write community id onto each node in the graph JSON")
    p.add_argument("--out", default=DEFAULT_CLUSTERS_PATH)
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("embed", help="index memory markdown files with a modern embedding model")
    p.add_argument("--memory-dir", default="~/.claude/projects/*/memory/**")
    p.add_argument("--model", default=DEFAULT_EMBED_MODEL,
                    help=f"HF model id, e.g. {DEFAULT_EMBED_MODEL} or BAAI/bge-m3")
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                    help="cap tokens per doc -- bounds attention-mask memory on long files (default %(default)s)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE,
                    help="docs per encode batch -- smaller bounds peak memory further (default %(default)s)")
    p.add_argument("--index", default=DEFAULT_INDEX_PATH)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("search", help="hybrid BM25 + embedding search over the index")
    p.add_argument("query")
    p.add_argument("--index", default=DEFAULT_INDEX_PATH)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--rerank", action="store_true", help="second-stage cross-encoder rerank")
    p.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                    help="cap tokens per [query,doc] pair for --rerank -- bounds attention memory (default %(default)s)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE,
                    help="pairs per rerank batch (default %(default)s)")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("dedup", help="find near-duplicate / mergeable memory files")
    p.add_argument("--index", default=DEFAULT_INDEX_PATH)
    p.add_argument("--threshold", type=float, default=0.85,
                    help="min cosine similarity to group as duplicates (default %(default)s; "
                         "lower finds more/looser matches, higher finds fewer/stricter ones)")
    p.add_argument("--sweep", metavar="START:STOP:STEP",
                    help="ignore --threshold; run at every threshold in this range and show "
                         "each candidate group's pairwise similarities, so you can see which "
                         "matches are solid (all pairs high) vs. a fragile transitive chain "
                         "(e.g. --sweep 0.80:0.95:0.05)")
    p.add_argument("--out", default=".memopt/dedup.json")
    p.set_defaults(func=cmd_dedup)

    p = sub.add_parser("report", help="regenerate an HTML memory map from live cluster+dedup data")
    p.add_argument("--index", default=DEFAULT_INDEX_PATH)
    p.add_argument("--clusters", default=DEFAULT_CLUSTERS_PATH)
    p.add_argument("--dedup", default=".memopt/dedup.json")
    p.add_argument("--out", default="memory_map_v2.html")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
