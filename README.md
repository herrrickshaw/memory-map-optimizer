# memory-map-optimizer

A command-line tool that audits and tidies a Claude Code memory system: it runs real Leiden community
detection on a graphify knowledge-graph JSON, embeds memory markdown files with a modern Hugging Face
model, searches them (BM25 + embeddings), finds near-duplicate files, and regenerates the "Memory Map"
HTML report with live numbers. Outputs are JSON files under `.memopt/` and `memory_map_v2.html`.

> Status: research-only / dormant (last active mid-Aug 2026). Written for one person's memory layout
> (`~/.graphify/global-graph.json`, `~/.claude/projects/*/memory`); paths are overridable by flag.

New to this? See [How to use this repo with AI](HOW_TO_USE_WITH_AI.md).

`memory_map_v1.html` is the original hand-made "Memory Map — Tiers, Clusters & Resource Routing" audit
(2026-08-12), kept as the baseline. The `report` command's default output, `memory_map_v2.html`, is the
same map driven by real numbers instead of a manual count.

## What it does
- `cluster` — Leiden clustering (`python-igraph` + `leidenalg`) of graphify's graph JSON; fixes graphify's
  broken community detection (it advertised communities but returned 0). `--write-back` stores the
  community id on each node.
- `embed` — indexes memory `.md` files with `Qwen/Qwen3-Embedding-0.6B` (default) or e.g. `BAAI/bge-m3`.
- `search` — hybrid BM25 + embedding retrieval fused by reciprocal rank (no weight tuning), optional
  `--rerank` with `Qwen/Qwen3-Reranker-0.6B`.
- `dedup` — near-duplicate detection by cosine similarity (default threshold 0.85); `--sweep START:STOP:STEP`
  shows pairwise similarities so fragile transitive groups are visible.
- `report` — renders the HTML memory map from the index, cluster and dedup files.

It does not replace graphify or qmd and does not talk to their MCP port or cron job; it reads and writes
the same files.

## Quick start
```bash
git clone https://github.com/herrrickshaw/memory-map-optimizer.git
cd memory-map-optimizer
python3 -m venv .venv && source .venv/bin/activate   # macOS Homebrew Python refuses global pip (PEP 668)
pip install -r requirements.txt
python memory_map_optimizer.py cluster --graph ~/.graphify/global-graph.json --write-back
```

The other stages:
```bash
python memory_map_optimizer.py embed  --memory-dir "~/.claude/projects/*/memory/**"
python memory_map_optimizer.py search "token optimizer routing" --rerank
python memory_map_optimizer.py dedup  --threshold 0.85
python memory_map_optimizer.py report --out memory_map_v2.html
```
If your graph JSON uses other key names, override with `--nodes-key --edges-key --source-key --target-key --id-key`.

## How it works
| Stage | Function(s) in `memory_map_optimizer.py` | Output |
|---|---|---|
| cluster | `_load_graph_json` → `run_leiden` → `write_back_communities` | `.memopt/clusters.json` |
| embed | `_read_memory_files` → `build_index` (`_embed_texts`) | `.memopt/index.json` |
| search | `load_index` → `search_index` (`hybrid_search`, `rerank`) | stdout |
| dedup | `find_duplicate_groups`, `group_pairwise_similarities` | `.memopt/dedup.json` |
| report | `render_report` | `memory_map_v2.html` |

Design notes:
- Model calls are isolated in `_embed_texts` / `_predict_rerank_scores` and injected as `embed_fn` /
  `predict_fn`, so everything else is testable without downloading weights.
- `--max-seq-length` (default 4096) and `--batch-size` (default 8) bound memory: one very long memory file
  once made the uncapped model try to allocate 17 GiB.
- Edges whose endpoints are missing from the graph's `nodes` list (seen in real graphify data) are added as
  nodes rather than crashing igraph; the CLI prints a note.
- Duplicate groups are connected components, so four sequential snapshots surface as one group of four.

## Repository layout
| Path | Purpose |
|---|---|
| `memory_map_optimizer.py` | The whole tool (CLI + library functions) |
| `tests/` | pytest suite; synthetic graphs and hand-built vectors, no network |
| `memory_map_v1.html`, `memory_map_v2.html` | Baseline hand-made audit and a sample generated report |
| `graphify-out/` | graphify knowledge-graph output for this repo (generated; not hand-edited) |
| `requirements.txt`, `pyproject.toml` | Dependencies; pytest and ruff config |
| `.github/workflows/` | `ci.yml` (ruff + pytest), `quality-gates.yml` (gitleaks) |
| `RECREATE_VENV.md` | Note on rebuilding the deleted venv |

## Data sources
No external data is fetched. Inputs are local files (graphify JSON, memory markdown). Embedding and rerank
weights are downloaded once from Hugging Face by `sentence-transformers` on first `embed`/`search`.

## Configuration
No environment variables or secrets. Defaults: graph `~/.graphify/global-graph.json`, memory
`~/.claude/projects/*/memory/**`, index `.memopt/index.json` (gitignored), clusters
`.memopt/clusters.json`, dedup `.memopt/dedup.json`. All overridable by flag.

## Testing
```bash
pip install pytest ruff
pytest -q        # 25 tests, network-free
ruff check .
```

## Limitations and known issues
- Built in a sandbox without access to the real `~/.graphify` or memory directory, and with Hugging Face
  blocked. `embed` was run there and failed only on the model download (proxy 403); it has not been
  verified against a live download in that environment. Run `embed` on a small subset first and sanity-check
  `search` before trusting `dedup` on the full set.
- `cluster` is verified on synthetic graphs with planted communities; `search`/`dedup` logic on hand-built
  vectors, not real embeddings.
- The 0.85 dedup default was lowered from 0.92 after a run on 111 real files found no groups at 0.92.
- `--write-back` rewrites the graph JSON in place (re-indented); keep a copy.
- `report` interpolates file paths into HTML without escaping.
- Memory and graph files contain personal notes; `.memopt/` is gitignored so indexes (which embed file text)
  are not committed.

## Licence
No licence file yet — all rights reserved.
