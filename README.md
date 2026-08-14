# memory-map-optimizer

`memory_map_v1.html` is the original "Memory Map — Tiers, Clusters &
Resource Routing" audit (2026-08-12) — the baseline snapshot, taken by hand.
Everything else in this repo exists to produce a v2 of it (`report`'s
default output is literally `memory_map_v2.html`) driven by real numbers
instead of a manual count.

Standalone tool that fixes the two concrete gaps that audit found in your
Claude Code memory system, using the Hugging Face models discussed
alongside it:

1. **graphify's community detection is broken** (advertised, returns 0).
   `cluster` runs real Leiden clustering (`python-igraph` + `leidenalg`)
   against `~/.graphify/global-graph.json` and writes real cluster
   assignments back in.
2. **qmd's embedding model is unspecified/dated.** `embed` re-indexes your
   memory markdown files with a modern small model — default
   `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, ~1.5GB), or pass
   `--model BAAI/bge-m3` for the hybrid dense+sparse+multi-vector model.
   `search` does hybrid BM25+embedding retrieval (reciprocal-rank fusion,
   no weight tuning needed) with an optional `--rerank` pass via
   `Qwen/Qwen3-Reranker-0.6B`.

It also adds **`dedup`** — near-duplicate memory-file detection via
embedding cosine similarity, the thing the audit had to do by hand ("4
sequential token-optimizer snapshots" merged into one).

It does **not** replace graphify or qmd, or talk to graphify's MCP port /
qmd's cron job — it reads and writes the same files those tools already
use, so it's a drop-in augmentation.

## Install

macOS's Homebrew Python refuses global `pip install` (PEP 668,
`externally-managed-environment`) — use a venv, which also sidesteps the
`python`/`pip` vs `python3`/`pip3` aliasing difference entirely since an
activated venv provides both:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(`deactivate` to leave the venv later; re-run the `source` line in new
shells to get back in.)

## Usage

With the venv active (see above):

```bash
# 1. Fix graphify's broken community detection
python memory_map_optimizer.py cluster \
  --graph ~/.graphify/global-graph.json --write-back

# 2. Re-index your memory files with a modern embedding model
python memory_map_optimizer.py embed \
  --memory-dir "~/.claude/projects/*/memory/**"

# 3. Search
python memory_map_optimizer.py search "token optimizer routing" --rerank

# 4. Find merge candidates
python memory_map_optimizer.py dedup --threshold 0.92

# 5. Regenerate the memory-map report with live numbers
python memory_map_optimizer.py report --out memory_map_v2.html
```

If your `~/.graphify/global-graph.json` uses different key names than
`nodes`/`edges`/`source`/`target`/`id`, override them:
`--nodes-key --edges-key --source-key --target-key --id-key`.

## What's tested vs. not (read this before trusting output)

Built and tested in a sandbox with **no access** to your real
`~/.graphify` or `~/.claude/projects/*/memory`, and with `huggingface.co`
itself blocked by that sandbox's network egress. So:

- **`cluster` is verified against real graph structure** — `test_memory_map_optimizer.py`
  builds a synthetic graph with 4 planted communities (dense within-cluster
  edges, sparse cross-cluster edges) and asserts Leiden finds them
  (confirmed: found all 4, modularity 0.72) and never returns 0 — the exact
  bug this fixes. Also ran the actual CLI end-to-end on a second synthetic
  graph (60 nodes, 3 planted clusters → found 3, modularity 0.62,
  `--write-back` confirmed to add the right `community` field).
- **`search`/`dedup`'s logic is verified with hand-built vectors** of known
  cosine similarity, not real embeddings — hybrid ranking and transitive
  duplicate-grouping are confirmed correct.
- **`embed` was actually run** against a small synthetic subset (5 files,
  mirroring the user/feedback/project/reference tiers) and **confirmed to
  fail in this specific sandbox**: `httpcore.ProxyError: 403 Forbidden`
  while `sentence-transformers` tries to download `Qwen/Qwen3-Embedding-0.6B`
  from `huggingface.co`, which this sandbox's network egress proxy blocks
  outright (same block hit earlier building the rest of this session's
  work). That's a property of *this build environment*, not the code —
  it's standard `SentenceTransformer(model_name).encode(...)`, the same
  call path every `sentence-transformers` user relies on. On a machine
  with normal internet access (i.e. yours), the model downloads once,
  caches locally, and every subsequent `embed`/`search` call is offline.
  Still: run `embed` on a small subset first there too, and sanity-check
  `search` results before trusting `dedup` on your full memory set.
- **`report`** is verified to degrade gracefully (correct placeholder text)
  when `embed`/`cluster` haven't been run yet, and to render real numbers
  once they have.

Run `python3 test_memory_map_optimizer.py` yourself to see all of the above.
