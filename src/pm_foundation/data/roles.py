"""Vocabulary-free activity fingerprints + directly-follows graph ("role" inputs).

``fit_role_graph`` turns a corpus of traces into per-activity **behavior fingerprints**
(blocked: graph / time / behavior) plus the weighted DFG adjacency — the inputs of the
:class:`~pm_foundation.models.role_encoder.ActivityEncoder`. Nothing here touches
activity *identity*: every feature is a corpus-level statistic, so two datasets with
disjoint vocabularies land in the same, comparable feature space.

LEAKAGE CONTRACT (mirrors ``fit_feature_spec``'s train-only convention):
    * Fingerprints are **corpus statistics over completed historical traces** — they are
      computed ONCE from a reference corpus and never from the case being predicted.
    * The reference corpus must be a TRAINING split: at pretraining time the pretrain
      train split; at evaluation time the EVAL dataset's train split — NEVER traces that
      will be scored. Callers pass the trace list explicitly; there is no "whole log"
      convenience path.
"""

from __future__ import annotations

import math
import zlib
from collections import defaultdict

import torch

from pm_foundation.data.schema import Trace
from pm_foundation.data.vocabulary import RESERVED_TOKENS, Vocabulary

# Fingerprint layout (rank-normalized to [0,1] except the binary in_cycle):
#   graph    (6): in_degree, out_degree, pagerank, betweenness, in_cycle, self_loop_p
#   time     (6): median/std/p90 of log1p in-gap; median/std/p90 of log1p out-gap
#   behavior (8): p_start, p_terminal, support, pred_entropy, succ_entropy,
#                 mean_pos, std_pos, rework_p
N_ROLE_FEATURES = 20
# Name channel: hashed character trigrams (stable crc32, NOT python hash), padded/truncated.
NAME_TRIGRAM_SLOTS = 24
NAME_HASH_BUCKETS = 4096  # bucket 0 is reserved for padding


def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Tie-aware average-rank normalization to [0,1] (equal values share a rank)."""
    n = x.shape[0]
    if n <= 1:
        return torch.zeros_like(x)
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = torch.arange(n, dtype=torch.float64)
    uniq, inv = torch.unique(x, return_inverse=True)
    sums = torch.zeros(len(uniq), dtype=torch.float64).scatter_add_(0, inv, ranks)
    counts = torch.zeros(len(uniq), dtype=torch.float64).scatter_add_(
        0, inv, torch.ones(n, dtype=torch.float64)
    )
    return ((sums / counts)[inv] / (n - 1)).to(torch.float32)


def _weak_components(succ: list[list[int]], n: int) -> list[list[int]]:
    """Weakly-connected components (edges treated as undirected) of the real subgraph.

    Multi-dataset corpora have disjoint activity vocabularies, so each source log forms its own
    component — which is what makes per-component rank normalization equivalent to fitting that
    log on its own (see the normalization note in :func:`fit_role_graph`).
    """
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, bs in enumerate(succ):
        for b in bs:
            adj[a].append(b)
            adj[b].append(a)
    seen = [False] * n
    comps: list[list[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _betweenness(succ: list[list[int]], n: int) -> list[float]:
    """Brandes betweenness centrality on an unweighted directed graph (n is small)."""
    bc = [0.0] * n
    for s in range(n):
        stack: list[int] = []
        preds: list[list[int]] = [[] for _ in range(n)]
        sigma = [0] * n
        dist = [-1] * n
        sigma[s], dist[s] = 1, 0
        queue = [s]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            stack.append(v)
            for w in succ[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = [0.0] * n
        for w in reversed(stack):
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                bc[w] += delta[w]
    return bc


def _pagerank(p_out: torch.Tensor, damping: float = 0.85, iters: int = 60) -> torch.Tensor:
    """Power-iteration PageRank over a row-stochastic transition matrix (dangling->uniform)."""
    n = p_out.shape[0]
    if n == 0:
        return torch.zeros(0)
    dangling = p_out.sum(dim=1) == 0
    pr = torch.full((n,), 1.0 / n)
    for _ in range(iters):
        spread = (pr * dangling.float()).sum() / n
        pr = (1 - damping) / n + damping * (p_out.t() @ pr + spread)
    return pr


def _in_cycle(succ: list[list[int]], n: int) -> list[bool]:
    """Node is in a cycle iff it can reach itself (BFS per node; n is small)."""
    flags = []
    for s in range(n):
        seen = [False] * n
        queue = list(succ[s])
        hit = False
        while queue:
            v = queue.pop()
            if v == s:
                hit = True
                break
            if not seen[v]:
                seen[v] = True
                queue.extend(succ[v])
        flags.append(hit)
    return flags


def _entropy(counts: dict[int, int], n_classes: int) -> float:
    total = sum(counts.values())
    if total == 0 or n_classes <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    return h / math.log(n_classes)


def _gap_stats(gaps: list[float]) -> tuple[float, float, float]:
    """(median, std, p90) of log1p gap seconds; zeros when no observations."""
    if not gaps:
        return 0.0, 0.0, 0.0
    t = torch.tensor([math.log1p(max(g, 0.0)) for g in gaps], dtype=torch.float64)
    return float(t.median()), float(t.std(unbiased=False)), float(t.quantile(0.9))


def name_trigram_ids(name: str) -> list[int]:
    """Stable hashed char-trigram ids (1..buckets-1; 0 is padding), fixed-length."""
    s = f"#{name.lower()}#"
    grams = [s[i : i + 3] for i in range(max(len(s) - 2, 1))]
    ids = [1 + zlib.crc32(g.encode("utf-8")) % (NAME_HASH_BUCKETS - 1) for g in grams]
    ids = ids[:NAME_TRIGRAM_SLOTS]
    return ids + [0] * (NAME_TRIGRAM_SLOTS - len(ids))


def fit_role_graph(traces: list[Trace], vocab: Vocabulary) -> dict[str, torch.Tensor]:
    """Fit per-activity fingerprints + DFG adjacency from a TRAINING corpus.

    Returns tensors aligned to ``vocab`` ids (reserved-token rows are zero):
        ``feats``     (V, N_ROLE_FEATURES)  rank-normalized fingerprints
        ``adj_in``    (V, V)  in-neighbor transition matrix, row-normalized
        ``adj_out``   (V, V)  out-neighbor transition matrix, row-normalized
        ``name_ids``  (V, NAME_TRIGRAM_SLOTS)  hashed char-trigram ids (0-padded)
        ``real_mask`` (V,)  bool — rows that are real activities seen in the corpus

    See the module docstring for the leakage contract: ``traces`` must be a training
    split, never traces that will later be predicted/scored.
    """
    vsize = len(vocab)
    n_reserved = len(RESERVED_TOKENS)
    tokens = vocab.to_list()

    # --- pass over the corpus: transitions, gaps, per-trace usage ---------------
    trans: dict[tuple[int, int], int] = defaultdict(int)
    in_gaps: dict[int, list[float]] = defaultdict(list)
    out_gaps: dict[int, list[float]] = defaultdict(list)
    starts: dict[int, int] = defaultdict(int)
    terminals: dict[int, int] = defaultdict(int)
    containing: dict[int, int] = defaultdict(int)
    repeated: dict[int, int] = defaultdict(int)
    pos_sum: dict[int, float] = defaultdict(float)
    pos_sq: dict[int, float] = defaultdict(float)
    occurrences: dict[int, int] = defaultdict(int)
    n_traces = 0

    for trace in traces:
        events = trace.events
        if not events:
            continue
        n_traces += 1
        ids = [vocab.encode(e.activity) for e in events]
        starts[ids[0]] += 1
        terminals[ids[-1]] += 1
        seen_counts: dict[int, int] = defaultdict(int)
        length = len(ids)
        for i, a in enumerate(ids):
            seen_counts[a] += 1
            occurrences[a] += 1
            pos = i / (length - 1) if length > 1 else 0.0
            pos_sum[a] += pos
            pos_sq[a] += pos * pos
        for a, c in seen_counts.items():
            containing[a] += 1
            if c >= 2:
                repeated[a] += 1
        for i in range(length - 1):
            a, b = ids[i], ids[i + 1]
            trans[(a, b)] += 1
            gap = (events[i + 1].timestamp - events[i].timestamp).total_seconds()
            out_gaps[a].append(gap)
            in_gaps[b].append(gap)

    real = sorted({a for a in containing if a >= n_reserved})
    n_real = len(real)
    row_of = {a: i for i, a in enumerate(real)}

    # --- adjacency (real activities only), row-normalized ----------------------
    adj_out = torch.zeros(vsize, vsize)
    for (a, b), c in trans.items():
        adj_out[a, b] = float(c)
    out_sums = adj_out.sum(dim=1, keepdim=True).clamp(min=1e-9)
    in_sums = adj_out.sum(dim=0, keepdim=True).clamp(min=1e-9)
    adj_in = (adj_out / in_sums).t().contiguous()  # row a = P(prev | a)
    adj_out_n = adj_out / out_sums  # row a = P(next | a)

    # --- graph block over the real subgraph ------------------------------------
    succ = [[row_of[b] for b in range(vsize) if adj_out[a, b] > 0 and b in row_of] for a in real]
    bc = _betweenness(succ, n_real)
    p_sub = torch.zeros(n_real, n_real)
    for (a, b), c in trans.items():
        if a in row_of and b in row_of:
            p_sub[row_of[a], row_of[b]] = float(c)
    p_sub = p_sub / p_sub.sum(dim=1, keepdim=True).clamp(min=1e-9)
    pr = _pagerank(p_sub)
    cyc = _in_cycle(succ, n_real)

    # --- assemble raw features per real activity -------------------------------
    raw = torch.zeros(n_real, N_ROLE_FEATURES)
    for a in real:
        i = row_of[a]
        n_occ = max(occurrences[a], 1)
        mean_pos = pos_sum[a] / n_occ
        var_pos = max(pos_sq[a] / n_occ - mean_pos * mean_pos, 0.0)
        pred_counts = {b: int(adj_out[b, a]) for b in range(vsize) if adj_out[b, a] > 0}
        succ_counts = {b: int(adj_out[a, b]) for b in range(vsize) if adj_out[a, b] > 0}
        raw[i, 0] = float(sum(1 for b in range(vsize) if adj_out[b, a] > 0))  # in-degree
        raw[i, 1] = float(sum(1 for b in range(vsize) if adj_out[a, b] > 0))  # out-degree
        raw[i, 2] = float(pr[i])
        raw[i, 3] = float(bc[i])
        raw[i, 4] = 1.0 if cyc[i] else 0.0
        raw[i, 5] = float(adj_out_n[a, a])  # self-loop probability
        raw[i, 6:9] = torch.tensor(_gap_stats(in_gaps.get(a, [])))
        raw[i, 9:12] = torch.tensor(_gap_stats(out_gaps.get(a, [])))
        raw[i, 12] = starts[a] / max(n_traces, 1)
        raw[i, 13] = terminals[a] / max(n_traces, 1)
        raw[i, 14] = containing[a] / max(n_traces, 1)
        raw[i, 15] = _entropy(pred_counts, n_real)
        raw[i, 16] = _entropy(succ_counts, n_real)
        raw[i, 17] = mean_pos
        raw[i, 18] = math.sqrt(var_pos)
        raw[i, 19] = repeated[a] / max(containing[a], 1)

    # --- rank-normalize PER WEAKLY-CONNECTED COMPONENT, scatter into vocab rows ----
    # A rank is only meaningful relative to the population it was computed over. Normalizing
    # GLOBALLY broke multi-dataset training: the corpus graph is fitted over the union of all
    # source logs, but at eval the graph is fitted on the target log ALONE — so the same activity
    # received a different fingerprint at train vs eval time (measured: mean |Δ| 0.13-0.22, max
    # 0.86 on a [0,1] scale, worst on the absolute-timescale gap features, because the union's
    # ranks were dominated by whichever source contributed the most activities). The GIN was then
    # fed out-of-distribution fingerprints at eval, which is why cross-domain matching scored
    # BELOW uniform while in-domain (where the ID channel carries the signal) looked fine.
    # Components == source logs here (disjoint vocabularies), so this makes a log's fingerprints
    # identical whether fitted alone or inside a corpus. A single-log graph is one component -> no-op.
    for comp in _weak_components(succ, n_real):
        idx = torch.tensor(comp, dtype=torch.long)
        for col in range(N_ROLE_FEATURES):
            if col == 4:  # in_cycle is binary — keep raw
                continue
            # a lone activity has no population to rank against -> neutral 0.5, not an extreme
            raw[idx, col] = (
                _rank_normalize(raw[idx, col]) if len(comp) > 1 else torch.full((1,), 0.5)
            )
    feats = torch.zeros(vsize, N_ROLE_FEATURES)
    name_ids = torch.zeros(vsize, NAME_TRIGRAM_SLOTS, dtype=torch.long)
    real_mask = torch.zeros(vsize, dtype=torch.bool)
    for a in real:
        feats[a] = raw[row_of[a]]
        name_ids[a] = torch.tensor(name_trigram_ids(tokens[a]), dtype=torch.long)
        real_mask[a] = True

    return {
        "feats": feats,
        "adj_in": adj_in,
        "adj_out": adj_out_n,
        "name_ids": name_ids,
        "real_mask": real_mask,
    }
