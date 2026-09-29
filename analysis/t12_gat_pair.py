"""
t12_gat_pair.py — T12: GCATSL-style graph attention probe (revision R2.2)
=========================================================================
A compact graph-attention reimplementation of the attention-based SL
paradigm (GCATSL class), evaluated under the temporal pair protocol:

  graph       PPI snapshot of the given vintage UNION the SL-pair edges
              available at the cutoff (label-dependent edges only from
              <=cutoff evidence), restricted to candidate genes
  node input  the three network features computed on that same PPI graph
              (degree rank, PPR from HR seeds, BFS proximity)
  model       2-layer GAT (4 heads, 16 dims each, ELU), 64-d embeddings,
              pair score = dot product of final embeddings
  training    BCE on cutoff-preceding SL pairs vs an equal number of the
              T10 shared training negatives (identical negative discipline)
  testing     post-cutoff pairs vs the T10 shared test negatives

The current-vintage graph (STRING 2021) and the 2016-vintage graph
(Hetionet) runs share everything except the PPI snapshot, so the
difference isolates the network-vintage effect inside a learned
attention representation.

Output: results/t12_gat_pair.csv
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "E:/崔雷/博士/设计/ovc_project/circ_aware_bib/v11_timemachine/src")
import tm_utils as U
from t2_vintage_chain import graph_hetionet, graph_string2021, features_on_graph
from t10_pair_level import export_pairs, eval_pairs, paired_pair_test

V11 = U.V11
SEED = U.SEED
CUTOFFS = U.CUTOFFS


class GATLayer(nn.Module):
    """Edge-centric single-relational GAT layer (pure torch, CPU-friendly)."""

    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)

    def forward(self, x, ei):
        h = self.W(x).view(-1, self.heads, self.out_dim)      # N x H x D
        src, dst = ei
        hs, hd = h[src], h[dst]                                # E x H x D
        score = F.leaky_relu((hs * self.a_src).sum(-1)
                             + (hd * self.a_dst).sum(-1), 0.2)  # E x H
        out = torch.zeros_like(h)
        exp = score.exp()                                      # E x H
        for k in range(self.heads):
            denom = torch.zeros(x.size(0)).index_add_(0, dst, exp[:, k])
            alpha = exp[:, k] / (denom[dst] + 1e-12)
            agg = torch.zeros(x.size(0), self.out_dim).index_add_(
                0, dst, hs[:, k, :] * alpha.unsqueeze(-1))
            out[:, k, :] = agg
        return out.reshape(x.size(0), -1)                      # N x (H*D)


class GATPair(nn.Module):
    def __init__(self, in_dim, hid=16, heads=4):
        super().__init__()
        self.l1 = GATLayer(in_dim, hid, heads)
        self.l2 = GATLayer(heads * hid, hid, heads)

    def embed(self, x, ei):
        return F.elu(self.l2(F.elu(self.l1(x, ei)), ei))       # N x heads*hid


def train_gat(graph_ppi, sl_edges, node_feats, pos_pairs, neg_pairs,
              epochs=120, lr=0.01, seed=SEED):
    torch.manual_seed(seed)
    nodes = sorted(({g for g, n in graph_ppi.items() if n}
                    | {g for e in sl_edges for g in e}
                    | {g for p in pos_pairs + neg_pairs for g in p}))
    idx = {g: i for i, g in enumerate(nodes)}
    N = len(nodes)

    e_set = set()
    for a, nbrs in graph_ppi.items():
        for b in nbrs:
            e_set.add((idx[a], idx[b]))
    for a, b in sl_edges:
        e_set.add((idx[a], idx[b]))
        e_set.add((idx[b], idx[a]))
    ei = torch.tensor(np.array(sorted(e_set)).T, dtype=torch.long)

    x = torch.tensor(np.stack([node_feats.get(g, np.zeros(3, np.float32))
                               for g in nodes]), dtype=torch.float32)

    pos = torch.tensor([[idx[a], idx[b]] for a, b in pos_pairs],
                       dtype=torch.long)
    neg = torch.tensor([[idx[a], idx[b]] for a, b in neg_pairs],
                       dtype=torch.long)

    model = GATPair(3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        h = model.embed(x, ei)
        ps = model_score(h, pos)
        ns = model_score(h, neg)
        logits = torch.cat([ps, ns])
        y = torch.cat([torch.ones(len(ps)), torch.zeros(len(ns))])
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
        if ep % 30 == 0 or ep == epochs - 1:
            with torch.no_grad():
                acc = ((logits > 0).float() == y).float().mean().item()
            print(f"    epoch {ep:3d} loss={loss.item():.4f} acc={acc:.3f}")
    model.eval()
    return model, idx, ei, x


def model_score(h, pairs):
    return (h[pairs[:, 0]] * h[pairs[:, 1]]).sum(-1)


def main():
    t0 = time.time()
    torch.set_num_threads(8)
    scores = U.load_scores()
    cand = scores["gene"].tolist()
    cand_set = set(cand)

    posE = export_pairs(cand_set)
    pos_list = list(zip(posE["a"], posE["b"]))
    pos_year = dict(zip(pos_list, posE["year"]))

    pools = pd.read_csv(f"{V11}/results/t10_pools.csv")
    rnd = pools[pools["pool"] == "random"]
    neg_tr = [tuple(x) for x in
              rnd[rnd["split"] == "train"][["a", "b"]].to_numpy()]
    neg_te = [tuple(x) for x in
              rnd[rnd["split"] == "test"][["a", "b"]].to_numpy()]
    print(f"positives {len(pos_list)}, train negs {len(neg_tr)}, "
          f"test negs {len(neg_te)}")

    sys.path.insert(0, f"{U.ROOT}/crisprsl")
    sys.path.insert(0, U.ROOT)
    import config as cfg
    seeds = list(cfg.HR_SEED_GENES)

    def restrict(graph):
        return {a: {b for b in nbrs if b in cand_set}
                for a, nbrs in graph.items() if a in cand_set}

    graphs = {"STRING_2021": restrict(graph_string2021()),
              "Hetionet_2016": restrict(graph_hetionet())}
    feats = {name: features_on_graph(g, cand, seeds)[0]
             for name, g in graphs.items()}
    node_feats = {name: {g: feats[name][i].astype(np.float32)
                         for i, g in enumerate(cand)} for name in graphs}

    rng = np.random.default_rng(SEED + 500)
    rows = []
    for c in CUTOFFS:
        tr_pos = [k for k in pos_list if pos_year[k] <= c]
        te_pos = [k for k in pos_list if pos_year[k] > c]
        n_neg = min(len(tr_pos), len(neg_tr))
        neg_samp = [neg_tr[i] for i in
                    rng.choice(len(neg_tr), n_neg, replace=False)]
        sl_edges = tr_pos
        outs = {}
        for vname in graphs:
            t = time.time()
            model, idx, ei, x = train_gat(
                graphs[vname], sl_edges, node_feats[vname],
                tr_pos, neg_samp, seed=SEED + hash(vname) % 100 + c)
            with torch.no_grad():
                h = model.embed(x, ei)
                te_all = te_pos + neg_te
                pairs_t = torch.tensor([[idx.get(a, -1), idx.get(b, -1)]
                                        for a, b in te_all], dtype=torch.long)
                ok = (pairs_t[:, 0] >= 0) & (pairs_t[:, 1] >= 0)
                s = np.zeros(len(te_all))
                s[ok.numpy()] = model_score(h, pairs_t[ok]).numpy()
            yte = np.array([1] * len(te_pos) + [0] * len(neg_te))
            ev = eval_pairs(yte, s, seed=SEED + c)
            outs[vname] = s
            rows.append({"model": f"GAT pair ({vname})", "cutoff": c, **ev,
                         "n_test_pos": len(te_pos)})
            print(f"  GAT {vname} @{c}: {ev['auroc']} [{ev['ci_lo']},"
                  f"{ev['ci_hi']}] n+={len(te_pos)} ({time.time() - t:.0f}s)")
        d, lo, hi, p = paired_pair_test(yte, outs["Hetionet_2016"],
                                        outs["STRING_2021"],
                                        seed=SEED + 313 + c)
        rows.append({"model": "GAT pair absorption (2021 minus 2016)",
                     "cutoff": c, "auroc": round(d, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4), "p": round(p, 4),
                     "n_test_pos": len(te_pos)})
        print(f"  GAT absorption @{c}: {d:+.4f} (p={p:.4f})")

    pd.DataFrame(rows).to_csv(f"{V11}/results/t12_gat_pair.csv", index=False)
    print(f"T12 done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
