# -*- coding: utf-8 -*-
"""算法一：Node2Vec 节点嵌入（含链接预测评估）。

流程：
  1. 读取去重后的无向好友边表
  2. 划出 10% 边作为链接预测的测试集
  3. 在剩余 90% 边上做二阶有偏随机游走（p=1, q=2，偏向 BFS/社区内）
  4. gensim Word2Vec（skip-gram）训练 64 维嵌入
  5. 用余弦相似度做链接预测，计算 AUC
  6. 输出嵌入矩阵与评估指标

用法：
  python node2vec.py --edges <friend_edges.txt> --out <outdir> [--walks 5] [--length 60]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
from gensim.models import Word2Vec


def load_edges(path):
    src, dst = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            src.append(int(parts[0]))
            dst.append(int(parts[1]))
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def build_adj(n_nodes, src, dst):
    """返回 CSR 风格的邻接表：list of numpy arrays（无向图双向添加）。"""
    deg = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    start = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(deg, out=start[1:])
    nbr = np.empty(int(deg.sum()), dtype=np.int32)
    pos = start[:-1].copy()
    for u, v in zip(src, dst):
        nbr[pos[u]] = v
        pos[u] += 1
        nbr[pos[v]] = u
        pos[v] += 1
    return start, nbr


def random_walks(start, nbr, n_nodes, num_walks, walk_len, p, q, seed=42):
    """二阶有偏随机游走，返回 (n_nodes*num_walks, walk_len) 的 int32 矩阵。"""
    rng = random.Random(seed)
    nbr_sets = [set(nbr[start[u]:start[u + 1]].tolist()) for u in range(n_nodes)]
    nbr_lists = [nbr[start[u]:start[u + 1]] for u in range(n_nodes)]
    total = n_nodes * num_walks
    out = np.empty((total, walk_len), dtype=np.int32)
    idx = 0
    t0 = time.time()
    for u in range(n_nodes):
        for _ in range(num_walks):
            out[idx, 0] = u
            prev = -1
            cur = u
            for step in range(1, walk_len):
                cand = nbr_lists[cur]
                if len(cand) == 0:
                    out[idx, step] = cur
                    continue
                if prev < 0 or len(cand) == 1:
                    nxt = int(cand[rng.randrange(len(cand))])
                else:
                    pset = nbr_sets[prev]
                    ws = []
                    for n in cand:
                        n = int(n)
                        if n == prev:
                            ws.append(1.0 / p)
                        elif n in pset:
                            ws.append(1.0)
                        else:
                            ws.append(1.0 / q)
                    r = rng.random() * sum(ws)
                    acc = 0.0
                    nxt = int(cand[-1])
                    for n, w in zip(cand, ws):
                        acc += w
                        if r <= acc:
                            nxt = int(n)
                            break
                out[idx, step] = nxt
                prev, cur = cur, nxt
            idx += 1
        if u % 20000 == 0 and u > 0:
            el = time.time() - t0
            print("  walks {}/{} nodes, {:.0f}s elapsed, eta {:.0f}s".format(
                u, n_nodes, el, el / u * (n_nodes - u)), flush=True)
    return out


class WalkCorpus:
    """可重复迭代的 gensim 语料（每个 epoch 重新遍历同一批游走）。"""

    def __init__(self, walks):
        self.walks = walks

    def __iter__(self):
        for w in self.walks:
            yield w.tolist()


def auc_score(pos, neg):
    """基于秩的 AUC，无需 sklearn。"""
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列：用平均秩
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    avg_rank = cum - (cnt - 1) / 2.0
    ranks = avg_rank[inv]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def node2vec(edges_path, outdir, num_walks=5, walk_len=60, dim=64, p=1.0, q=2.0,
             window=10, epochs=3, workers=8, sample_neg=50000, seed=42):
    os.makedirs(outdir, exist_ok=True)
    t_start = time.time()

    print("[1/6] 读取边表 ...", flush=True)
    raw_src, raw_dst = load_edges(edges_path)
    nodes = np.unique(np.concatenate([raw_src, raw_dst]))
    n_nodes = len(nodes)
    remap = {int(v): i for i, v in enumerate(nodes)}
    src = np.array([remap[int(v)] for v in raw_src], dtype=np.int64)
    dst = np.array([remap[int(v)] for v in raw_dst], dtype=np.int64)
    print("      边数={:,} 节点数={:,}".format(len(src), n_nodes), flush=True)

    print("[2/6] 划分训练/测试边 (90/10) ...", flush=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(src))
    n_test = int(len(src) * 0.10)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    tr_src, tr_dst = src[train_idx], dst[train_idx]
    te_src, te_dst = src[test_idx], dst[test_idx]
    print("      train={:,} test={:,}".format(len(tr_src), len(te_src)), flush=True)

    print("[3/6] 构建邻接表 ...", flush=True)
    adj_start, adj_nbr = build_adj(n_nodes, tr_src, tr_dst)

    print("[4/6] 二阶有偏随机游走 (walks={} len={} p={} q={}) ...".format(
        num_walks, walk_len, p, q), flush=True)
    walks = random_walks(adj_start, adj_nbr, n_nodes, num_walks, walk_len, p, q, seed)
    print("      walks 矩阵 {} , {:.1f}s".format(walks.shape, time.time() - t_start), flush=True)

    print("[5/6] Word2Vec 训练 (dim={} window={} epochs={}) ...".format(dim, window, epochs), flush=True)
    model = Word2Vec(
        sentences=WalkCorpus(walks),
        vector_size=dim, window=window, min_count=0, sg=1,
        workers=workers, epochs=epochs, seed=seed, negative=5, sample=1e-4,
    )
    print("      词表大小={:,}".format(len(model.wv)), flush=True)

    emb = np.zeros((n_nodes, dim), dtype=np.float32)
    for i in range(n_nodes):
        if i in model.wv:
            emb[i] = model.wv[i]
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    print("[6/6] 链接预测评估 ...", flush=True)
    train_pairs = set(map(tuple, np.sort(np.stack([tr_src, tr_dst], axis=1), axis=1).tolist()))
    pos_pairs = np.stack([te_src, te_dst], axis=1)
    # 采样负例：随机不存在的边
    negs = []
    tries = 0
    while len(negs) < sample_neg and tries < sample_neg * 50:
        tries += 1
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in train_pairs:
            continue
        negs.append((a, b))
    neg_pairs = np.asarray(negs, dtype=np.int64)[:sample_neg]

    pos_scores = np.einsum("ij,ij->i", emb[pos_pairs[:, 0]], emb[pos_pairs[:, 1]])
    neg_scores = np.einsum("ij,ij->i", emb[neg_pairs[:, 0]], emb[neg_pairs[:, 1]])
    auc = auc_score(pos_scores, neg_scores)

    metrics = {
        "n_nodes": int(n_nodes),
        "n_edges": int(len(raw_src)),
        "train_edges": int(len(tr_src)),
        "test_edges": int(len(te_src)),
        "num_walks": num_walks,
        "walk_length": walk_len,
        "dim": dim,
        "p": p,
        "q": q,
        "window": window,
        "epochs": epochs,
        "link_prediction_auc": round(float(auc), 6),
        "pos_score_mean": round(float(pos_scores.mean()), 6),
        "neg_score_mean": round(float(neg_scores.mean()), 6),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(os.path.join(outdir, "node2vec_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(outdir, "embeddings.npy"), emb)
    np.save(os.path.join(outdir, "node_ids.npy"), nodes)

    print("=== Node2Vec 完成 ===", flush=True)
    for k, v in metrics.items():
        print("  {:<22} {}".format(k, v), flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--walks", type=int, default=5)
    ap.add_argument("--length", type=int, default=60)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--neg", type=int, default=50000)
    args = ap.parse_args()
    node2vec(args.edges, args.out, num_walks=args.walks, walk_len=args.length,
             dim=args.dim, epochs=args.epochs, sample_neg=args.neg)
