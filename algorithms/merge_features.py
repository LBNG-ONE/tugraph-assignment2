# -*- coding: utf-8 -*-
"""把算法结果（PageRank / 嵌入聚类 / t-SNE 坐标）合并进用户顶点表，供重新导入 TuGraph。

输入：
  data/user_base.csv           原始用户特征
  algorithms/out/pagerank.csv  PageRank + degree
  algorithms/out/embeddings.npy, node_ids.npy
  algorithms/out/embedding_analysis.json (聚类结果不在此文件里，簇标签从新算或从 npz)
输出：
  data/user.csv                含 pagerank / degree / cluster_id / tsne_x / tsne_y
"""
import argparse
import io
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    # PageRank
    pr = {}
    deg = {}
    with io.open(os.path.join(args.outdir, "pagerank.csv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 3:
                continue
            pr[int(p[0])] = float(p[1])
            deg[int(p[0])] = int(p[2])
    print("pagerank rows:", len(pr), flush=True)

    # 聚类标签（用与 embedding_analysis 相同的随机种子重算，保证一致）
    emb = np.load(os.path.join(args.outdir, "embeddings.npy"))
    ids = np.load(os.path.join(args.outdir, "node_ids.npy"))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.k, n_init=4, random_state=42)
    labels = km.fit_predict(emb)
    cluster = {int(ids[i]): int(labels[i]) for i in range(len(ids))}
    print("cluster assigned:", len(cluster), flush=True)

    # t-SNE 坐标（只有抽样子集有）
    tsne = {}
    xy_path = os.path.join(args.outdir, "tsne_xy.npy")
    sel_path = os.path.join(args.outdir, "tsne_sel.npy")
    if os.path.exists(xy_path) and os.path.exists(sel_path):
        xy = np.load(xy_path)
        sel = np.load(sel_path)
        for k, idx in enumerate(sel):
            tsne[int(ids[idx])] = (float(xy[k, 0]), float(xy[k, 1]))
    print("tsne points:", len(tsne), flush=True)

    n_out = 0
    with io.open(args.base, "r", encoding="utf-8") as f, \
            io.open(args.dest, "w", encoding="utf-8", newline="") as g:
        header = next(f).rstrip("\n")
        g.write(header + ",pagerank,degree,cluster_id,tsne_x,tsne_y\n")
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 7:
                continue
            uid = int(p[0])
            t = tsne.get(uid)
            base_cols = ",".join(p[:7])
            g.write(base_cols + ",{:.10f},{},{},{},{}".format(
                pr.get(uid, 0.0), deg.get(uid, 0), cluster.get(uid, -1),
                ("{:.4f}".format(t[0]) if t else ""),
                ("{:.4f}".format(t[1]) if t else ""),
            ) + "\n")
            n_out += 1
    print("wrote rows:", n_out, "->", args.dest, flush=True)


if __name__ == "__main__":
    main()
