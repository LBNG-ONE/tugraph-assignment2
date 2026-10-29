# -*- coding: utf-8 -*-
"""Node2Vec 嵌入的下游分析与可视化。

  1. KMeans 聚类 + 轮廓系数
  2. t-SNE 降维可视化（按聚类着色、按活跃度着色）
  3. 嵌入近邻检索（"好友推荐"效果验证）
  4. 度分布 / PageRank 分布 / 签到小时分布图
"""
import argparse
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def load_users(path):
    d = {}
    with io.open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 7:
                continue
            d[int(p[0])] = (int(p[1]) if p[1] else 0, int(p[2]) if p[2] else 0,
                            float(p[5]) if p[5] else np.nan,
                            float(p[6]) if p[6] else np.nan)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--users", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tsne-n", type=int, default=5000)
    args = ap.parse_args()

    emb = np.load(os.path.join(args.out, "embeddings.npy"))
    ids = np.load(os.path.join(args.out, "node_ids.npy"))
    users = load_users(args.users)
    n, dim = emb.shape
    print("embeddings:", emb.shape, flush=True)
    res = {"n_nodes": int(n), "dim": int(dim)}

    # ---------- 1. KMeans ----------
    print("[1/4] KMeans (k={}) ...".format(args.k), flush=True)
    km = KMeans(n_clusters=args.k, n_init=4, random_state=42)
    labels = km.fit_predict(emb)
    rng = np.random.default_rng(0)
    sil_idx = rng.choice(n, size=min(20000, n), replace=False)
    sil = float(silhouette_score(emb[sil_idx], labels[sil_idx]))
    res["kmeans_k"] = args.k
    res["silhouette"] = round(sil, 4)
    sizes = np.bincount(labels, minlength=args.k)
    res["cluster_sizes"] = [int(x) for x in sizes]
    print("      silhouette = {:.4f}, sizes = {}".format(sil, sizes.tolist()), flush=True)

    # 各簇平均活跃度（把嵌入簇和"现实意义"对上）
    cnt = np.array([users.get(int(u), (0, 0, np.nan, np.nan))[0] for u in ids])
    res["cluster_mean_checkin"] = [round(float(cnt[labels == c].mean()), 3) for c in range(args.k)]
    print("      各簇平均签到数:", res["cluster_mean_checkin"], flush=True)

    # ---------- 2. t-SNE ----------
    print("[2/4] t-SNE (n={}) ...".format(args.tsne_n), flush=True)
    sel = rng.choice(n, size=min(args.tsne_n, n), replace=False)
    ts = TSNE(n_components=2, perplexity=30, init="pca", random_state=42, max_iter=600)
    xy = ts.fit_transform(emb[sel])
    np.save(os.path.join(args.out, "tsne_xy.npy"), xy)
    np.save(os.path.join(args.out, "tsne_sel.npy"), sel)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    c1 = labels[sel]
    ax = axes[0]
    for c in range(args.k):
        m = c1 == c
        ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.6, label="C{}".format(c))
    ax.set_title("Node2Vec embeddings (t-SNE) colored by KMeans cluster")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=4, fontsize=8, loc="best", framealpha=0.6)

    ax2 = axes[1]
    cc = cnt[sel]
    cap = np.percentile(cc[cc > 0], 95) if (cc > 0).any() else 1
    sc = ax2.scatter(xy[:, 0], xy[:, 1], s=3, c=np.clip(cc, 0, cap), cmap="viridis", alpha=0.75)
    ax2.set_title("colored by check-in count (clipped at p95)")
    ax2.set_xlabel("dim 1")
    fig.colorbar(sc, ax=ax2, label="check-in count")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_tsne.png"), dpi=150)
    plt.close(fig)
    print("      fig_tsne.png saved", flush=True)

    # ---------- 3. 嵌入近邻检索 ----------
    print("[3/4] 近邻检索 ...", flush=True)
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    samples = []
    for c in range(min(3, args.k)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        q = int(idx[0])
        sims = norm @ norm[q]
        top = np.argsort(sims)[::-1][1:6]
        samples.append({
            "query_user": int(ids[q]), "cluster": int(c),
            "top5": [{"user_id": int(ids[t]), "cosine": round(float(sims[t]), 4)} for t in top],
        })
    res["neighbor_examples"] = samples

    # ---------- 4. 统计图 ----------
    print("[4/4] 分布图 ...", flush=True)
    with open(os.path.join(args.out, "analysis.json"), "r", encoding="utf-8") as f:
        pa = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hours = sorted(pa["checkins_by_hour"].keys())
    vals = [pa["checkins_by_hour"][h] for h in hours]
    axes[0].bar([int(h) for h in hours], vals, color="#4e79a7")
    axes[0].set_title("Check-ins by hour of day (UTC)")
    axes[0].set_xlabel("hour"); axes[0].set_ylabel("check-ins")

    months = sorted(pa["checkins_by_month"].keys())
    mv = [pa["checkins_by_month"][m] for m in months]
    axes[1].plot(range(len(months)), mv, marker="o", ms=3, color="#f28e2b")
    axes[1].set_title("Check-ins by month")
    axes[1].set_xticks(range(0, len(months), 3))
    axes[1].set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, fontsize=7)

    pr = np.array([r["pagerank"] for r in pa["top20_pagerank"]])
    axes[2].barh(range(len(pr))[::-1], pr, color="#59a14f")
    axes[2].set_title("Top-20 PageRank users")
    axes[2].set_xlabel("pagerank"); axes[2].set_ylabel("rank")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_temporal_pagerank.png"), dpi=150)
    plt.close(fig)
    print("      fig_temporal_pagerank.png saved", flush=True)

    with open(os.path.join(args.out, "embedding_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("=== 完成 ===", flush=True)
    print("  silhouette      =", res["silhouette"], flush=True)
    print("  cluster sizes   =", res["cluster_sizes"], flush=True)
    print("  簇均签到数      =", res["cluster_mean_checkin"], flush=True)


if __name__ == "__main__":
    main()
