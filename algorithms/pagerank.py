# -*- coding: utf-8 -*-
"""算法二：PageRank 中心性 + 与时间/地理特征的联合分析。

内容：
  1. PageRank（幂迭代，无向图，scipy 稀疏矩阵）
  2. 度中心性对照
  3. PageRank 与签到活跃度的相关性（Spearman）
  4. 好友地理距离 vs 随机用户对地理距离（KDD 2011 的核心结论验证）
  5. 活跃度的时间分布（按月 / 按小时）
  6. 输出：pagerank.csv（供回写 TuGraph）、analysis.json、若干图

用法：
  python pagerank.py --edges <friend_edges.txt> --users <user.csv>
                      --checkins <loc-gowalla_totalCheckins.txt> --out <outdir>
"""
import argparse
import io
import json
import os
from collections import Counter, defaultdict

import numpy as np


def load_edges(path):
    src, dst = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            src.append(int(p[0]))
            dst.append(int(p[1]))
    return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def load_users(path):
    ids, cnt, days, lat, lon = [], [], [], [], []
    with io.open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 7:
                continue
            ids.append(int(p[0]))
            cnt.append(int(p[1]) if p[1] else 0)
            days.append(int(p[2]) if p[2] else 0)
            lat.append(float(p[5]) if p[5] else np.nan)
            lon.append(float(p[6]) if p[6] else np.nan)
    return (np.asarray(ids, dtype=np.int64), np.asarray(cnt, dtype=np.int64),
            np.asarray(days, dtype=np.int64), np.asarray(lat), np.asarray(lon))


def pagerank(n_nodes, src, dst, alpha=0.85, tol=1e-10, max_iter=200):
    """无向图 PageRank，幂迭代。"""
    deg = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(deg, src, 1.0)
    np.add.at(deg, dst, 1.0)
    deg[deg == 0] = 1.0
    pr = np.full(n_nodes, 1.0 / n_nodes)
    teleport = (1.0 - alpha) / n_nodes
    for it in range(max_iter):
        # 无向边：每个节点把 pr/deg 平均分给邻居
        contrib = pr / deg
        new = np.zeros(n_nodes, dtype=np.float64)
        np.add.at(new, src, contrib[dst])
        np.add.at(new, dst, contrib[src])
        new = alpha * new + teleport
        delta = np.abs(new - pr).sum()
        pr = new
        if delta < tol:
            break
    return pr, it + 1


def spearman(x, y):
    """Spearman 秩相关（无 scipy 依赖实现）。"""
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(1, len(a) + 1)
        # 并列取平均秩
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        cum = np.cumsum(cnt)
        r = (cum - (cnt - 1) / 2.0)[inv]
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def haversine(lat1, lon1, lat2, lon2):
    """向量化大圆距离（km）。"""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--users", required=True)
    ap.add_argument("--checkins", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}

    print("[1/5] 载入图与用户特征 ...", flush=True)
    src_raw, dst_raw = load_edges(args.edges)
    uids, ucnt, udays, ulat, ulon = load_users(args.users)
    nodes = uids  # user.csv 已包含全部节点
    n = len(nodes)
    remap = {int(v): i for i, v in enumerate(nodes)}
    src = np.array([remap[int(v)] for v in src_raw], dtype=np.int64)
    dst = np.array([remap[int(v)] for v in dst_raw], dtype=np.int64)
    print("      节点={:,} 边={:,}".format(n, len(src)), flush=True)
    res["n_nodes"] = int(n)
    res["n_edges"] = int(len(src))

    print("[2/5] PageRank 幂迭代 ...", flush=True)
    pr, iters = pagerank(n, src, dst)
    deg = np.zeros(n, dtype=np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    res["pagerank_iterations"] = int(iters)
    res["pagerank_sum"] = float(pr.sum())
    top = np.argsort(pr)[::-1][:20]
    res["top20_pagerank"] = [
        {"user_id": int(nodes[i]), "pagerank": float(pr[i]), "degree": int(deg[i]),
         "checkin_count": int(ucnt[i]), "active_days": int(udays[i])}
        for i in top
    ]
    print("      Top1 user={} pr={:.6f} deg={}".format(nodes[top[0]], pr[top[0]], deg[top[0]]), flush=True)

    print("[3/5] PageRank 与活跃度的相关性 ...", flush=True)
    mask = ucnt > 0
    rho_pr_cnt = spearman(pr[mask], ucnt[mask])
    rho_pr_days = spearman(pr[mask], udays[mask])
    rho_deg_cnt = spearman(deg[mask].astype(float), ucnt[mask])
    res["spearman_pagerank_vs_checkin"] = round(rho_pr_cnt, 4)
    res["spearman_pagerank_vs_activedays"] = round(rho_pr_days, 4)
    res["spearman_degree_vs_checkin"] = round(rho_deg_cnt, 4)
    res["users_with_checkin"] = int(mask.sum())
    print("      ρ(PageRank, 签到数) = {:.4f}".format(rho_pr_cnt), flush=True)

    print("[4/5] 好友地理距离 vs 随机用户对 ...", flush=True)
    idx = np.arange(n)
    # 只看有位置的用户
    has_loc = ~np.isnan(ulat)
    loc_index = np.full(n, -1, dtype=np.int64)
    loc_index[has_loc] = np.arange(has_loc.sum())
    m = (loc_index[src] >= 0) & (loc_index[dst] >= 0)
    s2, d2 = loc_index[src][m], loc_index[dst][m]
    lat_h, lon_h = ulat[has_loc], ulon[has_loc]
    friend_d = haversine(lat_h[s2], lon_h[s2], lat_h[d2], lon_h[d2])
    rng = np.random.default_rng(7)
    k = min(len(friend_d), 400000)
    ia = rng.integers(0, has_loc.sum(), k)
    ib = rng.integers(0, has_loc.sum(), k)
    keep = ia != ib
    rand_d = haversine(lat_h[ia[keep]], lon_h[ia[keep]], lat_h[ib[keep]], lon_h[ib[keep]])
    res["friend_pairs_with_location"] = int(m.sum())
    res["friend_dist_median_km"] = round(float(np.median(friend_d)), 2)
    res["friend_dist_mean_km"] = round(float(friend_d.mean()), 2)
    res["random_dist_median_km"] = round(float(np.median(rand_d)), 2)
    res["random_dist_mean_km"] = round(float(rand_d.mean()), 2)
    res["friend_dist_lt100km_ratio"] = round(float((friend_d < 100).mean()), 4)
    res["random_dist_lt100km_ratio"] = round(float((rand_d < 100).mean()), 4)
    print("      好友距离中位数 {:.1f} km vs 随机 {:.1f} km".format(
        res["friend_dist_median_km"], res["random_dist_median_km"]), flush=True)

    print("[5/5] 时间分布统计 ...", flush=True)
    by_month = Counter()
    by_hour = Counter()
    by_weekday = Counter()
    n_ck = 0
    with io.open(args.checkins, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            n_ck += 1
            t = p[1]
            by_month[t[:7]] += 1
            by_hour[t[11:13]] += 1
    res["n_checkins"] = int(n_ck)
    res["checkins_by_month"] = dict(sorted(by_month.items()))
    res["checkins_by_hour"] = {k2: by_hour[k2] for k2 in sorted(by_hour)}
    res["peak_hour"] = max(by_hour.items(), key=lambda kv: kv[1])[0]
    res["peak_month"] = max(by_month.items(), key=lambda kv: kv[1])[0]
    print("      签到峰值小时 = {} 点, 峰值月份 = {}".format(res["peak_hour"], res["peak_month"]), flush=True)

    with open(os.path.join(args.out, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # 供回写 TuGraph 的 pagerank.csv
    with io.open(os.path.join(args.out, "pagerank.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("id,pagerank,degree\n")
        for i in range(n):
            f.write("{},{:.10f},{}\n".format(int(nodes[i]), pr[i], int(deg[i])))

    print("=== 完成 ===", flush=True)
    for k2 in ("pagerank_iterations", "spearman_pagerank_vs_checkin",
               "spearman_degree_vs_checkin", "friend_dist_median_km",
               "random_dist_median_km", "friend_dist_lt100km_ratio",
               "random_dist_lt100km_ratio", "peak_hour", "peak_month"):
        print("  {:<34} {}".format(k2, res[k2]), flush=True)


if __name__ == "__main__":
    main()
