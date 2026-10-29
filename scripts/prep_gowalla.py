# -*- coding: utf-8 -*-
"""Gowalla 数据预处理：生成 TuGraph 导入用的 CSV。

输出（到 作业2数据集/processed/）：
  friend.csv      无向好友边，去重（保留 a<b），2 列：src_id,dst_id
  user.csv        用户顶点，含由签到记录派生的时序/地理特征
  stats.txt       预处理统计
  friend_edges.txt 纯边表（供 Node2Vec 使用，每行 "a b"）
"""
import io
import os
from collections import defaultdict

BASE = r"D:\dsh\workplace\专业综合实践作业\作业2数据集"
OUT = os.path.join(BASE, "processed")
os.makedirs(OUT, exist_ok=True)

EDGES = os.path.join(BASE, "loc-gowalla_edges.txt")
CHECKINS = os.path.join(BASE, "loc-gowalla_totalCheckins.txt")

log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(msg)


# ---------------- 1. 好友边去重 ----------------
log("=== 1. 处理好友边 ===")
undirected = set()
all_nodes = set()
rows = 0
with io.open(EDGES, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        rows += 1
        parts = line.split()
        if len(parts) < 2:
            continue
        a, b = int(parts[0]), int(parts[1])
        all_nodes.add(a)
        all_nodes.add(b)
        undirected.add((a, b) if a < b else (b, a))
log("  原始行数      : {:,}".format(rows))
log("  唯一节点数    : {:,}".format(len(all_nodes)))
log("  去重无向边数  : {:,}".format(len(undirected)))

friend_csv = os.path.join(OUT, "friend.csv")
with io.open(friend_csv, "w", encoding="utf-8", newline="") as f:
    f.write("src_id,dst_id\n")
    for a, b in sorted(undirected):
        f.write("{},{}\n".format(a, b))
log("  写出 friend.csv")

edges_txt = os.path.join(OUT, "friend_edges.txt")
with io.open(edges_txt, "w", encoding="utf-8", newline="") as f:
    for a, b in sorted(undirected):
        f.write("{} {}\n".format(a, b))
log("  写出 friend_edges.txt")

# ---------------- 2. 签到记录 -> 用户时序/地理特征 ----------------
log("=== 2. 处理签到记录 ===")
cnt = defaultdict(int)          # user -> checkin count
dates = defaultdict(set)        # user -> set of yyyy-mm-dd
first = {}                      # user -> min timestamp
last = {}                       # user -> max timestamp
sum_lat = defaultdict(float)
sum_lon = defaultdict(float)
loc_set = set()
n_checkins = 0

with io.open(CHECKINS, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        n_checkins += 1
        u = int(parts[0])
        t = parts[1]                    # 2010-07-24T13:45:06Z
        lat = float(parts[2])
        lon = float(parts[3])
        loc_set.add(parts[4])
        cnt[u] += 1
        d = t[:10]
        dates[u].add(d)
        if u not in first or t < first[u]:
            first[u] = t
        if u not in last or t > last[u]:
            last[u] = t
        sum_lat[u] += lat
        sum_lon[u] += lon

log("  签到记录数    : {:,}".format(n_checkins))
log("  有签到的用户  : {:,}".format(len(cnt)))
log("  唯一地点数    : {:,}".format(len(loc_set)))

# 用户集合 = 好友图节点 ∪ 有签到的用户
users = sorted(all_nodes | set(cnt.keys()))
log("  用户顶点总数  : {:,}".format(len(users)))

user_csv = os.path.join(OUT, "user.csv")
with io.open(user_csv, "w", encoding="utf-8", newline="") as f:
    f.write("id,checkin_count,active_days,first_checkin,last_checkin,mean_lat,mean_lon\n")
    for u in users:
        c = cnt.get(u, 0)
        ad = len(dates.get(u, ()))
        fc = first.get(u, "")
        lc = last.get(u, "")
        mlat = round(sum_lat[u] / c, 6) if c else ""
        mlon = round(sum_lon[u] / c, 6) if c else ""
        f.write("{},{},{},{},{},{},{}\n".format(u, c, ad, fc, lc, mlat, mlon))
log("  写出 user.csv")

# ---------------- 3. 统计 ----------------
log("=== 3. 统计 ===")
deg = defaultdict(int)
for a, b in undirected:
    deg[a] += 1
    deg[b] += 1
degs = sorted(deg.values(), reverse=True)
avg_deg = sum(degs) / len(degs) if degs else 0
log("  平均度        : {:.4f}".format(avg_deg))
log("  最大度        : {}".format(degs[0] if degs else 0))
log("  度>=2 节点数  : {:,}".format(sum(1 for d in degs if d >= 2)))
log("  度=0 节点数   : {:,}".format(len(users) - len(deg)))
with_checkin = sum(1 for u in users if cnt.get(u, 0) > 0)
log("  有签到用户占比: {:.2%} ({:,}/{:,})".format(with_checkin / len(users), with_checkin, len(users)))
log("  人均签到次数  : {:.2f}".format(n_checkins / max(1, len(cnt))))
log("  地点平均签到数: {:.2f}".format(n_checkins / max(1, len(loc_set))))

stats_path = os.path.join(OUT, "stats.txt")
with io.open(stats_path, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(log_lines) + "\n")

log("=== 完成 ===")
for name in ("friend.csv", "user.csv", "friend_edges.txt", "stats.txt"):
    p = os.path.join(OUT, name)
    log("  {:<20} {:.1f} MB".format(name, os.path.getsize(p) / 1024 / 1024))
