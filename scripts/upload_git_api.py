# -*- coding: utf-8 -*-
"""
通过 Git Data API 批量推送文件（一次性创建tree+commit，减少API调用次数）
适用于 fine-grained token 有 push 权限但 Contents API 403 的情况
"""
import os
import base64
import requests
import sys
import json

TOKEN = os.environ.get("GH_TOKEN", "")
OWNER = "david-casia"
REPO = "AI-quant"
BRANCH = "main"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}"

FILES = [
    ".gitignore",
    "index.html",
    "quant_analysis.ipynb",
    "scripts/upload_git_api.py",
    "scripts/upload_to_github.py",
    "TASK1/build_pdf.py",
    "TASK1/content.py",
    "TASK1/fetch_data.py",
    "TASK1/plot_close.py",
    "TASK1/style.css",
    "TASK1/002594_close.png",
    "TASK1/002594_daily.csv",
    "TASK1/薛德刚+TASK1.html",
    "TASK1/薛德刚+TASK1.pdf",
    "TASK2/build_pdf.py",
    "TASK2/content.py",
    "TASK2/indicators.py",
    "TASK2/plot_indicators.py",
    "TASK2/style.css",
    "TASK2/002594_boll.png",
    "TASK2/002594_kdj.png",
    "TASK2/002594_macd.png",
    "TASK2/002594_rsi.png",
    "TASK2/002594_daily.csv",
    "TASK2/002594_with_indicators.csv",
    "TASK2/薛德刚+TASK2.html",
    "TASK2/薛德刚+TASK2.pdf",
]


def api(method, endpoint, **kwargs):
    url = f"{API_BASE}{endpoint}" if endpoint.startswith("/") else f"{API_BASE}/{endpoint}"
    r = requests.request(method, url, headers=HEADERS, timeout=60, **kwargs)
    return r


def main():
    print("=" * 60)
    print(f"Git Data API 批量推送 {len(FILES)} 个文件")
    print("=" * 60)

    # 1. 获取 main 分支的当前 commit SHA（如果仓库为空，则从头创建）
    print("\n[1] 获取分支信息...")
    r = api("GET", f"/git/refs/heads/{BRANCH}")
    parent_sha = None
    base_tree_sha = None

    if r.status_code == 200:
        parent_sha = r.json()["object"]["sha"]
        # 获取 commit 获取 tree sha
        r2 = api("GET", f"/git/commits/{parent_sha}")
        if r2.status_code == 200:
            base_tree_sha = r2.json()["tree"]["sha"]
        print(f"  当前 HEAD: {parent_sha[:12]}")
        print(f"  Base tree: {base_tree_sha[:12]}")
    else:
        print(f"  分支不存在，将创建初始提交 (status {r.status_code})")

    # 2. 为每个文件创建 blob
    print(f"\n[2] 创建 blobs...")
    tree_items = []
    for i, fpath in enumerate(FILES, 1):
        full = os.path.join(BASE_DIR, fpath)
        with open(full, "rb") as f:
            content_bytes = f.read()
        content_b64 = base64.b64encode(content_bytes).decode("utf-8")
        size_kb = len(content_bytes) / 1024

        r = api("POST", "/git/blobs", json={
            "content": content_b64,
            "encoding": "base64",
        })
        if r.status_code == 201:
            blob_sha = r.json()["sha"]
            tree_items.append({
                "path": fpath,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha,
            })
            print(f"  [{i}/{len(FILES)}] {fpath} ({size_kb:.1f} KB) -> {blob_sha[:12]}")
        else:
            print(f"  [{i}/{len(FILES)}] FAIL {fpath}: {r.status_code} {r.text[:150]}")
            return 1

    # 3. 创建 tree
    print(f"\n[3] 创建 tree ({len(tree_items)} items)...")
    tree_payload = {"tree": tree_items}
    if base_tree_sha:
        tree_payload["base_tree"] = base_tree_sha

    r = api("POST", "/git/trees", json=tree_payload)
    if r.status_code != 201:
        print(f"  FAIL: {r.status_code} {r.text[:300]}")
        return 1
    new_tree_sha = r.json()["sha"]
    print(f"  Tree: {new_tree_sha[:12]}")

    # 4. 创建 commit
    print(f"\n[4] 创建 commit...")
    commit_data = {
        "message": "改用前复权(qfq)数据重跑全套\n\n- fetch_data.py: adj=None -> adj='qfq'\n- TASK1: 收盘价图移除除权日标注，分析文本更新\n- TASK2: 4张指标图移除除权日竖虚线，分析文本更新\n- 价格曲线连续平滑，指标信号真实可靠",
        "tree": new_tree_sha,
    }
    if parent_sha:
        commit_data["parents"] = [parent_sha]
    else:
        commit_data["parents"] = []

    r = api("POST", "/git/commits", json=commit_data)
    if r.status_code != 201:
        print(f"  FAIL: {r.status_code} {r.text[:300]}")
        return 1
    new_commit_sha = r.json()["sha"]
    print(f"  Commit: {new_commit_sha[:12]}")

    # 5. 更新分支引用
    print(f"\n[5] 更新 {BRANCH} 分支引用...")
    r = api("PATCH", f"/git/refs/heads/{BRANCH}", json={"sha": new_commit_sha})
    if r.status_code == 200:
        print(f"  SUCCESS: {BRANCH} -> {new_commit_sha[:12]}")
    else:
        # 可能分支还不存在，尝试创建
        r2 = api("POST", "/git/refs", json={
            "ref": f"refs/heads/{BRANCH}",
            "sha": new_commit_sha,
        })
        if r2.status_code == 201:
            print(f"  CREATED: {BRANCH} -> {new_commit_sha[:12]}")
        else:
            print(f"  FAIL update: {r.status_code} {r.text[:200]}")
            print(f"  FAIL create: {r2.status_code} {r2.text[:200]}")
            return 1

    print("\n" + "=" * 60)
    print("全部上传成功！")
    print(f"仓库: https://github.com/{OWNER}/{REPO}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
