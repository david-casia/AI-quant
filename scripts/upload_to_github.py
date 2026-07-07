# -*- coding: utf-8 -*-
"""
通过 GitHub REST API 上传文件（绕过 github.com:443 直连失败的问题）
api.github.com 可达，github.com 不可达时使用此方案
"""
import os
import base64
import requests
import time
import sys

TOKEN = os.environ.get("GH_TOKEN", "")
OWNER = "david-casia"
REPO = "AI-quant"
BRANCH = "main"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# 要上传的文件（相对于项目根目录的路径）
FILES = [
    ".gitignore",
    "index.html",
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


def get_file_sha(path):
    """检查文件是否已存在，返回sha（用于更新）"""
    url = f"{API}/{path}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def upload_file(local_path, repo_path):
    """上传单个文件到GitHub"""
    full_local = os.path.join(BASE_DIR, local_path)
    with open(full_local, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    size_kb = os.path.getsize(full_local) / 1024

    # 检查是否已存在
    existing_sha = get_file_sha(repo_path)

    data = {
        "message": f"Add {repo_path}",
        "content": content_b64,
        "branch": BRANCH,
    }
    if existing_sha:
        data["sha"] = existing_sha
        data["message"] = f"Update {repo_path}"

    url = f"{API}/{repo_path}"
    r = requests.put(url, headers=HEADERS, json=data, timeout=60)

    if r.status_code in (200, 201):
        print(f"  [OK] {repo_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"  [FAIL] {repo_path}: {r.status_code} {r.text[:200]}")
        return False


def main():
    print("=" * 60)
    print(f"上传 {len(FILES)} 个文件到 {OWNER}/{REPO} (branch: {BRANCH})")
    print("=" * 60)

    ok = 0
    fail = 0
    for i, f in enumerate(FILES, 1):
        print(f"[{i}/{len(FILES)}] 上传 {f}...")
        # URL编码路径中的中文和特殊字符
        repo_path = "/".join(
            requests.utils.quote(part, safe="")
            for part in f.split("/")
        )
        try:
            if upload_file(f, repo_path):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [ERROR] {f}: {e}")
            fail += 1
        time.sleep(0.5)  # 避免rate limit

    print("=" * 60)
    print(f"完成: 成功 {ok}, 失败 {fail}")
    if fail == 0:
        print(f"\n仓库地址: https://github.com/{OWNER}/{REPO}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
