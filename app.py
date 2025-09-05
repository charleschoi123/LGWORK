# app.py  —— LGWORK 后端（完整覆盖版）
# -*- coding: utf-8 -*-

import os
import re
import csv
import io
import json
import uuid
import time
import math
import uuid
import string
import random
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import requests
from flask import Flask, request, jsonify, send_from_directory, make_response

# -------------------------------
# 环境变量
# -------------------------------
JOBS_CSV_PATH      = os.environ.get("JOBS_CSV_PATH", "data/jobs.csv")
TRACKER_JSON_PATH  = os.environ.get("TRACKER_JSON_PATH", "data/tracker.json")
ADMIN_TOKEN        = os.environ.get("ADMIN_TOKEN", os.environ.get("ADMIN_TOKEN".lower(), ""))  # 兼容

LLM_BASE_URL       = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")  # 兼容 OpenAI 风格
LLM_API_KEY        = os.environ.get("LLM_API_KEY", "")
MODEL_NAME         = os.environ.get("MODEL_NAME", "deepseek-chat")               # 兼容 OpenAI 风格模型名
LLM_PROVIDER       = os.environ.get("LLM_PROVIDER", "deepseek")                  # 仅内部用于日志

os.makedirs(os.path.dirname(JOBS_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TRACKER_JSON_PATH), exist_ok=True)

app = Flask(__name__)


# -------------------------------
# 工具函数
# -------------------------------

def json_response(obj, status=200):
    resp = make_response(jsonify(obj), status)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

def _now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _normalize_text(s: str) -> str:
    return (s or "").replace("\u3000", " ").replace("\xa0", " ").strip()

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def _csv_headers():
    return [
        "id","title","company","source","url","location","posted_at","tags","ingested_at"
    ]

def _append_jobs_to_csv(items):
    """items: list of dict -> 追加写入 CSV。去重策略：同 url 不重复"""
    existing_urls = set()
    if os.path.exists(JOBS_CSV_PATH):
        with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                u = (row.get("url") or "").strip()
                if u:
                    existing_urls.add(u)

    new_rows = []
    for it in items:
        it = it.copy()
        url = _normalize_text(it.get("url",""))
        if not url or url in existing_urls:
            continue
        row = {
            "id": _normalize_text(it.get("id") or str(uuid.uuid4())),
            "title": _normalize_text(it.get("title")),
            "company": _normalize_text(it.get("company")),
            "source": _normalize_text(it.get("source") or "unknown"),
            "url": url,
            "location": _normalize_text(it.get("location")),
            "posted_at": _normalize_text(it.get("posted_at")),
            "tags": ";".join(_ensure_list(it.get("tags"))),
            "ingested_at": _now_iso(),
        }
        new_rows.append(row)
        existing_urls.add(url)

    wrote = 0
    if new_rows:
        write_header = not os.path.exists(JOBS_CSV_PATH) or os.path.getsize(JOBS_CSV_PATH) == 0
        with open(JOBS_CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_csv_headers())
            if write_header:
                w.writeheader()
            for row in new_rows:
                w.writerow(row)
                wrote += 1
    return wrote

def _read_jobs():
    jobs = []
    if not os.path.exists(JOBS_CSV_PATH):
        return jobs
    with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row = dict(row)
            row["tags"] = [x for x in (row.get("tags") or "").split(";") if x]
            jobs.append(row)
    return jobs

# -------------------------------
# LLM（OpenAI 兼容模式；可用 deepseek；若无 key 自动降级本地）
# -------------------------------

def llm_chat(system_prompt: str, user_prompt: str, max_tokens=600, temperature=0.2):
    """兼容 OpenAI Chat Completions 的最小实现"""
    if not LLM_API_KEY:
        # 无 Key：降级本地简易“假摘要”
        return _local_summary(user_prompt)

    url = LLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user",   "content": user_prompt or ""},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            return _local_summary(user_prompt)
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content.strip() or _local_summary(user_prompt)
    except Exception:
        return _local_summary(user_prompt)

def _local_summary(resume_text: str) -> str:
    """本地的简易“兜底摘要”，无 key 时不至于空白"""
    txt = _normalize_text(resume_text)[:2000]
    # 非常简陋的提炼
    lines = [x.strip() for x in txt.splitlines() if x.strip()]
    head = lines[:5]
    summary = " · ".join([x[:80] for x in head])[:500]
    return f"候选人摘要（离线模式）：{summary or '拥有相关教育背景与项目经历，具备一定的技术与沟通能力。'}"


# -------------------------------
# Workday 解析与抓取
# -------------------------------

def _parse_workday_line(line: str):
    """
    支持三种写法：
    1) tenant/site
    2) tenant/site@wd5   （手动指定集群）
    3) 完整 URL（自动抽取 tenant、site、cluster）
    返回: dict(tenant, site, cluster|None) 或 None
    """
    if not line:
        return None
    s = line.strip()

    # tenant/site@wdX
    m = re.match(r'^([^/\s]+)/([^@\s]+)@(?:(wd\d+))$', s, flags=re.I)
    if m:
        return {"tenant": m.group(1), "site": m.group(2), "cluster": m.group(3)}

    # tenant/site
    if 'myworkdayjobs.com' not in s and '/' in s:
        t, si = [x.strip() for x in s.split('/', 1)]
        if t and si:
            return {"tenant": t, "site": si, "cluster": None}

    # 完整 URL 1：tenant.wd5.myworkdayjobs.com/<locale>/<site>
    m = re.search(
        r'https?://([^.]+)\.(wd\d+)\.myworkdayjobs\.com/(?:[a-zA-Z-]+/)?([^/?#\s]+)',
        s, flags=re.I
    )
    if m:
        return {"tenant": m.group(1), "site": m.group(3), "cluster": m.group(2)}

    # 完整 URL 2：.../recruiting/<tenant>/<site>
    m = re.search(
        r'https?://.+?/recruiting/([^/]+)/([^/?#\s]+)',
        s, flags=re.I
    )
    if m:
        return {"tenant": m.group(1), "site": m.group(2), "cluster": None}

    return None


def fetch_workday_jobs(tenant: str, site: str, cluster: str | None = None,
                       max_pages: int = 30, page_size: int = 50):
    """
    Workday 官方 JSON 接口：
    https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs
    自动探测 cluster 列表
    """
    clusters_try = [cluster] if cluster else ['wd5', 'wd3', 'wd1', 'wd2', 'wd4', 'wd7']
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json",
    }

    all_items = []
    for c in clusters_try:
        if not c:
            continue
        host = f"https://{tenant}.{c}.myworkdayjobs.com"
        api = f"{host}/wday/cxs/{tenant}/{site}/jobs"
        try:
            offset = 0
            found_any = False
            while True:
                resp = requests.post(api, json={
                    "limit": page_size,
                    "offset": offset,
                    "searchText": "",
                    "appliedFacets": {}
                }, headers=headers, timeout=20)

                if resp.status_code != 200:
                    break

                data = resp.json()
                postings = data.get("jobPostings", []) or []
                if not postings:
                    break

                found_any = True
                for p in postings:
                    ext = p.get("externalPath") or ""
                    title = (p.get("title") or "").strip()
                    loc = p.get("locationsText") or ""
                    poston = p.get("postedOn") or ""
                    all_items.append({
                        "id": p.get("bulletFields", [None])[0] or ext or str(uuid.uuid4()),
                        "title": title,
                        "company": tenant,
                        "source": "workday",
                        "url": host + ext if ext else host,
                        "location": loc,
                        "posted_at": poston,
                        "tags": [f"workday:{tenant}"],
                    })

                offset += len(postings)
                if len(postings) < page_size or offset >= max_pages * page_size:
                    break

            if found_any:
                break  # 找到有效集群就停止尝试其它集群
        except Exception:
            continue

    return all_items


# -------------------------------
# Greenhouse / Lever 抓取
# -------------------------------

def fetch_greenhouse_jobs(slug: str, max_items: int = 500):
    """
    Greenhouse boards API:
    https://boards-api.greenhouse.io/v1/boards/{slug}/jobs
    """
    slug = (slug or "").strip()
    if not slug:
        return []
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return []
        data = resp.json()
        jobs = data.get("jobs", []) or []
        out = []
        for j in jobs[:max_items]:
            out.append({
                "id": j.get("id") or str(uuid.uuid4()),
                "title": _normalize_text(j.get("title")),
                "company": slug,
                "source": "greenhouse",
                "url": j.get("absolute_url") or "",
                "location": (j.get("location") or {}).get("name") or "",
                "posted_at": "",
                "tags": [f"greenhouse:{slug}"],
            })
        return out
    except Exception:
        return []


def fetch_lever_jobs(slug: str, max_items: int = 500):
    """
    Lever postings API:
    https://api.lever.co/v0/postings/{slug}?mode=json
    """
    slug = (slug or "").strip()
    if not slug:
        return []
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return []
        arr = resp.json() or []
        out = []
        for j in arr[:max_items]:
            out.append({
                "id": j.get("id") or str(uuid.uuid4()),
                "title": _normalize_text(j.get("text")),
                "company": slug,
                "source": "lever",
                "url": j.get("hostedUrl") or "",
                "location": ", ".join([l.get("name") for l in (j.get("categories") or {}).get("location", [])]) if isinstance((j.get("categories") or {}).get("location"), list) else (j.get("categories") or {}).get("location") or "",
                "posted_at": j.get("createdAt") and datetime.utcfromtimestamp(j["createdAt"]/1000).strftime("%Y-%m-%d") or "",
                "tags": [f"lever:{slug}"],
            })
        return out
    except Exception:
        return []


# -------------------------------
# 简历上传/解析
# -------------------------------

def _extract_text_from_pdf(fileobj: io.BytesIO) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(fileobj) or ""
    except Exception:
        return ""

def _extract_text_from_docx(fileobj: io.BytesIO) -> str:
    try:
        import docx
        fileobj.seek(0)
        doc = docx.Document(fileobj)
        return "\n".join([p.text for p in doc.paragraphs]) or ""
    except Exception:
        return ""

def _extract_resume_text(fs) -> str:
    """fs: Werkzeug FileStorage"""
    filename = (fs.filename or "").lower()
    content = fs.read()
    bio = io.BytesIO(content)

    if filename.endswith(".pdf"):
        txt = _extract_text_from_pdf(bio)
    elif filename.endswith(".docx"):
        txt = _extract_text_from_docx(bio)
    else:
        # 纯文本
        try:
            txt = content.decode("utf-8", errors="ignore")
        except Exception:
            txt = ""

    return _normalize_text(txt)


# -------------------------------
# 基础打分（本地启发式）
# -------------------------------

def _tokenize(s: str):
    s = (s or "").lower()
    for ch in ",.;:!|/\\()[]{}<>@#$%^&*_+-=\"'":
        s = s.replace(ch, " ")
    return [x for x in s.split() if x]

def _score_job_for_resume(job, resume_text, keywords=None, city=None):
    title = job.get("title","")
    loc = job.get("location","")
    url = job.get("url","")
    company = job.get("company","")

    base = 0
    r_tok = set(_tokenize(resume_text))
    t_tok = set(_tokenize(title))
    base += len(r_tok & t_tok) * 5

    # 关键词
    for kw in keywords or []:
        kw = (kw or "").strip().lower()
        if not kw: 
            continue
        if kw in (title or "").lower():
            base += 6
        if kw in (company or "").lower():
            base += 3

    # 地点
    if city:
        c = city.strip().lower()
        if c and c in (loc or "").lower():
            base += 5

    # URL 优势（仅用于打散）
    if "biotech" in url:
        base += 1

    # 归一化/限制
    base = max(0, min(100, base))
    return base


# -------------------------------
# 路由：静态
# -------------------------------

@app.route("/")
def home():
    # 默认打开求职页
    return send_from_directory("public", "agent.html")

@app.route("/admin")
def admin_page():
    return send_from_directory("public", "admin.html")

@app.route("/public/<path:path>")
def public_file(path):
    return send_from_directory("public", path)


# -------------------------------
# 路由：职位列表
# -------------------------------

@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    type_ = (request.args.get("type") or "").strip().lower()
    work_mode = (request.args.get("work_mode") or "").strip()
    location = (request.args.get("location") or "").strip()
    sort_posted_desc = request.args.get("sort_posted_desc", "0") == "1"

    items = _read_jobs()

    # 简易过滤
    if location:
        toks = [x.strip().lower() for x in _ensure_list(location.split(","))]
        items = [x for x in items if any(t in (x.get("location") or "").lower() for t in toks)]

    total = len(items)
    # 简易排序：按 ingested_at 倒序
    items = sorted(items, key=lambda x: x.get("ingested_at") or "", reverse=True)

    # 分页
    start = (page-1)*page_size
    end = start + page_size
    return json_response({
        "ok": True,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items[start:end],
    })


# -------------------------------
# 路由：简历上传
# -------------------------------

@app.route("/api/upload_resume", methods=["POST"])
def api_upload_resume():
    if "file" not in request.files:
        return json_response({"ok": False, "error": "no file"}, 400)
    fs = request.files["file"]
    text = _extract_resume_text(fs)
    return json_response({"ok": True, "text": text})


# -------------------------------
# 路由：简历分析与职位推荐
# -------------------------------

SYS_PROMPT_SUMMARY = (
    "你是资深招聘顾问，请基于用户粘贴的中文或英文简历，生成一个不超过200字的第三人称简介，"
    "不要包含姓名、电话、邮箱等隐私；然后用逗号给出 6~10 个关键词（技能/行业/职能/领域），"
    "格式：先输出【简介】：……；再输出【关键词】：kw1, kw2, …。"
)

@app.route("/api/analyze_recommend", methods=["POST"])
def api_analyze_recommend():
    data = request.get_json(silent=True) or {}
    resume_text = data.get("resume_text","") or ""
    keywords = [x.strip() for x in (data.get("keywords","") or "").split(",") if x.strip()]
    loc = (data.get("location") or "").strip()
    page = int(data.get("page", 1))
    page_size = int(data.get("page_size", 50))

    # 1) LLM/本地生成“画像摘要 + 关键词”
    summary = llm_chat(SYS_PROMPT_SUMMARY, resume_text, max_tokens=480) if resume_text else ""

    # 从摘要中尝试读取关键词（若模型已按要求输出）
    auto_kws = []
    m = re.search(r"【关键词】[:：]\s*(.+)$", summary, flags=re.S)
    if m:
        part = m.group(1)
        auto_kws = [x.strip() for x in re.split(r"[，,；;]", part) if x.strip()]
        # 清理摘要只保留第一段
        summary = re.split(r"【关键词】", summary)[0].strip()
    # 合并用户关键词
    kw_final = list({*(x.lower() for x in keywords), *(x.lower() for x in auto_kws)})

    # 2) 读取本地职位并做启发式匹配
    all_jobs = _read_jobs()
    scored = []
    for j in all_jobs:
        sc = _score_job_for_resume(j, resume_text, keywords=kw_final, city=loc)
        if sc > 0:
            scored.append((sc, j))
    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    total = len(scored)
    start = (page-1)*page_size
    end = start + page_size
    page_items = [dict(item, **{"match_score": sc}) for sc, item in scored[start:end]]

    return json_response({
        "ok": True,
        "profile": {
            "summary": summary,
            "keywords": kw_final,
        },
        "total": total,
        "items": page_items,
    })


# -------------------------------
# 路由：批量匹配（可选，兼容前端）
# -------------------------------

@app.route("/api/match_batch", methods=["POST"])
def api_match_batch():
    data = request.get_json(silent=True) or {}
    resume_text = data.get("resume_text","") or ""
    kw = data.get("keywords") or []
    city = data.get("city") or ""

    jobs = data.get("jobs") or []
    out = []
    for j in jobs:
        sc = _score_job_for_resume(j, resume_text, keywords=kw, city=city)
        out.append({"url": j.get("url"), "score": sc})

    return json_response({"ok": True, "items": out})


# -------------------------------
# 路由：职位采集
# -------------------------------

@app.route("/api/ingest_ats", methods=["POST"])
def api_ingest_ats():
    """
    POST JSON 示例：
    {
      "greenhouse_slugs": "airbnb, stripe, figma",
      "lever_slugs": "scaleai",
      "workday_urls": "https://beigene.wd5.myworkdayjobs.com/en-US/BeiGene\nhttps://pfizer.wd1.myworkdayjobs.com/PfizerCareers",
      "admin_token": "xxxx"   // 或 header: X-Admin-Token
    }
    """
    data = request.get_json(silent=True) or {}
    header_token = (request.headers.get("X-Admin-Token") or "").strip()
    body_token = (data.get("token") or data.get("admin_token") or "").strip()
    client_token = header_token or body_token

    if ADMIN_TOKEN and client_token != ADMIN_TOKEN:
        return json_response({"ok": False, "error": "unauthorized"}, 401)

    # 1) 解析输入
    gh_text = (data.get("greenhouse_slugs") or "").strip()
    lv_text = (data.get("lever_slugs") or "").strip()
    wd_text = (data.get("workday_urls") or data.get("workday_lines")
               or data.get("workday_sites") or data.get("wd") or "").strip()

    gh_slugs = [x.strip() for x in re.split(r"[,;\s]+", gh_text) if x.strip()]
    lv_slugs = [x.strip() for x in re.split(r"[,;\s]+", lv_text) if x.strip()]
    wd_lines = [ln for ln in wd_text.splitlines() if ln.strip()]

    detail = {"greenhouse": 0, "lever": 0, "workday": 0}
    all_items = []

    # 2) Greenhouse
    for s in gh_slugs:
        try:
            items = fetch_greenhouse_jobs(s)
            detail["greenhouse"] += len(items)
            all_items.extend(items)
        except Exception:
            continue

    # 3) Lever
    for s in lv_slugs:
        try:
            items = fetch_lever_jobs(s)
            detail["lever"] += len(items)
            all_items.extend(items)
        except Exception:
            continue

    # 4) Workday（支持 URL/简写/显式集群）
    for line in wd_lines:
        info = _parse_workday_line(line)
        if not info:
            continue
        try:
            items = fetch_workday_jobs(
                info["tenant"], info["site"], info.get("cluster"),
                max_pages=30, page_size=50
            )
            detail["workday"] += len(items)
            all_items.extend(items)
        except Exception:
            continue

    wrote = _append_jobs_to_csv(all_items)
    return json_response({"ok": True, "ingested": wrote, "detail": detail})


# 兼容：国内源占位（目前返回 0）
@app.route("/api/ingest_cn_jobs", methods=["POST"])
def api_ingest_cn_jobs():
    return json_response({"ok": True, "ingested": 0})


# -------------------------------
# 健康检查
# -------------------------------

@app.route("/healthz")
def healthz():
    return "ok", 200


# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
