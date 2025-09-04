# -*- coding: utf-8 -*-
import os
import csv
import json
import uuid
import time
import re
from typing import List, Dict, Any

import requests
from flask import Flask, request, jsonify, send_from_directory, make_response

# 可选第三方：PDF/DOCX 解析（没装也能跑，上传解析接口会给出友好提示）
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# ------------------------------------------------------------------------------
# 基本配置
# ------------------------------------------------------------------------------
app = Flask(__name__)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

JOBS_CSV_PATH = os.environ.get("JOBS_CSV_PATH", os.path.join(DATA_DIR, "jobs.csv"))
ADMIN_TOKEN   = os.environ.get("ADMIN_TOKEN", "").strip()

# LLM（DeepSeek/OpenAI 兼容接口）
LLM_API_KEY   = os.environ.get("LLM_API_KEY", "").strip()
LLM_BASE_URL  = (os.environ.get("LLM_BASE_URL", "").strip() or "https://api.deepseek.com").rstrip("/")
LLM_PROVIDER  = os.environ.get("LLM_PROVIDER", "deepseek").strip().lower()
MODEL_NAME    = os.environ.get("MODEL_NAME", "deepseek-chat").strip()

VERSION = "LGWORK-1.2.0-full"

# ------------------------------------------------------------------------------
# 通用工具
# ------------------------------------------------------------------------------
def json_response(obj: Dict[str, Any], status: int = 200):
    resp = make_response(json.dumps(obj, ensure_ascii=False), status)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp
_json_response = json_response

def _safe_float(v, dv=0.0):
    try:
        return float(v)
    except Exception:
        return dv

def _safe_json_loads_tags(s) -> List[str]:
    """把 CSV 里可能写坏的 tags 字段安全解析成 list[str]。"""
    if s is None:
        return []
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    s = str(s).strip()
    if not s:
        return []
    # 典型 JSON
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{")):
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return []
        except Exception:
            pass
    # 非 JSON，当作分隔字符串
    parts = [p.strip() for p in re.split(r"[,;/\s]+", s) if p.strip()]
    return parts

def _read_jobs_safe() -> List[Dict[str, Any]]:
    jobs = []
    if not os.path.exists(JOBS_CSV_PATH):
        return jobs
    with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "id": (row.get("id") or str(uuid.uuid4())).strip(),
                "title": (row.get("title") or "").strip(),
                "company": (row.get("company") or "").strip(),
                "location": (row.get("location") or "").strip(),
                "type": (row.get("type") or "full").strip(),            # full/part/project
                "work_mode": (row.get("work_mode") or "onsite").strip(),# onsite/hybrid/remote
                "jd_url": (row.get("jd_url") or "").strip(),
                "salary_min": _safe_float(row.get("salary_min"), 0.0),
                "salary_max": _safe_float(row.get("salary_max"), 0.0),
                "currency": (row.get("currency") or "CNY").strip(),
                "posted_at": (row.get("posted_at") or "").strip(),
                "source": (row.get("source") or "import").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
            item["tags"] = _safe_json_loads_tags(row.get("tags"))
            jobs.append(item)
    return jobs

def _append_jobs_to_csv(items: List[Dict[str, Any]]) -> int:
    """追加写入 CSV，同时按 jd_url 去重。"""
    if not items:
        return 0

    fieldnames = [
        "id", "title", "company", "location", "type", "work_mode", "jd_url",
        "salary_min", "salary_max", "currency", "posted_at", "source", "tags", "notes"
    ]

    # 读取现有去重集合
    existed_urls = set()
    if os.path.exists(JOBS_CSV_PATH):
        with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                url = (r.get("jd_url") or "").strip()
                if url:
                    existed_urls.add(url)

    write_items = []
    for it in items:
        url = (it.get("jd_url") or "").strip()
        if (not url) or (url in existed_urls):
            continue
        existed_urls.add(url)
        row = {
            "id": it.get("id") or str(uuid.uuid4()),
            "title": it.get("title", ""),
            "company": it.get("company", ""),
            "location": it.get("location", ""),
            "type": it.get("type", "full"),
            "work_mode": it.get("work_mode", "onsite"),
            "jd_url": url,
            "salary_min": _safe_float(it.get("salary_min"), 0.0),
            "salary_max": _safe_float(it.get("salary_max"), 0.0),
            "currency": it.get("currency", "CNY"),
            "posted_at": it.get("posted_at", ""),
            "source": it.get("source", "workday"),
            "tags": json.dumps(it.get("tags") or [], ensure_ascii=False),
            "notes": it.get("notes", ""),
        }
        write_items.append(row)

    if not write_items:
        return 0

    exists = os.path.exists(JOBS_CSV_PATH)
    with open(JOBS_CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in write_items:
            writer.writerow(row)
    return len(write_items)

# ------------------------------------------------------------------------------
# Workday 抓取
# ------------------------------------------------------------------------------
WD_ZONES = ["wd1", "wd2", "wd3", "wd5", "wd8"]

def detect_workday_zone(tenant: str, site: str, timeout=8) -> str | None:
    headers = {
        "Content-Type": "application/json",
        "Origin": f"https://{tenant}.myworkdayjobs.com",
        "Referer": f"https://{tenant}.myworkdayjobs.com/{site}",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    body = {"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""}

    for zone in WD_ZONES:
        url = f"https://{tenant}.{zone}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
            if r.status_code == 200 and "jobPostings" in r.text:
                return zone
        except Exception:
            pass
    return None

def fetch_workday_jobs(tenant: str, site: str, max_pages=30, page_size=50) -> List[Dict[str, Any]]:
    zone = detect_workday_zone(tenant, site)
    if not zone:
        return []

    headers = {
        "Content-Type": "application/json",
        "Origin": f"https://{tenant}.{zone}.myworkdayjobs.com",
        "Referer": f"https://{tenant}.{zone}.myworkdayjobs.com/{site}",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    base = f"https://{tenant}.{zone}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"

    results = []
    for p in range(max_pages):
        body = {"appliedFacets": {}, "limit": page_size, "offset": p * page_size, "searchText": ""}
        r = requests.post(base, json=body, headers=headers, timeout=15)
        if r.status_code != 200:
            break
        data = r.json()
        postings = data.get("jobPostings") or []
        if not postings:
            break

        for j in postings:
            title = (j.get("title") or "").strip()
            locs = j.get("locations") or []
            location = " / ".join([l.get("descriptor", "").strip() for l in locs if l.get("descriptor")])
            external_path = j.get("externalPath") or ""
            jd_url = f"https://{tenant}.{zone}.myworkdayjobs.com/{site}{external_path}"

            results.append({
                "id": str(uuid.uuid4()),
                "title": title,
                "company": tenant,
                "location": location,
                "type": "full",
                "work_mode": "onsite",
                "jd_url": jd_url,
                "salary_min": 0,
                "salary_max": 0,
                "currency": "CNY",
                "posted_at": "",
                "source": "workday",
                "tags": [],
                "notes": "",
            })
    return results

# ------------------------------------------------------------------------------
# LLM 客户端（DeepSeek/OpenAI 兼容）
# ------------------------------------------------------------------------------
def llm_available() -> bool:
    return bool(LLM_API_KEY and LLM_BASE_URL)

def call_llm(messages: List[Dict[str, str]], model: str = None,
             temperature: float = 0.2, max_tokens: int = 1200, timeout: int = 60) -> str:
    """
    返回纯文本（失败时返回 ""，不抛异常）
    """
    if not llm_available():
        return ""

    url = f"{LLM_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return ""
    except Exception:
        return ""

def call_llm_json(messages: List[Dict[str, str]], model: str = None,
                  temperature: float = 0.2, max_tokens: int = 1200, timeout: int = 60) -> Any:
    """
    让 LLM 输出 JSON：失败/解析失败则返回 None（不抛异常）
    """
    content = call_llm(messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    if not content:
        return None
    # 提取 JSON
    m = re.search(r"\{[\s\S]+\}", content)
    if not m:
        m = re.search(r"\[[\s\S]+\]", content)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ------------------------------------------------------------------------------
# 健康检查 & 版本
# ------------------------------------------------------------------------------
@app.route("/health")
def health():
    return json_response({"status": "ok", "time": int(time.time())})

@app.route("/version")
def version():
    return json_response({"version": VERSION})

# ------------------------------------------------------------------------------
# 静态文件 / 首页 / 管理页
# ------------------------------------------------------------------------------
@app.route("/public/<path:filename>")
def public_files(filename):
    return send_from_directory("public", filename)

@app.route("/")
def index():
    return send_from_directory("public", "agent.html")

@app.route("/admin")
def admin_shortcut():
    return send_from_directory("public", "admin.html")

# ------------------------------------------------------------------------------
# 职位导入（Workday）
# ------------------------------------------------------------------------------
def _save_jobs(items):
    # 若你的文件里已经有 _save_jobs，保留你原来的，别重复定义
    import csv, os
    os.makedirs(os.path.dirname(JOBS_CSV_PATH), exist_ok=True)
    exists = os.path.exists(JOBS_CSV_PATH)
    with open(JOBS_CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id","title","company","location","type","work_mode","salary_min",
            "salary_max","currency","jd_url","notes","posted_at","source","tags"
        ])
        if not exists:
            w.writeheader()
        for it in items:
            it["tags"] = ",".join(it.get("tags") or [])
            w.writerow(it)

def _ingest_greenhouse(slug: str):
    import requests, uuid
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    items = []
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return []
        data = r.json() or {}
        for j in data.get("jobs", []):
            items.append({
                "id": str(uuid.uuid4()),
                "title": j.get("title") or "",
                "company": slug.upper(),
                "location": (j.get("location") or {}).get("name") or "",
                "type": "full",
                "work_mode": "",
                "salary_min": "", "salary_max": "", "currency": "CNY",
                "jd_url": j.get("absolute_url") or "",
                "notes": "",
                "posted_at": (j.get("updated_at") or j.get("created_at") or "")[:10],
                "source": "greenhouse",
                "tags": [d.get("name") for d in (j.get("departments") or []) if d.get("name")]
            })
    except Exception:
        return []
    if items: _save_jobs(items)
    return items

def _ingest_lever(slug: str):
    import requests, uuid, datetime as dt
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    items = []
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200: return []
        data = r.json() or []
        for j in data:
            loc=""
            if isinstance(j.get("categories"), dict):
                loc = j["categories"].get("location") or ""
            posted=""
            if j.get("createdAt"):
                try:
                    posted = dt.datetime.fromtimestamp(int(j["createdAt"])/1000).strftime("%Y-%m-%d")
                except Exception:
                    posted=""
            items.append({
                "id": str(uuid.uuid4()),
                "title": j.get("text") or j.get("title") or "",
                "company": slug.upper(),
                "location": loc,
                "type": "full",
                "work_mode": "",
                "salary_min": "", "salary_max": "", "currency": "CNY",
                "jd_url": j.get("hostedUrl") or j.get("applyUrl") or "",
                "notes": "",
                "posted_at": posted,
                "source": "lever",
                "tags": [t for t in (j.get("tags") or []) if t]
            })
    except Exception:
        return []
    if items: _save_jobs(items)
    return items

@app.post("/api/ingest_ats")
def api_ingest_ats():
    """
    JSON 示例：
    {
      "greenhouse": ["airbnb","stripe"],
      "lever": ["scaleai"],
      "token": "<与服务端 ADMIN_TOKEN 相同，可留空>"
    }
    Header 也可用 X-Admin-Token 传 token。
    """
    data = request.get_json(silent=True) or {}

    header_token = (request.headers.get("X-Admin-Token") or "").strip()
    body_token   = (data.get("token") or data.get("admin_token") or "").strip()
    client_token = header_token or body_token
    if ADMIN_TOKEN and client_token != ADMIN_TOKEN:
        return _json_response({"ok": False, "error": "unauthorized"}, 401)

    gh_slugs = [s.strip() for s in (data.get("greenhouse") or []) if s and s.strip()]
    lv_slugs = [s.strip() for s in (data.get("lever") or []) if s and s.strip()]

    total, detail = 0, []
    for s in gh_slugs:
        got = _ingest_greenhouse(s)
        total += len(got); detail.append({"source":"greenhouse","slug":s,"count":len(got)})
    for s in lv_slugs:
        got = _ingest_lever(s)
        total += len(got); detail.append({"source":"lever","slug":s,"count":len(got)})

    return _json_response({"ok": True, "ingested": total, "detail": detail})


# ------------------------------------------------------------------------------
# 职位列表（分页/筛选）
# ------------------------------------------------------------------------------
@app.get("/api/jobs")
def api_jobs():
    page = int(request.args.get("page", 1) or 1)
    page_size = int(request.args.get("page_size", 20) or 20)
    q = (request.args.get("q") or "").strip().lower()
    type_ = (request.args.get("type") or "").strip().lower()
    mode = (request.args.get("work_mode") or "").strip().lower()
    loc_text = (request.args.get("location") or "").strip()
    locs = [x.strip().lower() for x in re.split(r"[,\s]+", loc_text) if x.strip()]

    items = _read_jobs_safe()

    def match(job):
        if type_ and job.get("type", "").lower() != type_:
            return False
        if mode and job.get("work_mode", "").lower() != mode:
            return False
        if locs:
            jloc = job.get("location", "").lower()
            if not any(l in jloc for l in locs):
                return False
        if q:
            txt = " ".join([
                job.get("title",""),
                job.get("company",""),
                job.get("location",""),
                " ".join(job.get("tags") or [])
            ]).lower()
            if q not in txt:
                return False
        return True

    items = [j for j in items if match(j)]
    total = len(items)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    page_items = items[start:end]

    return json_response({
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": page_items
    })

# ------------------------------------------------------------------------------
# 简历上传解析（PDF/DOCX/TXT）
# ------------------------------------------------------------------------------
def extract_text_from_filestorage(fs) -> str:
    filename = (fs.filename or "").lower()
    raw = fs.read()
    fs.stream.seek(0)

    if filename.endswith(".txt"):
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    if filename.endswith(".pdf"):
        if not pdf_extract_text:
            return "[解析失败] 服务器未安装 pdfminer.six"
        try:
            return pdf_extract_text(fs.stream) or ""
        except Exception:
            return ""

    if filename.endswith(".docx"):
        if not docx:
            return "[解析失败] 服务器未安装 python-docx"
        try:
            doc = docx.Document(fs.stream)
            return "\n".join([p.text for p in doc.paragraphs]) or ""
        except Exception:
            return ""

    # 其它后缀按文本兜底
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode("latin-1", errors="ignore")

@app.post("/api/upload_resume")
def api_upload_resume():
    if "file" not in request.files:
        return json_response({"ok": False, "error": "no_file"}, 400)
    fs = request.files["file"]
    text = extract_text_from_filestorage(fs)
    text = (text or "").strip()
    return json_response({"ok": True, "text": text[:20000]})

# ------------------------------------------------------------------------------
# LLM 画像（可退化）
# ------------------------------------------------------------------------------
@app.post("/api/profile")
def api_profile():
    data = request.get_json(silent=True) or {}
    resume_text = (data.get("resume_text") or "").strip()
    if not resume_text:
        return json_response({"ok": False, "error": "empty_resume"}, 400)

    if not llm_available():
        # 退化：简单提取前几行 + 关键词
        words = [w.lower() for w in re.findall(r"[a-zA-Z#\+\-]{2,}", resume_text)]
        kws = []
        for w in words:
            if w not in kws:
                kws.append(w)
            if len(kws) >= 20:
                break
        return json_response({
            "ok": True,
            "profile": {
                "summary": resume_text[:500],
                "strengths": [],
                "weaknesses": [],
                "roles": [],
                "keywords": kws
            }
        })

    system = "你是资深猎头，请从简历中精准总结画像。以 JSON 输出：summary, strengths(3-5), weaknesses(2-4), roles(5-8), keywords(10-20)。"
    user = f"简历：\n{resume_text}\n请用简体中文输出 JSON。"
    js = call_llm_json(
        [{"role":"system","content":system},{"role":"user","content":user}],
        model=MODEL_NAME, temperature=0.2, max_tokens=1200
    )
    if not js or not isinstance(js, dict):
        js = {"summary": resume_text[:500], "strengths": [], "weaknesses": [], "roles": [], "keywords": []}
    return json_response({"ok": True, "profile": js})

# ------------------------------------------------------------------------------
# LLM 对职位打分（可退化）
# ------------------------------------------------------------------------------
@app.post("/api/llm_score")
def api_llm_score():
    """
    body:
    {
      "resume_text": "...",
      "jobs": [{title, company, location, jd_url, tags}],  # 建议 <= 10 条
    }
    """
    data = request.get_json(silent=True) or {}
    resume_text = (data.get("resume_text") or "").strip()
    jobs = data.get("jobs") or []
    if not resume_text or not isinstance(jobs, list) or not jobs:
        return json_response({"ok": False, "error": "bad_request"}, 400)

    # 退化：本地粗评分
    def rough_score(j):
        text = " ".join([j.get("title",""), j.get("company",""), " ".join(j.get("tags") or [])]).lower()
        words = [w for w in re.findall(r"[a-zA-Z#\+\-]{2,}", resume_text.lower())]
        s = 0
        for w in words[:30]:
            if w in text:
                s += 1
        return min(100, s*3)

    if not llm_available():
        for j in jobs:
            j["llm_score"] = rough_score(j)
        return json_response({"ok": True, "items": jobs})

    # LLM 版本：让模型对列表逐条打 0-100 分，并返回 JSON
    desc_list = []
    for idx, j in enumerate(jobs, 1):
        desc_list.append(f"{idx}. {j.get('title','')} | {j.get('company','')} | {j.get('location','')} | tags={j.get('tags',[])}")

    system = "你是专业招聘顾问，请基于候选人简历内容评估职位匹配度（0-100），仅给出 JSON 数组，如：[{" \
             "index:1, score:85, reason:\"...\"}, ...]。分数含义：>=85 强匹配；70-84 可考虑；<70 不匹配。"
    user = f"候选人简历：\n{resume_text}\n待评估职位：\n" + "\n".join(desc_list)

    out = call_llm_json(
        [{"role":"system","content":system},{"role":"user","content":user}],
        model=MODEL_NAME, temperature=0.2, max_tokens=1500
    )
    if not isinstance(out, list):
        # 回退粗评分
        for j in jobs:
            j["llm_score"] = rough_score(j)
        return json_response({"ok": True, "items": jobs})

    # 合并回原 jobs
    for item in out:
        try:
            idx = int(item.get("index", 0)) - 1
            if 0 <= idx < len(jobs):
                jobs[idx]["llm_score"] = int(item.get("score", 0))
                jobs[idx]["llm_reason"] = item.get("reason", "")
        except Exception:
            pass

    # 兜底为粗评分
    for j in jobs:
        if j.get("llm_score") is None:
            j["llm_score"] = rough_score(j)

    return json_response({"ok": True, "items": jobs})

# ------------------------------------------------------------------------------
# 综合：分析并推荐（先粗筛，再可选 LLM 精评 topN）
# ------------------------------------------------------------------------------
@app.post("/api/analyze_recommend")
def api_analyze_recommend():
    data = request.get_json(silent=True) or {}
    resume_text = (data.get("resume_text") or data.get("text") or "").strip().lower()
    kw_text     = (data.get("keywords") or "").strip().lower()
    loc_text    = (data.get("location") or "").strip().lower()
    type_       = (data.get("type") or "").strip().lower()
    mode        = (data.get("work_mode") or "").strip().lower()
    use_llm     = bool(data.get("use_llm") or False)
    topn        = int(data.get("topn") or 10)

    kws = [k for k in re.split(r"[,;/\s]+", kw_text) if k]
    if resume_text:
        for t in re.findall(r"[a-zA-Z\-#\+]{2,}", resume_text):
            if len(kws) >= 32:  # 控制上限
                break
            t = t.lower()
            if t not in kws:
                kws.append(t)

    jobs = _read_jobs_safe()

    def base_filter(j):
        if type_ and j.get("type","").lower() != type_:
            return False
        if mode and j.get("work_mode","").lower() != mode:
            return False
        if loc_text:
            if loc_text not in (j.get("location","").lower()):
                return False
        return True

    cand = [j for j in jobs if base_filter(j)]

    def rough_score(j):
        text = " ".join([j.get("title",""), j.get("company",""), " ".join(j.get("tags") or [])]).lower()
        score = 0
        for k in kws:
            if k and k in text:
                score += 1
        # 简单增加：标题里直接命中加权
        if any(k in (j.get("title","").lower()) for k in kws):
            score += 2
        return score

    scored = []
    for j in cand:
        rs = rough_score(j)
        scored.append({**j, "rough_score": rs})

    scored.sort(key=lambda x: (x.get("rough_score") or 0), reverse=True)

    # 可选：对 TopN 做 LLM 精评
    if use_llm and llm_available() and scored:
        top = scored[:max(1, min(topn, 10))]
        payload_jobs = [{
            "title": t.get("title",""),
            "company": t.get("company",""),
            "location": t.get("location",""),
            "jd_url": t.get("jd_url",""),
            "tags": t.get("tags") or []
        } for t in top]
        try:
            r = app.test_client().post("/api/llm_score", json={"resume_text": resume_text, "jobs": payload_jobs})
            if r.status_code == 200:
                items = r.get_json().get("items") or []
                # 按顺序写回
                for i, it in enumerate(items):
                    scored[i]["llm_score"] = it.get("llm_score")
                    scored[i]["llm_reason"] = it.get("llm_reason")
        except Exception:
            pass

    return json_response({"items": scored[:200]})

# ------------------------------------------------------------------------------
# 404 兜底
# ------------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    try:
        return send_from_directory("public", "agent.html")
    except Exception:
        return json_response({"error": "not_found"}, 404)

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
