# app.py — LGWORK 后端（整文件覆盖版）
# -*- coding: utf-8 -*-

import os, re, csv, io, json, uuid, time, math
from datetime import datetime
from urllib.parse import urlparse
import requests
from flask import Flask, request, jsonify, send_from_directory, make_response

# =========================
# 环境变量
# =========================
JOBS_CSV_PATH      = os.environ.get("JOBS_CSV_PATH", "data/jobs.csv")
TRACKER_JSON_PATH  = os.environ.get("TRACKER_JSON_PATH", "data/tracker.json")
ADMIN_TOKEN        = os.environ.get("ADMIN_TOKEN", "")

LLM_BASE_URL       = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY        = os.environ.get("LLM_API_KEY", "")
MODEL_NAME         = os.environ.get("MODEL_NAME", "deepseek-chat")

# Bright Data
BRIGHTDATA_API_TOKEN    = os.environ.get("BRIGHTDATA_API_TOKEN", "")
BRIGHTDATA_COLLECTOR_ID = os.environ.get("BRIGHTDATA_COLLECTOR_ID", "")

os.makedirs(os.path.dirname(JOBS_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TRACKER_JSON_PATH), exist_ok=True)

app = Flask(__name__)

# =========================
# 通用工具
# =========================
def json_response(obj, status=200):
    r = make_response(jsonify(obj), status)
    r.headers["Content-Type"] = "application/json; charset=utf-8"
    return r

def _now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _normalize_text(s):
    return (s or "").replace("\u3000"," ").replace("\xa0"," ").strip()

def _ensure_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])

def _csv_headers():
    return ["id","title","company","source","url","location","posted_at","tags","ingested_at"]

def _append_jobs_to_csv(items):
    exists = set()
    if os.path.exists(JOBS_CSV_PATH):
        with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                u = (row.get("url") or "").strip()
                if u:
                    exists.add(u)

    write_header = not os.path.exists(JOBS_CSV_PATH) or os.path.getsize(JOBS_CSV_PATH)==0
    wrote = 0
    if items:
        with open(JOBS_CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_csv_headers())
            if write_header:
                w.writeheader()
            for it in items:
                u = _normalize_text(it.get("url",""))
                if not u or u in exists:
                    continue
                row = {
                    "id": _normalize_text(it.get("id") or str(uuid.uuid4())),
                    "title": _normalize_text(it.get("title")),
                    "company": _normalize_text(it.get("company")),
                    "source": _normalize_text(it.get("source") or "unknown"),
                    "url": u,
                    "location": _normalize_text(it.get("location")),
                    "posted_at": _normalize_text(it.get("posted_at")),
                    "tags": ";".join(_ensure_list(it.get("tags"))),
                    "ingested_at": _now_iso(),
                }
                w.writerow(row)
                wrote += 1
                exists.add(u)
    return wrote

def _read_jobs():
    jobs = []
    if not os.path.exists(JOBS_CSV_PATH):
        return jobs
    with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["tags"] = [x for x in (row.get("tags") or "").split(";") if x]
            jobs.append(row)
    return jobs

# =========================
# LLM （带本地回退）
# =========================
def _local_summary(text):
    lines = [x for x in _normalize_text(text).splitlines() if x][:6]
    head = " · ".join([x[:90] for x in lines])[:480]
    return "候选人概述（离线）："+ (head or "具备相关教育背景与项目经验，沟通协作能力良好。")

def llm_chat(sys_prompt, user_prompt, max_tokens=600, temperature=0.2):
    if not LLM_API_KEY:
        return _local_summary(user_prompt)
    try:
        resp = requests.post(
            LLM_BASE_URL.rstrip("/") + "/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role":"system","content": sys_prompt},
                    {"role":"user","content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            },
            timeout=60
        )
        if resp.status_code != 200:
            return _local_summary(user_prompt)
        data = resp.json()
        content = (data.get("choices",[{}])[0].get("message",{}).get("content","") or "").strip()
        return content or _local_summary(user_prompt)
    except Exception:
        return _local_summary(user_prompt)

# =========================
# 简历解析（PDF/DOCX/TXT）
# =========================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    from pdfminer.high_level import extract_text
    try:
        with io.BytesIO(file_bytes) as bio:
            return extract_text(bio) or ""
    except Exception:
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        import docx
        with io.BytesIO(file_bytes) as bio:
            doc = docx.Document(bio)
            return "\n".join([p.text for p in doc.paragraphs if p.text]) or ""
    except Exception:
        return ""

def extract_text_from_plain(file_bytes: bytes, encoding="utf-8") -> str:
    try:
        return file_bytes.decode(encoding, errors="ignore")
    except Exception:
        for enc in ("gbk","gb2312","big5","latin-1"):
            try:
                return file_bytes.decode(enc, errors="ignore")
            except Exception:
                continue
    return ""

# =========================
# 匹配：关键词/地点/标题 等启发式打分
# =========================
def _tokenize_keywords(s):
    s = _normalize_text(s).lower()
    tokens = re.split(r"[^a-z0-9+\u4e00-\u9fa5]+", s)
    return [t for t in tokens if len(t)>=2]

def _score_job(resume_text, job):
    # 关键词 TF-like 命中 + title/公司/地点轻权重
    r_tokens = set(_tokenize_keywords(resume_text))
    score = 0
    title = (job.get("title") or "").lower()
    comp  = (job.get("company") or "").lower()
    loc   = (job.get("location") or "").lower()

    # 关键词对 title 的命中加权更高
    for t in r_tokens:
        if not t: continue
        if t in title: score += 6
        if t in comp:  score += 2
        if t in loc:   score += 1

    # tags 适当加分
    for tg in job.get("tags", []):
        tg = (tg or "").lower()
        if tg in r_tokens:
            score += 1

    # 基本线性缩放
    return max(0, min(100, score))

# =========================
# 路由：静态 & 健康
# =========================
@app.route("/")
def home():
    return send_from_directory("public", "agent.html")

@app.route("/admin")
def admin_page():
    return send_from_directory("public", "admin.html")

@app.route("/public/<path:p>")
def pub(p):
    return send_from_directory("public", p)

@app.route("/healthz")
def health():
    return "ok",200

# =========================
# API：简历上传 → 文本
# =========================
@app.route("/api/upload_resume", methods=["POST"])
def api_upload_resume():
    f = request.files.get("file")
    if not f:
        return json_response({"ok": False, "error": "no_file"}, 400)
    data = f.read() or b""
    name = (f.filename or "").lower()

    text = ""
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(data)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(data)
    else:
        text = extract_text_from_plain(data)

    text = _normalize_text(text)
    if not text:
        return json_response({"ok": False, "error":"empty_text"}, 400)
    return json_response({"ok": True, "text": text})

# =========================
# API：画像（第三人称摘要 + 关键词）
# =========================
@app.route("/api/profile", methods=["POST"])
def api_profile():
    data = request.get_json(silent=True) or {}
    resume_text = _normalize_text(data.get("resume_text") or "")
    if not resume_text:
        return json_response({"ok": False, "error":"missing resume_text"}, 400)

    sys = "你是资深人才顾问，请用第三人称为候选人写一段150~250字的专业概述，口吻客观、凝练，适合放在简历最上方。最后输出3~6个核心关键词（中文或英文均可）。"
    user = resume_text[:6000]
    summary = llm_chat(sys, user, max_tokens=320)

    # 简单从摘要里提取关键词（如果模型没有给，我们就fallback）
    kws = re.findall(r"[#\u25CF\u2022\-•]\s*([A-Za-z0-9\u4e00-\u9fa5\-\+\./ ]{2,40})", summary)
    if not kws:
        # fallback：基于简历前500字粗提
        tokens = _tokenize_keywords(resume_text[:1000])
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t,0)+1
        kws = [x for x,_ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:6]]

    return json_response({
        "ok": True,
        "summary": summary.strip(),
        "keywords": kws[:8]
    })

# =========================
# API：生成职位推荐（匹配 + 可选 LLM 调优）
# =========================
@app.route("/api/analyze_recommend", methods=["POST"])
def api_analyze_recommend():
    data = request.get_json(silent=True) or {}
    resume_text = _normalize_text(data.get("resume_text") or "")
    if not resume_text:
        return json_response({"ok": True, "items": []})

    page_size = int(data.get("page_size") or 50)
    jobs = _read_jobs()
    # 基于启发式分数
    scored = []
    for j in jobs:
        s = _score_job(resume_text, j)
        if s <= 0: 
            continue
        it = dict(j)
        it["match_score"] = s
        it["llm_score"] = None
        scored.append(it)
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return json_response({"ok": True, "items": scored[:page_size]})

# =========================
# API：一批职位做“精评/打分”
# =========================
@app.route("/api/match_batch", methods=["POST"])
def api_match_batch():
    data = request.get_json(silent=True) or {}
    resume_text = _normalize_text(data.get("resume_text") or "")
    ids = _ensure_list(data.get("ids"))
    if not resume_text or not ids:
        return json_response({"ok": True, "scores": {}})

    # 为保证响应及时，先用本地启发式；如果配置了 LLM，也可进一步完善（演示期不建议全开）
    all_jobs = {j["id"]: j for j in _read_jobs()}
    out = {}
    for jid in ids:
        j = all_jobs.get(jid)
        if not j:
            continue
        out[jid] = _score_job(resume_text, j)

    return json_response({"ok": True, "scores": out})

# =========================
# API：分页获取职位
# =========================
@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    page = int(request.args.get("page") or 1)
    page_size = int(request.args.get("page_size") or 20)
    location = _normalize_text(request.args.get("location") or "")
    src = _normalize_text(request.args.get("source") or "")

    jobs = _read_jobs()
    if location:
        jobs = [j for j in jobs if location.lower() in (j.get("location","").lower())]
    if src:
        jobs = [j for j in jobs if src.lower() in (j.get("source","").lower())]

    total = len(jobs)
    start = max(0, (page-1)*page_size)
    end   = start + page_size
    return json_response({"ok": True, "total": total, "items": jobs[start:end]})

# =========================
# ATS 采集（Greenhouse / Lever / Workday）
# =========================

# --- Workday 解析（增强） ---
# ===== Workday robust helpers & fetcher (drop-in replacement) =====
import re, json, time
from urllib.parse import urlparse
import requests

def _http_json(method, url, json_body=None, headers=None, timeout=15):
    h = {
        "accept": "application/json, text/plain, */*",
        "content-type": "application/json;charset=UTF-8",
        "origin": f"{urlparse(url).scheme}://{urlparse(url).netloc}",
        "referer": f"{urlparse(url).scheme}://{urlparse(url).netloc}/",
        # Workday/CloudFront 对 UA 比较敏感，给一个常见桌面 UA
        "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        "accept-language": "en-US,en;q=0.9",
    }
    if headers:
        h.update(headers)
    r = requests.request(method, url, json=json_body, headers=h, timeout=timeout)
    r.raise_for_status()
    return r.json()

def parse_workday_line(line: str):
    """
    支持三种输入：
    1) 完整 URL（推荐）： https://beigene.wd5.myworkdayjobs.com/en-US/BeiGene
    2) 域名或带路径：      beigene.wd5.myworkdayjobs.com/en-US/BeiGene
    3) 简写 tenant/site：  beigene/BeiGene
    返回: (tenant, site, base_url, referer)
    """
    s = line.strip()
    if not s:
        return None

    # 完整 URL 或省略 scheme 的域名
    if "myworkdayjobs.com" in s:
        if not s.startswith("http"):
            s = "https://" + s
        u = urlparse(s)
        host = u.netloc  # e.g. beigene.wd5.myworkdayjobs.com
        m = re.match(r"^([a-z0-9\-]+)\.wd\d+\.myworkdayjobs\.com$", host)
        tenant = m.group(1) if m else host.split(".")[0]
        # 可能有 /en-US/<site> 或 /{locale}/{site}
        parts = [p for p in u.path.split("/") if p]
        site = None
        if len(parts) >= 2:
            # 形如 en-US/BeiGene -> 取最后一个
            site = parts[-1]
        base = f"{u.scheme}://{host}"
        referer = s
        return tenant, site, base, referer

    # 简写：tenant/site
    if "/" in s:
        tenant, site = [x.strip() for x in s.split("/", 1)]
        base = None  # 下面会推断
        referer = None
        return tenant, site, base, referer

    # 只有 tenant
    tenant = s
    return tenant, None, None, None

def fetch_workday_jobs(line: str, max_pages=50, page_size=50):
    """
    line 是 admin 页面里每行的输入（支持 URL / 域名 / tenant/site）。
    返回标准化 job 列表（title, company, location, url, source 等）。
    """
    parsed = parse_workday_line(line)
    if not parsed:
        return []

    tenant, site, base, referer = parsed

    # 没有 base 就用常规推断
    if not base:
        # wd 区号一般不影响 cxs 路径；默认 wd5，若不通再由 CloudFront 跳转
        base = f"https://{tenant}.wd5.myworkdayjobs.com"
    if not referer:
        referer = f"{base}/"

    jobs = []
    offsets = range(0, max_pages * page_size, page_size)

    # Workday 的三种常见路径，按顺序尝试，直到某条成功拿到数据
    # 1) /wday/cxs/{tenant}/careers/jobs
    # 2) /wday/cxs/{tenant}/career-site/jobs
    # 3) /wday/cxs/{tenant}/{site}/jobs  (此条需带 site)
    path_candidates = [
        f"/wday/cxs/{tenant}/careers/jobs",
        f"/wday/cxs/{tenant}/career-site/jobs",
    ]
    if site:
        path_candidates.append(f"/wday/cxs/{tenant}/{site}/jobs")

    headers = {"referer": referer}
    found_any = False

    for path in path_candidates:
        try:
            # 先探测第一页
            url = base + path
            body = {"limit": page_size, "offset": 0, "searchText": ""}
            data = _http_json("POST", url, json_body=body, headers=headers)

            # Workday 响应结构也不统一，做兼容
            def normalize_list(payload):
                if isinstance(payload, dict):
                    if "jobPostings" in payload:
                        return payload["jobPostings"]
                    if "items" in payload:
                        return payload["items"]
                    if "data" in payload and isinstance(payload["data"], dict):
                        if "jobPostings" in payload["data"]:
                            return payload["data"]["jobPostings"]
                        if "items" in payload["data"]:
                            return payload["data"]["items"]
                return []

            first_items = normalize_list(data)
            if not first_items:
                # 这条 path 没数据，换下一条
                continue

            found_any = True

            def to_job(item):
                # 兼容不同字段名
                title = item.get("title") or item.get("titleLocalized") or ""
                loc = (
                    item.get("locationsText")
                    or item.get("locations", [{}])[0].get("displayName")
                    or item.get("location")
                    or ""
                )
                company = tenant
                # externalPath 通常是 "/{site}/job/xxx"
                ext = item.get("externalPath") or item.get("externalUrl") or ""
                # 少数站点返回的是 careerSiteId + jobPostingId，需要兜底构造
                if ext:
                    job_url = base + ext
                else:
                    jid = (
                        item.get("jobPostingId")
                        or item.get("id")
                        or item.get("bulletFields", [{}])[0].get("jobId")
                    )
                    if site and jid:
                        job_url = f"{base}/{site}/job/{jid}"
                    elif jid:
                        job_url = f"{base}/job/{jid}"
                    else:
                        job_url = base

                return {
                    "title": title.strip(),
                    "company": company,
                    "location": loc.strip(),
                    "url": job_url,
                    "source": "workday",
                }

            # 分页拉取
            for offset in offsets:
                body = {"limit": page_size, "offset": offset, "searchText": ""}
                page = _http_json("POST", url, json_body=body, headers=headers)
                items = normalize_list(page)
                if not items:
                    break
                for it in items:
                    jobs.append(to_job(it))

                # 某些返回包含 total/hasMore，可提前终止
                if isinstance(page, dict):
                    total = page.get("total")
                    has_more = page.get("hasMore")
                    if isinstance(total, int) and offset + page_size >= total:
                        break
                    if has_more is False:
                        break

            # 这一条 path 成功了就不要再试其他 path
            break

        except Exception as e:
            # 换下一种 path
            # print(f"Workday path failed: {path} -> {e}")
            continue

    # 若三条 path 都没拿到，则返回空
    return jobs if found_any else []
# ===== end of Workday robust helpers ====
   

# --- Greenhouse/Lever ---
def fetch_greenhouse_jobs(slug, limit=800):
    slug=(slug or "").strip()
    if not slug:
        return []
    url=f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    try:
        r=requests.get(url,timeout=20)
        if r.status_code!=200:
            return []
        jobs=(r.json() or {}).get("jobs",[])[:limit]
        out=[]
        for j in jobs:
            out.append({
                "id": j.get("id") or str(uuid.uuid4()),
                "title": _normalize_text(j.get("title")),
                "company": slug,
                "source": "greenhouse",
                "url": j.get("absolute_url") or "",
                "location": (j.get("location") or {}).get("name") or "",
                "posted_at": "",
                "tags":[f"greenhouse:{slug}"]
            })
        return out
    except Exception:
        return []

def fetch_lever_jobs(slug, limit=800):
    slug=(slug or "").strip()
    if not slug:
        return []
    url=f"https://api.lever.co/v0/postings/{slug}?mode=json"
    try:
        r=requests.get(url,timeout=20)
        if r.status_code!=200:
            return []
        arr=r.json() or []
        out=[]
        for j in arr[:limit]:
            cats=j.get("categories") or {}
            loc = cats.get("location") or ""
            if isinstance(loc, list):
                loc=", ".join([x.get("name","") for x in loc])
            out.append({
                "id": j.get("id") or str(uuid.uuid4()),
                "title": _normalize_text(j.get("text")),
                "company": slug,
                "source": "lever",
                "url": j.get("hostedUrl") or "",
                "location": loc or "",
                "posted_at": j.get("createdAt") and datetime.utcfromtimestamp(j["createdAt"]/1000).strftime("%Y-%m-%d") or "",
                "tags":[f"lever:{slug}"]
            })
        return out
    except Exception:
        return []

# --- MokaHR（实验：单页列表接口） ---
def fetch_mokahr_jobs(list_url, limit=500):
    # 仅做演示：很多 mokahr 域名会做前端渲染 & 反爬。此处尽力抓可见结构，不保证全量。
    try:
        html = requests.get(list_url, timeout=20, headers={"User-Agent":"Mozilla/5.0"}).text
    except Exception:
        return []
    # 非严格解析：找 href 中的 jobId
    links = re.findall(r'href="([^"]+job[\w\-_/]*\d+[^"]*)"', html, flags=re.I)
    items=[]
    for href in links[:limit]:
        url = href if href.startswith("http") else re.sub(r"/+$","",list_url.split("#")[0]) + ("" if href.startswith("/") else "/") + href
        items.append({
            "id": str(uuid.uuid4()),
            "title": "职位",
            "company": urlparse(list_url).netloc.split(".")[0],
            "source": "mokahr",
            "url": url,
            "location": "",
            "posted_at": "",
            "tags": ["mokahr"]
        })
    return items

@app.route("/api/ingest_ats", methods=["POST"])
def api_ingest_ats():
    """
    JSON 示例：
    {
      "greenhouse_slugs": "airbnb, stripe, figma",
      "lever_slugs": "scaleai, databricks",
      "workday_lines": "beigene/BeiGene@wd5\npfizer/PfizerCareers@wd1\nhttps://roche.wd3.myworkdayjobs.com/roche",
      "admin_token": "..."
    }
    Header 也可传：X-Admin-Token
    """
    data = request.get_json(silent=True) or {}
    token = (request.headers.get("X-Admin-Token") or data.get("admin_token") or "").strip()
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return json_response({"ok": False, "error": "unauthorized"}, 401)

    gh_text = (data.get("greenhouse_slugs") or "").strip()
    lv_text = (data.get("lever_slugs") or "").strip()
    wd_text = (data.get("workday_lines") or data.get("workday_sites") or data.get("wd") or "").strip()

    gh_slugs = [x.strip() for x in re.split(r'[,;\s]+', gh_text) if x.strip()]
    lv_slugs = [x.strip() for x in re.split(r'[,;\s]+', lv_text) if x.strip()]
    wd_lines = [ln.strip() for ln in wd_text.splitlines() if ln.strip()]

    total = 0
    detail = {"greenhouse":0, "lever":0, "workday":0}

    # greenhouse
    for s in gh_slugs:
        try:
            items = fetch_greenhouse_jobs(s)
            total += len(items); detail["greenhouse"] += len(items)
            _append_jobs_to_csv(items)
        except Exception:
            continue

    # lever
    for s in lv_slugs:
        try:
            items = fetch_lever_jobs(s)
            total += len(items); detail["lever"] += len(items)
            _append_jobs_to_csv(items)
        except Exception:
            continue

    # workday
    for line in wd_lines:
        cfg = _parse_workday_line(line)
        if not cfg: 
            continue
        try:
            items = fetch_workday_jobs(cfg["tenant"], cfg["site"], cfg.get("cluster"))
            total += len(items); detail["workday"] += len(items)
            _append_jobs_to_csv(items)
        except Exception:
            continue

    return json_response({"ok": True, "ingested": total, "detail": detail})

@app.route("/api/ingest_cn_jobs", methods=["POST"])
def api_ingest_cn_jobs():
    """
    JSON:
    {
      "mokahr_urls": ["https://app.mokahr.com/social-recruitment/hengrui/145996#/"],
      "admin_token": "..."
    }
    """
    data = request.get_json(silent=True) or {}
    token = (request.headers.get("X-Admin-Token") or data.get("admin_token") or "").strip()
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return json_response({"ok": False, "error":"unauthorized"}, 401)

    urls = _ensure_list(data.get("mokahr_urls"))
    total=0
    detail={"mokahr":0}
    for u in urls:
        try:
            items=fetch_mokahr_jobs(u)
            total += len(items); detail["mokahr"] += len(items)
            _append_jobs_to_csv(items)
        except Exception:
            continue
    return json_response({"ok": True, "ingested": total, "detail": detail})

# =========================
# Bright Data 导入
# =========================
def _bd_headers():
    return {"Authorization": BRIGHTDATA_API_TOKEN, "Content-Type":"application/json"}

def _bd_trigger(collector_id, payload):
    url=f"https://api.brightdata.com/dca/trigger?collector_id={collector_id}"
    r=requests.post(url, headers=_bd_headers(), json=payload, timeout=30)
    if r.status_code!=200:
        raise RuntimeError(f"BD trigger error: {r.text}")
    return r.json().get("id")  # run id

def _bd_poll_dataset(run_id, wait_timeout=240):
    url=f"https://api.brightdata.com/dca/dataset?id={run_id}"
    t0=time.time()
    while True:
        r=requests.get(url, headers=_bd_headers(), timeout=30)
        if r.status_code==200 and r.headers.get("content-type","").startswith("application/json"):
            return r.json()
        if time.time()-t0>wait_timeout:
            raise RuntimeError("BD dataset timeout")
        time.sleep(3)

def _map_bd_item(x):
    title = x.get("title") or x.get("job_title") or x.get("position") or ""
    company = x.get("company") or x.get("employer") or x.get("org") or ""
    url = x.get("url") or x.get("job_url") or x.get("apply_url") or x.get("source_url") or ""
    loc = x.get("location") or x.get("job_location") or ""
    posted = x.get("posted_at") or x.get("date_posted") or x.get("published_at") or ""
    return {
        "id": x.get("id") or str(uuid.uuid4()),
        "title": _normalize_text(title),
        "company": _normalize_text(company),
        "source": "brightdata",
        "url": url,
        "location": _normalize_text(loc),
        "posted_at": _normalize_text(posted),
        "tags": ["brightdata"]
    }

@app.route("/api/ingest_brightdata", methods=["POST"])
def api_ingest_brightdata():
    """
    POST:
    { "collector_id":"...", "query":"biotech jobs", "country":"US", "limit":1000, "admin_token":"..." }
    或
    { "dataset_id":"...", "admin_token":"..." }
    """
    data = request.get_json(silent=True) or {}
    token = (request.headers.get("X-Admin-Token") or data.get("admin_token") or "").strip()
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return json_response({"ok": False, "error": "unauthorized"}, 401)
    if not BRIGHTDATA_API_TOKEN:
        return json_response({"ok": False, "error": "missing BRIGHTDATA_API_TOKEN"}, 400)

    # 直读 Dataset
    dataset_id = (data.get("dataset_id") or "").strip()
    if dataset_id:
        url=f"https://api.brightdata.com/datasets/v3/prime/{dataset_id}/data"
        r=requests.get(url, headers=_bd_headers(), timeout=60)
        if r.status_code!=200:
            return json_response({"ok": False, "error": r.text}, 500)
        raw=r.json() or []
        mapped=[_map_bd_item(x) for x in raw]
        wrote=_append_jobs_to_csv(mapped)
        return json_response({"ok": True, "ingested": wrote, "detail": {"brightdata": len(mapped)}})

    collector_id = (data.get("collector_id") or BRIGHTDATA_COLLECTOR_ID or "").strip()
    if not collector_id:
        return json_response({"ok": False, "error":"missing collector_id"}, 400)

    q = (data.get("query") or "biotech jobs").strip()
    country = (data.get("country") or "US").strip()
    limit = int(data.get("limit") or 500)

    payload={"input":{"query": q, "country": country, "limit": limit}}
    run_id=_bd_trigger(collector_id, payload)
    dataset=_bd_poll_dataset(run_id, wait_timeout=240)

    raw = dataset if isinstance(dataset, list) else (dataset.get("results") or dataset.get("data") or [])
    mapped=[_map_bd_item(x) for x in raw]
    wrote=_append_jobs_to_csv(mapped)
    return json_response({"ok": True, "ingested": wrote, "detail": {"brightdata": len(mapped)}})

# =========================
# 主入口
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
