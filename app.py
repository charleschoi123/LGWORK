import os
import re
import csv
import json
import uuid
import time
import threading
from datetime import datetime
from urllib.parse import unquote_plus

from flask import Flask, request, jsonify, make_response, send_from_directory
import requests
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

app = Flask(__name__)

# =========================
# Config
# =========================
JOBS_CSV_PATH     = os.environ.get("JOBS_CSV_PATH", "data/jobs.csv")
TRACKER_JSON_PATH = os.environ.get("TRACKER_JSON_PATH", "data/tracker.json")

DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL    = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

ADMIN_TOKEN       = os.environ.get("ADMIN_TOKEN", "")   # 采集接口可选保护
UPLOAD_DIR        = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

_tracker_lock = threading.Lock()
_csv_lock     = threading.Lock()


# =========================
# Utils
# =========================
def _parse_date(s):
    if not s: return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(s.strip(), fmt)
        except Exception: pass
    try: return datetime.fromisoformat(s.strip())
    except Exception: return None

def _safe_float(x):
    try: return float(x)
    except Exception: return None

def _ensure_jobs_csv(headers):
    os.makedirs(os.path.dirname(JOBS_CSV_PATH), exist_ok=True)
    if not os.path.exists(JOBS_CSV_PATH):
        with open(JOBS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

def _read_jobs():
    jobs = []
    if not os.path.exists(JOBS_CSV_PATH): return jobs
    with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "id": (row.get("id") or "").strip() or str(uuid.uuid4()),
                "title": (row.get("title") or "").strip(),
                "company": (row.get("company") or "").strip(),
                "location": (row.get("location") or "").strip(),
                "work_mode": (row.get("work_mode") or "").strip().lower(),  # onsite/hybrid/remote
                "type": (row.get("type") or "").strip().lower(),           # full/part/project
                "tags": [],
                "salary_min": _safe_float(row.get("salary_min")),
                "salary_max": _safe_float(row.get("salary_max")),
                "currency": (row.get("currency") or "").strip().upper(),
                "jd_url": (row.get("jd_url") or "").strip(),
                "posted_at": (row.get("posted_at") or "").strip(),
                "source": (row.get("source") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
            raw_tags = row.get("tags") or ""
            pieces = []
            for sep in [";", ","]:
                pieces += [t.strip() for t in raw_tags.split(sep)]
            item["tags"] = sorted(list({t for t in pieces if t}))

            dt = _parse_date(item["posted_at"])
            item["_posted_ts"] = int(dt.timestamp()) if dt else 0
            jobs.append(item)
    return jobs

def _extract_text_from_file(path: str, ext: str) -> str:
    ext = (ext or "").lower()
    text = ""
    if ext == ".pdf":
        text = pdf_extract_text(path) or ""
    elif ext == ".docx":
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    text = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
    return text[:80000]

def _allow_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Admin-Token"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PATCH"
    return resp

@app.after_request
def after_request(resp): return _allow_cors(resp)

def _json_response(data, status=200):
    resp = make_response(jsonify(data), status);  return _allow_cors(resp)

def _extract_json(text):
    if not text: return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").replace("json\n", "", 1).replace("json\r\n", "", 1)
    try:
        return json.loads(t.strip())
    except Exception:
        try:
            l = t.find("{"); r = t.rfind("}")
            if l != -1 and r != -1 and r > l: return json.loads(t[l:r+1])
        except Exception:
            return None
    return None

def _llm_unavailable():
    return {"error": "LLM 服务暂不可用，请稍后再试。"}

# ========= LLM =========
def _call_deepseek(messages, temperature=0.3, timeout=60):
    if not DEEPSEEK_API_KEY:
        return {"ok": False, "content": None}
    url = f"{DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": DEEPSEEK_MODEL, "messages": messages, "temperature": temperature}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        return {"ok": True, "content": content}
    except Exception:
        return {"ok": False, "content": None}

def _call_profile(resume_text: str):
    sys = (
        "You are a career analyst. Given a resume, output STRICT JSON with fields: "
        "{headline:string, summary:string(<=150 Chinese chars if zh, <=500 English chars if en), "
        "strengths:[], target_roles:[], industries:[], skills_core:[], skills_nice:[], locations:[], "
        "salary_hint:string, improvements:[], keywords:[10-25 tokens]} Return JSON only."
    )
    res = _call_deepseek([
        {"role":"system","content":sys},
        {"role":"user","content":f"RESUME:\n{resume_text}\nReturn JSON only."}
    ])
    if not res["ok"]:
        # 兜底：启发式摘要+关键词
        kws = _heuristic_keywords(resume_text, top_k=18)
        summary = _heuristic_summary(resume_text, kws)
        return {
            "headline":"", "summary": summary, "strengths":[], "target_roles":[],
            "industries":[], "skills_core":[], "skills_nice":[], "locations":[],
            "salary_hint":"", "improvements":[], "keywords": kws
        }
    parsed = _extract_json(res["content"]) or {}
    for k,v in {
        "headline":"", "summary":"", "strengths":[], "target_roles":[],
        "industries":[], "skills_core":[], "skills_nice":[], "locations":[],
        "salary_hint":"", "improvements":[], "keywords":[]
    }.items():
        parsed.setdefault(k,v)
    return parsed

def _call_match(resume_text: str, jd_text: str):
    sys = (
        "You are an expert job-matching assistant. Carefully read the JD (must-have & nice-to-have) "
        "and the candidate resume, then return STRICT JSON with fields: "
        "{match_score:int(0-100), must_have_hits:[], must_have_misses:[], "
        "nice_to_have_hits:[], nice_to_have_misses:[], gap_advice:[], resume_bullets:[]}. "
        "Keep concise and actionable."
    )
    res = _call_deepseek([
        {"role":"system","content":sys},
        {"role":"user","content":f"JD:\n{jd_text}\n---------------------\nRESUME:\n{resume_text}\nReturn JSON only."}
    ], temperature=0.2, timeout=80)
    if not res["ok"]:
        return _llm_unavailable()
    parsed = _extract_json(res["content"]) or {}
    for k,v in {
        "match_score":0,"must_have_hits":[],"must_have_misses":[],
        "nice_to_have_hits":[],"nice_to_have_misses":[],
        "gap_advice":[],"resume_bullets":[]
    }.items(): parsed.setdefault(k,v)
    return parsed

# 关键词/摘要兜底
def _heuristic_keywords(text: str, top_k: int = 20):
    text = (text or "")
    words = re.findall(r"[A-Za-z][A-Za-z0-9+\-#]{2,}", text)
    words += re.findall(r"[一-龥]{2,6}", text)
    seen, out = set(), []
    for w in words:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw); out.append(w)
        if len(out) >= 80: break
    return out[:top_k]

def _heuristic_summary(text: str, kws):
    kws = [k for k in (kws or [])][:8]
    if not text.strip(): return "候选人背景信息待补充。"
    return f"候选人具备：{ '、'.join(kws) } 等相关能力与经历，综合背景匹配生物医药/科技方向岗位。"

# 严格一点的粗评分
def _coarse_score(it, kw, filters):
    hay_title   = (it.get("title","") or "").lower()
    hay_company = (it.get("company","") or "").lower()
    hay_loc     = (it.get("location","") or "").lower()
    hay_tags    = " ".join((it.get("tags") or [])).lower()
    hay_notes   = (it.get("notes","") or "").lower()
    hay_all     = " ".join([hay_title, hay_company, hay_loc, hay_tags, hay_notes])

    s = 0.0
    for k in kw:
        if not k: continue
        # 更严格：标题/标签 > 其他
        if k in hay_title: s += 3.5
        if k in hay_tags:  s += 2.2
        if k in hay_company: s += 1.0
        if k in hay_notes: s += 0.8
        if k in hay_loc:   s += 0.7
        if k in hay_all:   s += 0.3

    if filters.get("type") and it.get("type") == filters.get("type"):           s += 1.0
    if filters.get("work_mode") and it.get("work_mode") == filters.get("work_mode"): s += 0.8
    return round(s, 2)


# =========================
# APIs
# =========================

# 职位列表（含多城市 OR）
@app.route("/api/jobs", methods=["GET", "OPTIONS"])
def api_jobs():
    if request.method == "OPTIONS": return _json_response({"ok": True})
    jobs = _read_jobs()

    q = (request.args.get("q") or "").strip().lower()
    location_raw = (request.args.get("location") or "").strip().lower()
    loc_tokens = [t for t in re.split(r"[,\s/，、]+", location_raw) if t]

    type_ = (request.args.get("type") or "").strip().lower()
    work_mode = (request.args.get("work_mode") or "").strip().lower()
    tag = (request.args.get("tag") or "").strip().lower()
    min_salary = _safe_float(request.args.get("min_salary"))
    max_salary = _safe_float(request.args.get("max_salary"))
    currency = (request.args.get("currency") or "").strip().upper()

    sort = (request.args.get("sort") or "posted_desc").strip().lower()
    page = max(1, int(request.args.get("page", "1") or 1))
    page_size = min(100, max(1, int(request.args.get("page_size", "20") or 20)))

    def match(item):
        if q:
            hay = " ".join([
                item.get("title",""), item.get("company",""), item.get("location",""),
                " ".join(item.get("tags",[]) or []), item.get("notes","")
            ]).lower()
            if q not in hay: return False
        if loc_tokens:
            loc_field = (item.get("location","") or "").lower()
            if not any(tok in loc_field for tok in loc_tokens): return False
        if type_ and item.get("type") != type_: return False
        if work_mode and item.get("work_mode") != work_mode: return False
        if currency and (item.get("currency") or "") != currency: return False
        if tag:
            want = {t.strip() for t in tag.split(",") if t.strip()}
            have = {t.lower() for t in (item.get("tags") or [])}
            if not (want & have): return False
        if min_salary is not None or max_salary is not None:
            smin = item.get("salary_min"); smax = item.get("salary_max")
            if smin is not None or smax is not None:
                lo = smin if smin is not None else smax
                hi = smax if smax is not None else smin
                want_lo = min_salary if min_salary is not None else -1e18
                want_hi = max_salary if max_salary is not None else 1e18
                if not (hi >= want_lo and lo <= want_hi): return False
        return True

    filtered = [x for x in jobs if match(x)]
    if sort == "posted_asc":
        filtered.sort(key=lambda x: x.get("_posted_ts", 0))
    else:
        filtered.sort(key=lambda x: x.get("_posted_ts", 0), reverse=True)

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    data = filtered[start:end]
    for it in data: it.pop("_posted_ts", None)
    return _json_response({"total": total, "page": page, "page_size": page_size, "items": data})


# 上传简历 → 文本
@app.route("/api/upload_resume", methods=["POST", "OPTIONS"])
def api_upload_resume():
    if request.method == "OPTIONS": return _json_response({"ok": True})
    if "file" not in request.files: return _json_response({"error": "未接收到文件"}, 400)
    f = request.files["file"]
    if not f or not f.filename: return _json_response({"error": "空文件"}, 400)

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        return _json_response({"error": "仅支持 PDF/DOCX/TXT"}, 400)

    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    f.save(save_path)
    text = _extract_text_from_file(save_path, ext)
    if not text: return _json_response({"error": "解析失败"}, 500)
    return _json_response({"filename": filename, "ext": ext, "chars": len(text), "text": text})


# 一键：解析画像 + 推荐（严格粗评分）
@app.route("/api/analyze_recommend", methods=["POST", "OPTIONS"])
def api_analyze_recommend():
    if request.method == "OPTIONS": return _json_response({"ok": True})
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    filters     = body.get("filters") or {}
    top_n       = int(body.get("top_n") or 50)
    if not resume_text: return _json_response({"error":"resume_text is required."}, 400)

    t0 = time.time()
    profile = _call_profile(resume_text)
    keywords = (profile.get("keywords") or []) if isinstance(profile, dict) else _heuristic_keywords(resume_text)

    jobs = _read_jobs()
    loc_raw = (filters.get("location") or "").strip().lower()
    loc_tokens = [t for t in re.split(r"[,\s/，、]+", loc_raw) if t]

    def pass_filters(it):
        if loc_tokens:
            lf = (it.get("location","") or "").lower()
            if not any(tok in lf for tok in loc_tokens): return False
        tp = (filters.get("type") or "").strip().lower()
        wm = (filters.get("work_mode") or "").strip().lower()
        if tp and it.get("type") != tp: return False
        if wm and it.get("work_mode") != wm: return False
        return True

    cand = [j for j in jobs if pass_filters(j)]
    kw = [k.strip().lower() for k in keywords if isinstance(k,str) and k.strip()]

    scored = []
    for it in cand:
        it2 = dict(it)
        it2["_score"] = _coarse_score(it, kw, filters)
        it2.pop("_posted_ts", None)
        scored.append(it2)

    scored.sort(key=lambda x: (x.get("_score",0), x.get("posted_at","")), reverse=True)
    return _json_response({
        "profile": profile if isinstance(profile, dict) else {},
        "summary_text": (profile.get("summary") if isinstance(profile, dict) else _heuristic_summary(resume_text, keywords)) or "",
        "items": scored[:top_n],
        "total": len(scored),
        "runtime_ms": int((time.time()-t0)*1000)
    })


# 批量精评（前端会把 TopN id 传来，后端逐个 LLM）
@app.route("/api/match_batch", methods=["POST", "OPTIONS"])
def api_match_batch():
    if request.method == "OPTIONS": return _json_response({"ok": True})
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    job_ids     = body.get("job_ids") or []
    limit       = int(body.get("limit") or len(job_ids))
    if not resume_text or not job_ids:
        return _json_response({"error":"resume_text and job_ids are required."}, 400)

    jobs = _read_jobs()
    id2job = {j["id"]: j for j in jobs}
    out = []
    for jid in job_ids[:limit]:
        j = id2job.get(jid)
        if not j: 
            out.append({"id": jid, "error":"job not found"}); 
            continue
        jd_text = "\n".join(filter(None, [
            f"Title: {j.get('title','')}",
            f"Company: {j.get('company','')}",
            f"Location: {j.get('location','')}",
            f"Work Mode: {j.get('work_mode','')}, Type: {j.get('type','')}",
            "Tags: " + ", ".join(j.get("tags") or []),
            j.get("notes","")
        ]))
        res = _call_match(resume_text, jd_text)
        if res.get("error"):
            out.append({"id": jid, "error": "LLM 服务暂不可用"})
        else:
            out.append({"id": jid, **res})
    return _json_response({"results": out})


# 单个精评（保留）
@app.route("/api/match", methods=["POST", "OPTIONS"])
def api_match():
    if request.method == "OPTIONS": return _json_response({"ok": True})
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    jd_text     = (body.get("jd_text") or "").strip()
    if not resume_text or not jd_text:
        return _json_response({"error":"resume_text and jd_text are required."}, 400)
    return _json_response(_call_match(resume_text, jd_text))


# 面试准备包（保留）
@app.route("/api/interview", methods=["POST", "OPTIONS"])
def api_interview():
    if request.method == "OPTIONS": return _json_response({"ok": True})
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    job = body.get("job") or {}
    job_id = (body.get("job_id") or "").strip()
    if not resume_text: return _json_response({"error":"resume_text is required."}, 400)

    if job_id and not job:
        job = next((x for x in _read_jobs() if x.get("id")==job_id), {})
    job_brief = []
    if job:
        job_brief += [
            f"Title: {job.get('title','')}",
            f"Company: {job.get('company','')}",
            f"Location: {job.get('location','')}",
            f"Work Mode: {job.get('work_mode','')}, Type: {job.get('type','')}",
            "Tags: " + ", ".join(job.get("tags") or []),
        ]
        if job.get("notes"):  job_brief.append("Notes: " + job.get("notes",""))
        if job.get("jd_url"): job_brief.append("JD URL: " + job.get("jd_url",""))
    else:
        custom = (body.get("job_brief") or "").strip()
        if custom: job_brief.append(custom)
    return _json_response(_call_match(resume_text, "\n".join(job_brief)))


# 公开职位采集（Greenhouse/Lever）
HEADERS = ["id","title","company","location","work_mode","type","tags","salary_min","salary_max","currency","jd_url","posted_at","source","notes"]

def _append_jobs(rows):
    _ensure_jobs_csv(HEADERS)
    with _csv_lock, open(JOBS_CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        for r in rows:
            r["tags"] = ", ".join(r.get("tags") or [])
            writer.writerow({k:r.get(k,"") for k in HEADERS})

@app.route("/api/ingest_ats", methods=["POST", "OPTIONS"])
def api_ingest_ats():
    if request.method == "OPTIONS": return _json_response({"ok": True})
    # 简单权限
    token = request.headers.get("X-Admin-Token","")
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return _json_response({"error":"unauthorized"}, 403)

    body  = request.get_json(silent=True) or {}
    gh    = body.get("greenhouse") or []  # ["company_slug", ...]
    lever = body.get("lever") or []       # ["company_slug", ...]

    rows = []

    # Greenhouse
    for slug in gh:
        try:
            url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
            data = requests.get(url, timeout=30).json()
            for j in data.get("jobs", []):
                loc = (j.get("location") or {}).get("name","") or ""
                posted = (j.get("updated_at") or j.get("created_at") or "")[:10]
                rows.append({
                    "id": str(uuid.uuid4()),
                    "title": j.get("title",""),
                    "company": slug,
                    "location": loc,
                    "work_mode": "", "type": "full",
                    "tags": [d.get("name","") for d in (j.get("departments") or [])],
                    "salary_min":"", "salary_max":"", "currency":"",
                    "jd_url": j.get("absolute_url",""),
                    "posted_at": posted,
                    "source": "greenhouse",
                    "notes": ""
                })
        except Exception:
            continue

    # Lever
    for slug in lever:
        try:
            url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
            data = requests.get(url, timeout=30).json()
            for j in data:
                loc = ((j.get("categories") or {}).get("location") or "") or ""
                posted = (j.get("createdAt") or 0)
                if isinstance(posted,int) and posted>0:
                    posted = datetime.utcfromtimestamp(posted/1000).strftime("%Y-%m-%d")
                rows.append({
                    "id": str(uuid.uuid4()),
                    "title": j.get("text",""),
                    "company": slug,
                    "location": loc,
                    "work_mode": "", "type": "full",
                    "tags": [t for t in [(j.get("categories") or {}).get("team","")] if t],
                    "salary_min":"", "salary_max":"", "currency":"",
                    "jd_url": j.get("hostedUrl") or j.get("applyUrl") or "",
                    "posted_at": posted or "",
                    "source": "lever",
                    "notes": ""
                })
        except Exception:
            continue

    if rows:
        _append_jobs(rows)
    return _json_response({"ingested": len(rows)})


# 健康/静态
@app.route("/public/<path:filename>", methods=["GET"])
def public_files(filename): return send_from_directory("public", filename)

@app.route("/", methods=["GET"])
def home(): return send_from_directory("public", "index.html")

@app.route("/health", methods=["GET"])
def health(): return _json_response({"status":"ok", "time": int(time.time())})

@app.route("/version", methods=["GET"])
def version(): return _json_response({"name":"LGWORK API", "version":"0.5.0", "jobs_csv": JOBS_CSV_PATH})
