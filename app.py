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
import requests  # 调 DeepSeek
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

app = Flask(__name__)

# -----------------------------
# 配置（环境变量）
# -----------------------------
JOBS_CSV_PATH = os.environ.get("JOBS_CSV_PATH", "data/jobs.csv")
TRACKER_JSON_PATH = os.environ.get("TRACKER_JSON_PATH", "data/tracker.json")

DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 线程安全文件写锁
_tracker_lock = threading.Lock()


# -----------------------------
# 工具函数
# -----------------------------
def _parse_date(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s.strip())
    except Exception:
        return None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _read_jobs():
    jobs = []
    if not os.path.exists(JOBS_CSV_PATH):
        return jobs

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
            # tags 支持分号/逗号
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
    # 清洗与限长
    text = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
    return text[:80000]


def _allow_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PATCH"
    return resp


@app.after_request
def after_request(resp):
    return _allow_cors(resp)


def _json_response(data, status=200):
    resp = make_response(jsonify(data), status)
    return _allow_cors(resp)


def _extract_json(text):
    """从 LLM 文本里鲁棒提取 JSON"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.replace("json\n", "", 1).replace("json\r\n", "", 1)
    t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        try:
            l = t.find("{")
            r = t.rfind("}")
            if l != -1 and r != -1 and r > l:
                return json.loads(t[l:r+1])
        except Exception:
            return None
    return None


# -----------------------------
# DeepSeek 调用
# -----------------------------
def _call_deepseek_match(resume_text: str, jd_text: str):
    if not DEEPSEEK_API_KEY:
        return {"error": "Missing DEEPSEEK_API_KEY. Please set it first."}

    url = f"{DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    system_prompt = (
        "You are an expert job-matching assistant. Carefully read the JD (must-have & nice-to-have) "
        "and the candidate resume, then return STRICT JSON with fields: "
        "{match_score:int(0-100), must_have_hits:[], must_have_misses:[], "
        "nice_to_have_hits:[], nice_to_have_misses:[], gap_advice:[], resume_bullets:[]} "
        "Keep answers concise and actionable."
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"JD:\n{jd_text}\n---------------------\nRESUME:\n{resume_text}\nReturn JSON only."},
        ],
        "temperature": 0.3,
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        content = ""
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "") or ""
        parsed = _extract_json(content)
        if parsed is None:
            return {"error": "LLM returned non-JSON or empty result.", "raw": content}
        for k, v in {
            "match_score": 0,
            "must_have_hits": [],
            "must_have_misses": [],
            "nice_to_have_hits": [],
            "nice_to_have_misses": [],
            "gap_advice": [],
            "resume_bullets": [],
        }.items():
            parsed.setdefault(k, v)
        return parsed
    except requests.exceptions.RequestException as e:
        return {"error": f"DeepSeek API error: {str(e)}"}


def _call_deepseek_profile(resume_text: str):
    """根据简历生成画像与关键词"""
    if not DEEPSEEK_API_KEY:
        return {"error": "Missing DEEPSEEK_API_KEY. Please set it first."}

    url = f"{DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    system_prompt = (
        "You are a career analyst. Given a resume, output STRICT JSON with fields: "
        "{headline:string, summary:string, strengths:[], target_roles:[], industries:[], "
        "skills_core:[], skills_nice:[], locations:[], salary_hint:string, "
        "improvements:[], keywords:[]} "
        "Keywords should be 10-25 short tokens usable for job search (English or Chinese as appropriate). "
        "Return JSON only."
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"RESUME:\n{resume_text}\nReturn JSON only."},
        ],
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        content = ""
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "") or ""
        parsed = _extract_json(content)
        if not isinstance(parsed, dict):
            return {"error": "LLM returned non-JSON.", "raw": content}
        for k, v in {
            "headline": "", "summary": "", "strengths": [], "target_roles": [],
            "industries": [], "skills_core": [], "skills_nice": [], "locations": [],
            "salary_hint": "", "improvements": [], "keywords": []
        }.items():
            parsed.setdefault(k, v)
        return parsed
    except requests.exceptions.RequestException as e:
        return {"error": f"DeepSeek API error: {str(e)}"}


def _call_deepseek_interview(resume_text: str, job_brief: str):
    """生成面试准备包"""
    if not DEEPSEEK_API_KEY:
        return {"error": "Missing DEEPSEEK_API_KEY. Please set it first."}

    url = f"{DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    system_prompt = (
        "You are an interview coach. Given a candidate resume and a target job brief, "
        "produce STRICT JSON: {company_brief:string, role_focus_points:[], must_ask_topics:[], "
        "candidate_gaps:[], practice_questions:[{q:string, talking_points:[]}], "
        "thirty_sec_pitch:string, research_checklist:[], followup_email:string}. "
        "Keep items concise and China job market friendly; bilingual terms allowed. "
        "Return JSON only."
    )
    user_msg = (
        f"JOB BRIEF:\n{job_brief}\n"
        "---------------------\n"
        f"RESUME:\n{resume_text}\n"
        "Return JSON only."
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        content = ""
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "") or ""
        parsed = _extract_json(content)
        if not isinstance(parsed, dict):
            return {"error": "LLM returned non-JSON.", "raw": content}
        # 兜底字段
        skeleton = {
            "company_brief": "", "role_focus_points": [], "must_ask_topics": [],
            "candidate_gaps": [], "practice_questions": [], "thirty_sec_pitch": "",
            "research_checklist": [], "followup_email": ""
        }
        for k, v in skeleton.items():
            parsed.setdefault(k, v)
        # 题目结构兜底
        if isinstance(parsed.get("practice_questions"), list):
            parsed["practice_questions"] = [
                {"q": (x.get("q") if isinstance(x, dict) else str(x)), "talking_points": (x.get("talking_points") if isinstance(x, dict) else [])}
                for x in parsed["practice_questions"]
            ]
        return parsed
    except requests.exceptions.RequestException as e:
        return {"error": f"DeepSeek API error: {str(e)}"}


# -----------------------------
# 路由：职位列表 + 过滤（支持多城市 OR）
# -----------------------------
@app.route("/api/jobs", methods=["GET", "OPTIONS"])
def api_jobs():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})

    jobs = _read_jobs()

    q = (request.args.get("q") or "").strip().lower()
    # 支持 “上海/苏州, 远程” 这种多分隔符
    location_raw = (request.args.get("location") or "").strip().lower()
    loc_tokens = [t for t in re.split(r"[,\s/，、]+", location_raw) if t]

    type_ = (request.args.get("type") or "").strip().lower()           # full/part/project
    work_mode = (request.args.get("work_mode") or "").strip().lower()  # onsite/hybrid/remote
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
                item.get("title", ""),
                item.get("company", ""),
                item.get("location", ""),
                " ".join(item.get("tags", []) or []),
                item.get("notes", "")
            ]).lower()
            if q not in hay:
                return False

        # 多城市 OR
        if loc_tokens:
            loc_field = (item.get("location", "") or "").lower()
            if not any(tok in loc_field for tok in loc_tokens):
                return False

        if type_ and item.get("type") != type_:
            return False
        if work_mode and item.get("work_mode") != work_mode:
            return False
        if currency and (item.get("currency") or "") != currency:
            return False
        if tag:
            want = {t.strip() for t in tag.split(",") if t.strip()}
            have = {t.lower() for t in (item.get("tags") or [])}
            if not (want & have):
                return False
        if min_salary is not None or max_salary is not None:
            smin = item.get("salary_min")
            smax = item.get("salary_max")
            if smin is not None or smax is not None:
                lo = smin if smin is not None else smax
                hi = smax if smax is not None else smin
                want_lo = min_salary if min_salary is not None else -1e18
                want_hi = max_salary if max_salary is not None else 1e18
                if hi is None and lo is None:
                    pass
                else:
                    lo = lo if lo is not None else hi
                    hi = hi if hi is not None else lo
                    if not (hi >= want_lo and lo <= want_hi):
                        return False
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

    for it in data:
        it.pop("_posted_ts", None)

    return _json_response({
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": data
    })


# -----------------------------
# 路由：上传简历 → 提取文本
# -----------------------------
@app.route("/api/upload_resume", methods=["POST", "OPTIONS"])
def api_upload_resume():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})
    if "file" not in request.files:
        return _json_response({"error": "No file found. Use form field 'file'."}, 400)

    f = request.files["file"]
    if not f or not f.filename:
        return _json_response({"error": "Empty file."}, 400)

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        return _json_response({"error": "Unsupported file type. Use PDF/DOCX/TXT."}, 400)

    save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    f.save(save_path)

    text = _extract_text_from_file(save_path, ext)
    if not text:
        return _json_response({"error": "Failed to extract text."}, 500)

    return _json_response({"filename": filename, "ext": ext, "chars": len(text), "text": text})


# -----------------------------
# 路由：画像（简历文本 → 画像JSON）
# -----------------------------
@app.route("/api/profile", methods=["POST", "OPTIONS"])
def api_profile():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})

    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    if not resume_text:
        return _json_response({"error": "resume_text is required."}, 400)

    result = _call_deepseek_profile(resume_text)
    return _json_response(result)


# -----------------------------
# 路由：推荐（画像关键词/简历 → 职位粗评分）
# -----------------------------
@app.route("/api/recommend", methods=["POST", "OPTIONS"])
def api_recommend():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    keywords = body.get("keywords") or []
    filters = body.get("filters") or {}  # {"location":"上海","type":"full","work_mode":"remote"}
    top_n = int(body.get("top_n") or 50)

    if not keywords and resume_text:
        prof = _call_deepseek_profile(resume_text)
        if isinstance(prof, dict) and not prof.get("error"):
            keywords = prof.get("keywords") or []
    if not keywords and not resume_text:
        keywords = []

    jobs = _read_jobs()

    def pass_filters(it):
        loc = (filters.get("location") or "").strip().lower()
        tp = (filters.get("type") or "").strip().lower()
        wm = (filters.get("work_mode") or "").strip().lower()
        if loc and loc not in (it.get("location", "").lower()):
            return False
        if tp and it.get("type") != tp:
            return False
        if wm and it.get("work_mode") != wm:
            return False
        return True

    cand = [j for j in jobs if pass_filters(j)]

    kw = [k.strip().lower() for k in keywords if isinstance(k, str) and k.strip()]

    def score(it):
        hay_title = (it.get("title", "") or "").lower()
        hay_company = (it.get("company", "") or "").lower()
        hay_loc = (it.get("location", "") or "").lower()
        hay_tags = " ".join((it.get("tags") or [])).lower()
        hay_notes = (it.get("notes", "") or "").lower()
        hay = " ".join([hay_title, hay_company, hay_loc, hay_tags, hay_notes])

        s = 0.0
        for k in kw:
            if not k:
                continue
            if k in hay_title:
                s += 3.0
            if k in hay_tags:
                s += 2.5
            if k in hay_company:
                s += 1.0
            if k in hay_notes or k in hay_loc:
                s += 1.0
            if k in hay:
                s += 0.5
        if filters.get("type") and it.get("type") == filters.get("type"):
            s += 1.0
        if filters.get("work_mode") and it.get("work_mode") == filters.get("work_mode"):
            s += 1.0
        return s

    scored = []
    for it in cand:
        s = score(it)
        it2 = dict(it)
        it2["_score"] = round(s, 2)
        it2.pop("_posted_ts", None)
        scored.append(it2)

    scored.sort(key=lambda x: (x.get("_score", 0), x.get("posted_at", "")), reverse=True)
    return _json_response({"items": scored[:top_n], "total": len(scored)})


# -----------------------------
# 路由：深度匹配（JD vs 简历）
# -----------------------------
@app.route("/api/match", methods=["POST", "OPTIONS"])
def api_match():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})

    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    jd_text = (body.get("jd_text") or "").strip()

    if not resume_text or not jd_text:
        return _json_response({"error": "resume_text and jd_text are required."}, 400)

    result = _call_deepseek_match(resume_text, jd_text)
    return _json_response(result)


# -----------------------------
# 路由：面试准备（简历 + job / job_id）
# -----------------------------
@app.route("/api/interview", methods=["POST", "OPTIONS"])
def api_interview():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})

    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    job = body.get("job") or {}
    job_id = (body.get("job_id") or "").strip()

    if not resume_text:
        return _json_response({"error": "resume_text is required."}, 400)

    # 若提供 job_id，用本地职位库补齐 job brief
    if job_id and not job:
        jobs = _read_jobs()
        job = next((x for x in jobs if x.get("id") == job_id), {})

    # 组合 job brief（不从外网抓取，只用本地元数据）
    job_brief = []
    if job:
        job_brief.append(f"Title: {job.get('title','')}")
        job_brief.append(f"Company: {job.get('company','')}")
        job_brief.append(f"Location: {job.get('location','')}")
        job_brief.append(f"Work Mode: {job.get('work_mode','')}, Type: {job.get('type','')}")
        tags = job.get("tags") or []
        if tags:
            job_brief.append("Tags: " + ", ".join(tags))
        if job.get("notes"):
            job_brief.append("Notes: " + job.get("notes", ""))
        if job.get("jd_url"):
            job_brief.append("JD URL: " + job.get("jd_url", ""))
    else:
        # 允许只传自定义 brief
        custom = (body.get("job_brief") or "").strip()
        if custom:
            job_brief.append(custom)

    jb = "\n".join([x for x in job_brief if x])

    result = _call_deepseek_interview(resume_text, jb)
    return _json_response(result)


# -----------------------------
# 路由：轻量投递/跟进
# -----------------------------
def _ensure_tracker():
    os.makedirs(os.path.dirname(TRACKER_JSON_PATH), exist_ok=True)
    if not os.path.exists(TRACKER_JSON_PATH):
        with open(TRACKER_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"items": []}, f, ensure_ascii=False, indent=2)


def _load_tracker():
    _ensure_tracker()
    with open(TRACKER_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_tracker(data):
    with _tracker_lock:
        with open(TRACKER_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/api/track", methods=["GET", "POST", "OPTIONS"])
def api_track():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})

    if request.method == "GET":
        data = _load_tracker()
        return _json_response(data)

    body = request.get_json(silent=True) or {}
    job_id = (body.get("job_id") or "").strip()
    status_ = (body.get("status") or "").strip()  # 准备中/已投/面试中/Offer/关闭
    notes = (body.get("notes") or "").strip()

    if not job_id or not status_:
        return _json_response({"error": "job_id and status are required."}, 400)

    data = _load_tracker()
    now_iso = datetime.utcnow().isoformat()

    item_id = (body.get("id") or "").strip()
    if item_id:
        found = False
        for it in data.get("items", []):
            if it.get("id") == item_id:
                it["status"] = status_
                it["notes"] = notes
                it["updated_at"] = now_iso
                found = True
                break
        if not found:
            return _json_response({"error": "track item not found."}, 404)
    else:
        new_item = {
            "id": str(uuid.uuid4()),
            "job_id": job_id,
            "status": status_,
            "notes": notes,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        data.setdefault("items", []).append(new_item)

    _save_tracker(data)
    return _json_response({"ok": True})


# -----------------------------
# 静态文件 & 健康检查
# -----------------------------
@app.route("/public/<path:filename>", methods=["GET"])
def public_files(filename):
    return send_from_directory("public", filename)


@app.route("/", methods=["GET"])
def home():
    return send_from_directory("public", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return _json_response({"status": "ok", "time": int(time.time())})


@app.route("/version", methods=["GET"])
def version():
    return _json_response({
        "name": "LGWORK API",
        "version": "0.3.1",
        "jobs_csv": JOBS_CSV_PATH,
        "model": DEEPSEEK_MODEL
    })


# -----------------------------
# 本地启动
# -----------------------------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(JOBS_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TRACKER_JSON_PATH), exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
