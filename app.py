import os, csv, json, uuid, time, re
from datetime import datetime
from urllib.parse import unquote_plus

import requests
from flask import Flask, request, jsonify, make_response, send_from_directory

# =========================
# 配置
# =========================

app = Flask(__name__)

VERSION = "0.5.0"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

JOBS_CSV_PATH = os.environ.get("JOBS_CSV_PATH", os.path.join(DATA_DIR, "jobs.csv"))
TRACKER_JSON_PATH = os.environ.get("TRACKER_JSON_PATH", os.path.join(DATA_DIR, "tracker.json"))

# DeepSeek / LLM 兼容读取
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY") or ""
DEEPSEEK_API_BASE = (os.environ.get("DEEPSEEK_API_BASE") or os.environ.get("LLM_BASE_URL") or "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL") or os.environ.get("MODEL_NAME") or "deepseek-chat"

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# =========================
# 通用工具
# =========================

def _json_response(data, status=200):
    resp = make_response(jsonify(data), status)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

def _safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except:
        return None

def _ensure_jobs_csv_exists():
    if not os.path.exists(JOBS_CSV_PATH):
        # 写入一个空文件头
        fields = [
            "id","company","title","location","type","work_mode","currency",
            "salary_min","salary_max","posted_at","jd_url","source","tags","notes"
        ]
        with open(JOBS_CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()

def _read_jobs():
    """
    从 CSV 读取职位（对 tags 容错，空/脏数据一律当 []）
    """
    _ensure_jobs_csv_exists()
    jobs = []
    with open(JOBS_CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ---- tags 容错处理 ----
            tags = []
            raw = row.get("tags")
            if isinstance(raw, str):
                s = raw.strip()
                if s:
                    try:
                        t = json.loads(s)
                        if isinstance(t, list):
                            tags = t
                    except Exception:
                        tags = []
            # -----------------------

            jobs.append({
                "id": row.get("id") or str(uuid.uuid4()),
                "company": (row.get("company") or "").strip(),
                "title": (row.get("title") or "").strip(),
                "location": (row.get("location") or "").strip(),
                "type": (row.get("type") or "").strip(),
                "work_mode": (row.get("work_mode") or "").strip(),
                "currency": (row.get("currency") or "").strip(),
                "salary_min": _safe_float(row.get("salary_min")),
                "salary_max": _safe_float(row.get("salary_max")),
                "posted_at": (row.get("posted_at") or "").strip(),
                "jd_url": (row.get("jd_url") or "").strip(),
                "source": (row.get("source") or "").strip(),
                "tags": tags,
                "notes": (row.get("notes") or "").strip(),
            })
    return jobs


# =========================
# LLM 调用（DeepSeek）
# =========================

def _call_deepseek(messages, temperature=0.2, timeout=30):
    """
    调用 DeepSeek Chat Completions
    """
    if not DEEPSEEK_API_KEY:
        return {"ok": False, "error": "missing_api_key"}

    url = f"{DEEPSEEK_API_BASE}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return {"ok": False, "error": f"status_{r.status_code}", "detail": r.text[:500]}
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "content": content}
    except Exception as e:
        return {"ok": False, "error": "exception", "detail": str(e)[:400]}

# =========================
# 简历解析
# =========================

def _extract_text_from_pdf(file_stream):
    try:
        from pdfminer.high_level import extract_text
        return extract_text(file_stream)
    except Exception:
        return ""

def _extract_text_from_docx(file_stream):
    try:
        from docx import Document
        import io
        doc = Document(io.BytesIO(file_stream.read()))
        paras = [p.text for p in doc.paragraphs]
        return "\n".join([p for p in paras if p and p.strip()])
    except Exception:
        return ""

def _extract_text_from_upload(file_storage):
    name = (file_storage.filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_text_from_pdf(file_storage.stream)
    if name.endswith(".docx"):
        file_storage.stream.seek(0)
        return _extract_text_from_docx(file_storage)
    # 纯文本
    try:
        file_storage.stream.seek(0)
        return file_storage.stream.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

# =========================
# 画像 & 摘要
# =========================

PROFILE_SYS = (
    "你是一名资深猎头顾问，擅长从简历中提取候选人的画像、优势、关键词与目标岗位方向。"
    "请用简洁中文输出一个【100~180字】的第三人称摘要（像写候选人介绍），"
    "并给出 8~12 个关键技能/领域关键词（短词），用 JSON 返回："
    '{"summary":"...", "keywords":["..."]}'
)

def _profile_from_resume(text):
    if not text.strip():
        return {"summary":"", "keywords":[]}
    res = _call_deepseek([
        {"role":"system","content":PROFILE_SYS},
        {"role":"user","content":text[:5000]}
    ], temperature=0.2, timeout=40)
    if not res["ok"]:
        # 兜底：简单规则
        words = list({w.strip(",.;:()[] ") for w in re.findall(r"[A-Za-z\u4e00-\u9fa5\-\+/#]{2,20}", text) if len(w)>1})
        return {"summary": text.strip().split("\n",1)[0][:140], "keywords": words[:10]}
    # 尝试解析 JSON
    content = res.get("content") or ""
    m = re.search(r'\{.*\}', content, flags=re.S)
    if not m:
        return {"summary": content.strip()[:180], "keywords":[]}
    try:
        data = json.loads(m.group(0))
        return {"summary": (data.get("summary") or "")[:200], "keywords": data.get("keywords") or []}
    except:
        return {"summary": content.strip()[:180], "keywords":[]}

# =========================
# LLM 职位精评
# =========================

MATCH_SYS = (
    "你是资深招聘顾问，请根据职位 JD 与候选人简历进行人岗匹配评估，"
    "给出 0~100 的整数分（越高越匹配），并列出 Must-have 和 Nice-to-have 的命中/缺失项，"
    "最后给出 2-4 条补差建议。用 JSON 返回："
    '{"match_score":85,"must_have_hits":["A"],"must_have_misses":["B"],'
    '"nice_to_have_hits":["C"],"nice_to_have_misses":["D"],"gap_advice":["建议1","建议2"]}'
)

def _llm_match(jd_text, resume_text, timeout=60):
    if not DEEPSEEK_API_KEY:
        # 兜底：关键词重叠简单评分
        s1 = set(re.findall(r"[A-Za-z\u4e00-\u9fa5\-+#]{2,20}", jd_text.lower()))
        s2 = set(re.findall(r"[A-Za-z\u4e00-\u9fa5\-+#]{2,20}", resume_text.lower()))
        inter = len(s1 & s2)
        score = min(100, 20 + inter)
        return {"ok": True, "match_score": score}
    prompt = f"【职位JD】\n{jd_text[:5000]}\n\n【候选人简历】\n{resume_text[:5000]}\n"
    res = _call_deepseek([
        {"role":"system","content":MATCH_SYS},
        {"role":"user","content":prompt}
    ], temperature=0.2, timeout=timeout)
    if not res["ok"]:
        return {"ok": False, "error": res.get("error")}
    content = res.get("content") or ""
    m = re.search(r'\{.*\}', content, flags=re.S)
    try:
        data = json.loads(m.group(0) if m else content)
    except:
        data = {"match_score": 0}
    data["ok"] = True
    data["match_score"] = int(float(data.get("match_score") or 0))
    return data

# =========================
# ATS 采集函数
# =========================

def _fetch_greenhouse_jobs(slug: str):
    """
    Greenhouse 公开接口：
    https://boards-api.greenhouse.io/v1/boards/{slug}/jobs
    """
    items = []
    if not slug:
        return items
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return items
        data = r.json()
        for j in data.get("jobs", []):
            items.append({
                "id": str(uuid.uuid4()),
                "company": j.get("company",{}).get("name") or slug,
                "title": j.get("title") or "",
                "location": (j.get("location",{}) or {}).get("name",""),
                "type": "", "work_mode":"", "currency":"",
                "salary_min": None, "salary_max": None,
                "posted_at": (j.get("updated_at") or "")[:10],
                "jd_url": j.get("absolute_url") or "",
                "source": "greenhouse",
                "tags": [], "notes": ""
            })
    except Exception:
        pass
    return items

def _fetch_lever_jobs(slug: str):
    """
    Lever 公开接口：
    https://api.lever.co/v0/postings/{slug}?mode=json
    """
    items = []
    if not slug:
        return items
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return items
        for j in r.json():
            # locations 可能为数组
            locs = j.get("categories", {}).get("location") or ""
            items.append({
                "id": str(uuid.uuid4()),
                "company": slug,
                "title": j.get("text") or "",
                "location": locs,
                "type": "", "work_mode":"", "currency":"",
                "salary_min": None, "salary_max": None,
                "posted_at": (j.get("createdAt") and datetime.utcfromtimestamp(int(j["createdAt"])/1000).strftime("%Y-%m-%d")) or "",
                "jd_url": j.get("hostedUrl") or j.get("applyUrl") or "",
                "source": "lever",
                "tags": [], "notes": ""
            })
    except Exception:
        pass
    return items

def _fetch_workday_jobs(tenant: str, site: str, max_pages: int = 20):
    """
    采集 Workday 的公开职位（不登录，走官方 JSON 接口）。
    tenant/site 例如：beigene / BeiGene
    Workday 的域名分区不固定，常见 wd5/wd3/wd1... 这里会轮询尝试。
    """
    tenant = (tenant or "").strip()
    site = (site or "").strip()
    if not tenant or not site:
        return []

    zones = ["wd5","wd3","wd1","wd2","wd4","wd8","wd9","wd6","wd7"]
    session = requests.Session()
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": f"https://{tenant}.wd5.myworkdayjobs.com",
        "Referer": f"https://{tenant}.wd5.myworkdayjobs.com/en-US/{site}",
        "User-Agent": "Mozilla/5.0"
    }

    base = None
    for z in zones:
        test = f"https://{tenant}.{z}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
        try:
            t = session.post(test, headers=headers, json={"limit":1,"offset":0,"searchText":""}, timeout=15)
            if t.ok and t.headers.get("content-type","").startswith("application/json"):
                base = test  # 成功即用该分区
                break
        except Exception:
            pass
    if not base:
        return []

    out = []
    offset = 0
    limit = 50
    for _ in range(max_pages):
        try:
            r = session.post(base, headers=headers, json={"limit":limit,"offset":offset,"searchText":""}, timeout=20)
            if not r.ok:
                break
            data = r.json()
            posts = data.get("jobPostings") or []
            if not posts:
                break
            for p in posts:
                title = (p.get("title") or "").strip()
                locations = ", ".join([(x.get("descriptor") or "").strip() for x in (p.get("locations") or []) if x.get("descriptor")])
                posted = (p.get("postedOn") or "").strip()
                tt = (p.get("timeType") or "").lower()
                path = p.get("externalPath") or p.get("externalUrl") or ""
                jd_url = f"https://{tenant}.wd5.myworkdayjobs.com{path}" if path.startswith("/") else path
                out.append({
                    "id": str(uuid.uuid4()),
                    "company": site,
                    "title": title,
                    "location": locations,
                    "type": "full" if "full" in tt else ("part" if "part" in tt else ""),
                    "work_mode": "",
                    "currency": "",
                    "salary_min": None,
                    "salary_max": None,
                    "posted_at": posted,
                    "jd_url": jd_url,
                    "source": "workday",
                    "tags": [],
                    "notes": ""
                })
            if len(posts) < limit:
                break
            offset += limit
        except Exception:
            break
    return out

# =========================
# 路由：健康/版本/静态
# =========================

@app.route("/health")
def health():
    return _json_response({"status":"ok","time": int(time.time())})

@app.route("/version")
def version():
    return _json_response({"version": VERSION})

@app.route("/public/<path:filename>")
def public_files(filename):
    return send_from_directory("public", filename)

# =========================
# 路由：职位列表
# =========================

# 让根路径直接打开前端首页（agent.html）
@app.route("/")
def index():
    return send_from_directory("public", "agent.html")

# 可选：/admin 直达采集页（不想记 /public/admin.html）
@app.route("/admin")
def admin_shortcut():
    return send_from_directory("public", "admin.html")

# 可选：把 404 兜底到首页，避免用户输错路径看到 404
@app.errorhandler(404)
def not_found(_):
    try:
        return send_from_directory("public", "agent.html")
    except Exception:
        return jsonify({"error": "not_found"}), 404


@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    location = (request.args.get("location") or "").strip()
    type_ = (request.args.get("type") or "").strip().lower()      # full/part/project
    work_mode = (request.args.get("work_mode") or "").strip().lower()  # onsite/hybrid/remote
    sort_posted_desc = request.args.get("sort_posted_desc")

    items = _read_jobs()

    # 过滤
    out = []
    for it in items:
        if location and location not in (it.get("location") or ""):
            continue
        if type_ and type_ != (it.get("type") or "").lower():
            continue
        if work_mode and work_mode != (it.get("work_mode") or "").lower():
            continue
        out.append(it)

    # 排序（按 posted_at 降序）
    if sort_posted_desc is not None:
        def _dt(s):
            try:
                return datetime.strptime(s[:10], "%Y-%m-%d")
            except:
                return datetime.min
        out.sort(key=lambda x: _dt(x.get("posted_at") or ""), reverse=True)

    total = len(out)
    start = (page - 1) * page_size
    end = start + page_size
    return _json_response({"items": out[start:end], "page": page, "page_size": page_size, "total": total})

# =========================
# 路由：上传简历 → 文本
# =========================

@app.route("/api/upload_resume", methods=["POST"])
def api_upload_resume():
    if "file" not in request.files:
        return _json_response({"error":"missing_file"}, 400)
    f = request.files["file"]
    text = _extract_text_from_upload(f)
    return _json_response({"text": text})

# =========================
# 路由：根据简历生成画像（单独接口）
# =========================

@app.route("/api/profile", methods=["POST"])
def api_profile():
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    prof = _profile_from_resume(resume_text)
    return _json_response({"profile": prof})

# =========================
# 路由：一键分析并推荐（摘要+关键词+职位）
# =========================

@app.route("/api/analyze_recommend", methods=["POST"])
def api_analyze_recommend():
    body = request.get_json(silent=True) or {}
    resume_text = (body.get("resume_text") or "").strip()
    filters = body.get("filters") or {}
    top_n = int(body.get("top_n") or 100)

    prof = _profile_from_resume(resume_text)
    # 关键词用于初筛（很粗略）
    kws = set([k.strip().lower() for k in prof.get("keywords") or [] if k and isinstance(k, str)])

    # 拉所有职位，然后进行粗过滤（地点/类型/关键词）
    jobs = _read_jobs()

    loc = (filters.get("location") or "").strip()
    type_ = (filters.get("type") or "").strip().lower()
    work_mode = (filters.get("work_mode") or "").strip().lower()

    prelim = []
    for j in jobs:
        if loc and loc not in (j.get("location") or ""):
            continue
        if type_ and type_ != (j.get("type") or "").lower():
            continue
        if work_mode and work_mode != (j.get("work_mode") or "").lower():
            continue
        # 关键词简单打分（出现则 +1）
        score = 0
        field = " ".join([j.get("title",""), j.get("company",""), " ".join(j.get("tags") or [])]).lower()
        for k in kws:
            if k and k in field:
                score += 1
        prelim.append((score, j))

    prelim.sort(key=lambda x: x[0], reverse=True)
    items = [it for _, it in prelim[:top_n]]

    return _json_response({
        "summary_text": prof.get("summary") or "",
        "profile": prof,
        "items": items,
        "total": len(prelim)
    })

# =========================
# 路由：单个职位 LLM 精评
# =========================

@app.route("/api/match", methods=["POST"])
def api_match():
    body = request.get_json(silent=True) or {}
    jd_text = (body.get("jd_text") or "").strip()
    resume_text = (body.get("resume_text") or "").strip()
    if not jd_text or not resume_text:
        return _json_response({"error":"missing_text"}, 400)
    out = _llm_match(jd_text, resume_text, timeout=60)
    if not out.get("ok"):
        return _json_response({"error":"llm_unavailable"}, 503)
    return _json_response(out)

# =========================
# 路由：批量职位 LLM 精评（前端用来自动出“匹配度/LLM 分”）
# =========================

@app.route("/api/match_batch", methods=["POST"])
def api_match_batch():
    body = request.get_json(silent=True) or {}
    job_ids = body.get("job_ids") or []
    resume_text = (body.get("resume_text") or "").strip()
    if not job_ids or not resume_text:
        return _json_response({"results": []})

    all_jobs = {j["id"]: j for j in _read_jobs()}
    results = []
    for jid in job_ids:
        j = all_jobs.get(jid)
        if not j:
            results.append({"id": jid, "error": "not_found"})
            continue
        jd_text = "\n".join([
            f"Company: {j.get('company','')}",
            f"Title: {j.get('title','')}",
            f"Location: {j.get('location','')}",
            f"Work Mode: {j.get('work_mode','')}, Type: {j.get('type','')}",
            f"Tags: {', '.join(j.get('tags') or [])}",
            j.get("notes","")
        ])
        out = _llm_match(jd_text, resume_text, timeout=55)
        if not out.get("ok"):
            results.append({"id": jid, "error": "llm_unavailable"})
        else:
            results.append({"id": jid, **out})
    return _json_response({"results": results})

# =========================
# 路由：导入公开职位（Greenhouse / Lever / Workday）
# =========================

@app.route("/api/ingest_ats", methods=["POST"])
def api_ingest_ats():
    if ADMIN_TOKEN and request.headers.get("X-Admin-Token") != ADMIN_TOKEN:
        return _json_response({"error":"unauthorized"}, 401)

    body = request.get_json(silent=True) or {}
    gh_list = body.get("greenhouse") or []
    lv_list = body.get("lever") or []
    wd_list = body.get("workday") or []   # [{tenant, site}, ...]

    new_jobs = []

    # Greenhouse
    for slug in gh_list:
        slug = (slug or "").strip()
        if not slug:
            continue
        try:
            new_jobs.extend(_fetch_greenhouse_jobs(slug))
        except Exception:
            pass

    # Lever
    for slug in lv_list:
        slug = (slug or "").strip()
        if not slug:
            continue
        try:
            new_jobs.extend(_fetch_lever_jobs(slug))
        except Exception:
            pass

    # Workday
    for itm in wd_list:
        tenant = (itm.get("tenant") or "").strip()
        site   = (itm.get("site") or "").strip()
        if not tenant or not site:
            continue
        try:
            new_jobs.extend(_fetch_workday_jobs(tenant, site))
        except Exception:
            pass

    # 去重（按 jd_url）
    seen = set()
    dedup = []
    for j in new_jobs:
        k = (j.get("jd_url") or "").strip()
        if not k or k in seen:
            continue
        seen.add(k)
        dedup.append(j)

    _append_jobs_to_csv(dedup)
    return _json_response({"ingested": len(dedup)})

# =========================
# 路由：LLM 自检
# =========================

@app.route("/api/llm_ping", methods=["GET"])
def api_llm_ping():
    if not DEEPSEEK_API_KEY:
        return _json_response({"ok": False, "error":"missing_api_key"}, 503)
    res = _call_deepseek([{"role":"user","content":"reply with: ok"}], temperature=0.0, timeout=20)
    if not res["ok"]:
        return _json_response({"ok": False, "error":"llm_unavailable"}, 503)
    return _json_response({"ok": True, "content": (res.get("content") or "").strip()[:50]})

# =========================
# 入口
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
