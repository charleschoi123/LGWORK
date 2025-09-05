# app.py  —— 向后兼容 + 超集版本
# 功能清单（保持/增强）：
# - 静态路由：/  -> public/agent.html；/public/* -> 静态资源
# - 简历上传解析：/api/upload_resume（PDF/DOCX/TXT）
# - 简历画像&建议：/api/profile（优先 DeepSeek，无Key自动回退本地）
# - 检索职位：/api/jobs（分页/筛选/排序，兼容旧 query）
# - 生成推荐：/api/analyze_recommend（严格过滤、启发式打分）
# - 批量精评：/api/match_batch（批量刷新卡片分数）
# - ATS 导入：/api/ingest_ats（Greenhouse/Lever）
# - Workday 导入：/api/ingest_cn_jobs（支持 Workday 入口）
# - 单条职位：/api/job?id=xxx（可选，给前端查看详情用）
# - 事件追踪：/api/track（写入 tracker.json）
# - 健康检查：/health
# 兼容修复：_json_response 别名、旧字段名兼容、CSV 读写、tags/notes 处理

import os, csv, json, re, uuid, time
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import urlparse
from flask import Flask, request, jsonify, make_response, send_from_directory
import requests

app = Flask(__name__)

# -------------------- 环境变量 --------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

JOBS_CSV = os.environ.get("JOBS_CSV_PATH", os.path.join(DATA_DIR, "jobs.csv"))
TRACKER_JSON = os.environ.get("TRACKER_JSON_PATH", os.path.join(DATA_DIR, "tracker.json"))

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()

LLM_API_KEY  = os.environ.get("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
LLM_MODEL    = os.environ.get("MODEL_NAME", "deepseek-chat").strip()

# -------------------- 工具 --------------------
def json_response(obj, status=200):
    return make_response(jsonify(obj), status)

# 兼容旧代码使用的 _json_response
_json_response = json_response

def now_ts() -> int:
    return int(time.time())

def safe_json_loads(s, default=None):
    try:
        return json.loads(s)
    except Exception:
        return default

# -------------------- CSV 读写 --------------------
JOB_FIELDS = [
    "id","company","title","jd_url","location","type","work_mode",
    "posted_at","currency","salary_min","salary_max","source","tags","notes"
]

def _ensure_fields(row: dict) -> dict:
    out = {k: row.get(k) for k in JOB_FIELDS}
    out["id"] = out["id"] or str(uuid.uuid4())
    # tags 兼容 list/str
    if isinstance(out.get("tags"), list):
        pass
    elif out.get("tags"):
        out["tags"] = [x.strip() for x in str(out["tags"]).split(",") if x.strip()]
    else:
        out["tags"] = []
    # 统一字符串化
    for k in JOB_FIELDS:
        if k == "tags": continue
        if out.get(k) is None: out[k] = ""
        out[k] = str(out[k])
    return out

def _row_to_csv(row: dict) -> dict:
    row = _ensure_fields(row)
    row2 = row.copy()
    row2["tags"] = json.dumps(row["tags"], ensure_ascii=False)
    return row2

def _row_from_csv(row: dict) -> dict:
    d = dict(row)
    d["tags"] = safe_json_loads(d.get("tags",""), default=[]) or []
    return d

def read_jobs():
    if not os.path.exists(JOBS_CSV):
        return []
    out=[]
    with open(JOBS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(_row_from_csv(row))
    return out

def append_jobs(rows):
    if not rows: return 0
    exists = os.path.exists(JOBS_CSV)
    with open(JOBS_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        if not exists: w.writeheader()
        for x in rows:
            w.writerow(_row_to_csv(x))
    return len(rows)

def overwrite_jobs(all_rows):
    with open(JOBS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOB_FIELDS)
        w.writeheader()
        for x in all_rows:
            w.writerow(_row_to_csv(x))
    return len(all_rows)

# -------------------- 文本解析 --------------------
def extract_text_from_pdf(bts: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(BytesIO(bts)) or ""
    except Exception:
        return ""

def extract_text_from_docx(bts: bytes) -> str:
    try:
        from docx import Document
        bio = BytesIO(bts)
        doc = Document(bio)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text_from_txt(bts: bytes) -> str:
    try:
        return bts.decode("utf-8", errors="ignore")
    except Exception:
        return bts.decode("latin1", errors="ignore")

# -------------------- 关键词 / 打分 --------------------
STOP = set("""
the a an and or for with without into from of in to as is are was were be been being on at by about
this that these those you your our we they he she it their his her its than then over under until while against
""".split())

def tokenize(s: str):
    s = (s or "").lower()
    return [t for t in re.split(r"[^a-z0-9\+\#]+", s) if t and t not in STOP and len(t)>=2]

def extract_keywords(resume: str, extra: str="") -> list:
    toks = tokenize((resume or "") + " " + (extra or ""))
    freq={}
    for t in toks:
        if t.isdigit(): continue
        freq[t]=freq.get(t,0)+1
    boost = {
        "adc","ai","ml","nlp","python","java","biotech","oncology","immuno","bd","licensing","cro","gcp",
        "clinical","chemist","protein","cell","assay","phd","cv","llm","data","bio","med","pharma"
    }
    scored=[(v+(3 if k in boost else 0), k) for k,v in freq.items()]
    scored.sort(reverse=True)
    return [k for _,k in scored[:50]]

def strict_match_filter(resume_kws: list, job) -> bool:
    title = " " + (job.get("title") or "").lower() + " "
    tags  = " ".join([t.lower() for t in (job.get("tags") or [])])

    domain = {"biotech","oncology","immuno","clinical","cro","gcp","chem","assay","pharma","bio"}
    resume_is_bio = any(k in resume_kws for k in domain)
    if resume_is_bio:
        if re.search(r'\b(software|backend|frontend|fullstack|ios|android|react|sre|devops|qa|tester)\b', title):
            return False
    return any(k in title or k in tags for k in resume_kws[:12])

def heuristic_score(resume_kws: list, job, loc_pref: str="") -> int:
    title = (job.get("title") or "").lower()
    tags  = [t.lower() for t in (job.get("tags") or [])]
    company = (job.get("company") or "").lower()
    loc  = (job.get("location") or "").lower()
    recency = 0
    try:
        if job.get("posted_at"):
            dt = datetime.strptime(job["posted_at"][:10], "%Y-%m-%d")
            days = max(1, (datetime.utcnow()-dt).days)
            recency = max(0, 20 - int(days/7))
    except Exception: pass

    hit_title = sum(1 for k in resume_kws if k in title)
    hit_tags  = sum(1 for k in resume_kws if k in tags)
    s = hit_title * 18 + hit_tags * 10
    if any(x in company for x in ["bio","pharm","onc","gene","med","thera"]):
        s += 8
    if loc_pref and loc_pref in loc:
        s += 10
    s += recency
    return max(0, min(100, s))

# -------------------- LLM 调用（DeepSeek） --------------------
def llm_chat(messages: list, temperature=0.3, max_tokens=600) -> str:
    if not LLM_API_KEY:
        return ""
    try:
        url = f"{LLM_BASE_URL}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            print("LLM error", r.status_code, r.text[:200])
            return ""
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message",{}).get("content","").strip()
    except Exception as e:
        print("LLM exception:", e)
        return ""

def build_profile_with_llm(resume_text: str) -> dict:
    sys = (
        "你是资深职业顾问。请阅读用户简历，生成："
        "1）3~4 句客观中性的画像摘要（第二/第三人称）；"
        "2）10~20 个英文关键词（逗号分隔）；"
        "3）2~4 条求职建议（短句）。"
        "只输出 JSON，形如：{\"summary\":\"...\",\"keywords\":[...],\"advice\":[...]}"
    )
    content = llm_chat([
        {"role":"system","content":sys},
        {"role":"user","content":resume_text[:8000]}
    ], temperature=0.2, max_tokens=800)
    if content:
        j = safe_json_loads(content, default=None)
        if isinstance(j, dict):
            j["keywords"] = [x.strip() for x in (j.get("keywords") or []) if x.strip()]
            return j
    kws = extract_keywords(resume_text)
    top3 = ", ".join(kws[:3]) if kws else "综合背景"
    return {
        "summary": f"候选人在「{top3}」方向具备经验；建议突出量化成果与落地成效，补强关键技能。",
        "keywords": kws[:15],
        "advice": ["补充数字化成果", "量化产出指标", "强化跨部门协作描述"]
    }

def llm_score_fit(resume_text: str, job: dict) -> int:
    if not LLM_API_KEY: return -1
    prompt = (
        "请基于简历与岗位标题/标签，评估匹配度（0-100）。只输出单个整数。"
        f"\n\n[简历]\n{resume_text[:2000]}\n\n[岗位]\n{job.get('title','')}\nTags: {', '.join(job.get('tags') or [])}"
    )
    out = llm_chat([{"role":"system","content":"只给出一个整数分数。"},
                    {"role":"user","content":prompt}], temperature=0.0, max_tokens=10)
    try:
        n = int(re.findall(r"\d+", out or "")[0])
        return max(0, min(100, n))
    except Exception:
        return -1

# -------------------- ATS/Workday 导入 --------------------
def import_greenhouse(slugs):
    out=[]
    for slug in slugs:
        s=slug.strip()
        if not s: continue
        url=f"https://boards-api.greenhouse.io/v1/boards/{s}/jobs?content=true"
        try:
            r=requests.get(url, timeout=12)
            if r.status_code!=200: continue
            for j in (r.json() or {}).get("jobs",[]):
                out.append(_ensure_fields({
                    "id": str(uuid.uuid4()),
                    "company": s.upper(),
                    "title": j.get("title") or "",
                    "jd_url": j.get("absolute_url"),
                    "location": (j.get("location") or {}).get("name",""),
                    "type": "full",
                    "work_mode": "",
                    "posted_at": (j.get("updated_at") or "")[:10],
                    "currency": "",
                    "salary_min": "",
                    "salary_max": "",
                    "source": "greenhouse",
                    "tags": [s],
                    "notes": ""
                }))
        except Exception as e:
            print("greenhouse error:", s, e)
    return out

def import_lever(slugs):
    out=[]
    for slug in slugs:
        s=slug.strip()
        if not s: continue
        url=f"https://api.lever.co/v0/postings/{s}?mode=json"
        try:
            r=requests.get(url, timeout=12)
            if r.status_code!=200: continue
            for j in (r.json() or []):
                out.append(_ensure_fields({
                    "id": str(uuid.uuid4()),
                    "company": s.upper(),
                    "title": j.get("text") or "",
                    "jd_url": j.get("hostedUrl") or "",
                    "location": (j.get("categories") or {}).get("location",""),
                    "type": ((j.get("categories") or {}).get("commitment") or "").lower(),
                    "work_mode": "",
                    "posted_at": (j.get("createdAt") and datetime.utcfromtimestamp(int(j["createdAt"])/1000).strftime("%Y-%m-%d")) or "",
                    "currency": "",
                    "salary_min": "",
                    "salary_max": "",
                    "source": "lever",
                    "tags": [s] + list(filter(None, [(j.get("categories") or {}).get("team")])),
                    "notes": ""
                }))
        except Exception as e:
            print("lever error:", s, e)
    return out

def parse_workday_endpoint(u):
    # 例：https://beigene.wd5.myworkdayjobs.com/en-US/BeiGene
    m = re.search(r'https?://([\w\-]+)\.wd\d+\.myworkdayjobs\.com/(?:[a-z]{2}-[A-Z]{2}/)?([^/]+)', u)
    if not m: return None
    host = m.group(1); site = m.group(2)
    return f"https://{host}.wd5.myworkdayjobs.com/wday/cxs/{host}/{site}/jobs"

def import_workday(urls, limit=600):
    out=[]
    for line in urls:
        line=line.strip()
        if not line: continue
        api=parse_workday_endpoint(line)
        if not api:
            print("workday parse fail:", line); continue
        offset=0
        try:
            for _ in range(0, limit, 50):
                q=f"{api}?limit=50&offset={offset}"
                r=requests.get(q, timeout=12)
                if r.status_code!=200: break
                data=r.json() or {}
                arr=data.get("jobPostings") or data.get("data") or []
                if not arr: break
                for j in arr:
                    title = j.get("title") or ""
                    url  = j.get("externalPath") or j.get("externalPathUrl") or j.get("canonicalPositionUrl")
                    if url and not url.startswith("http"):
                        base=line.rstrip("/")
                        url=base+url
                    out.append(_ensure_fields({
                        "id": str(uuid.uuid4()),
                        "company": urlparse(line).hostname.split(".")[0].upper(),
                        "title": title,
                        "jd_url": url or line,
                        "location": j.get("locationsText") or j.get("locations") or "",
                        "type": "full",
                        "work_mode": "",
                        "posted_at": "",
                        "currency": "",
                        "salary_min": "",
                        "salary_max": "",
                        "source": "workday",
                        "tags": ["workday"],
                        "notes": ""
                    }))
                offset += 50
        except Exception as e:
            print("workday error:", line, e)
    return out

# -------------------- 静态/健康 --------------------
@app.get("/")
def index():
    return send_from_directory("public","agent.html")

@app.get("/public/<path:p>")
def pub(p):
    return send_from_directory("public", p)

@app.get("/health")
def health():
    return json_response({"status":"ok","time":now_ts()})

# -------------------- 上传简历解析 --------------------
@app.post("/api/upload_resume")
def api_upload_resume():
    if "file" not in request.files:
        return json_response({"ok":False, "error":"no file"}, 400)
    f = request.files["file"]
    b = f.read()
    name = (f.filename or "").lower()
    text = ""
    if name.endswith(".pdf"):
        text = extract_text_from_pdf(b)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(b)
    else:
        text = extract_text_from_txt(b)
    return json_response({"ok":True, "text": text.strip()})

# -------------------- 画像/建议 --------------------
@app.post("/api/profile")
def api_profile():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text: return json_response({"ok":False,"error":"empty"},400)
    prof = build_profile_with_llm(text)
    return json_response({"ok":True, "profile": prof})

# -------------------- 职位列表（分页/筛选/排序） --------------------
@app.get("/api/jobs")
def api_jobs():
    q   = (request.args.get("q") or "").lower()
    tp  = (request.args.get("type") or "").lower()   # full/part/project
    wm  = (request.args.get("work_mode") or "").lower()
    loc = (request.args.get("location") or "").lower()
    tag = (request.args.get("tag") or "").lower()
    sort= (request.args.get("sort") or "posted_desc").lower()

    page = max(1, int(request.args.get("page","1")))
    size = max(1, min(200, int(request.args.get("page_size","20"))))

    jobs = read_jobs()
    # 过滤
    def ok(j):
        if q and (q not in (j.get("title") or "").lower()) and (q not in (j.get("company") or "").lower()):
            return False
        if tp and (j.get("type","").lower()!=tp): return False
        if wm and (j.get("work_mode","").lower()!=wm): return False
        if loc and (loc not in (j.get("location","").lower())): return False
        if tag and tag not in [t.lower() for t in (j.get("tags") or [])]: return False
        return True
    jobs=[j for j in jobs if ok(j)]

    # 排序
    if sort=="posted_desc":
        jobs.sort(key=lambda x:(x.get("posted_at") or "", x.get("id")), reverse=True)
    else:
        jobs.sort(key=lambda x:x.get("id"), reverse=True)

    total=len(jobs)
    start=(page-1)*size; end=start+size
    return json_response({"items":jobs[start:end],"page":page,"page_size":size,"total":total})

# -------------------- 单条职位 --------------------
@app.get("/api/job")
def api_job():
    id_ = request.args.get("id","")
    for j in read_jobs():
        if j.get("id")==id_:
            return json_response({"ok":True,"item":j})
    return json_response({"ok":False,"error":"not found"},404)

# -------------------- 推荐（严格过滤 + 启发式打分） --------------------
@app.post("/api/analyze_recommend")
def api_analyze_recommend():
    data=request.get_json(silent=True) or {}
    resume = (data.get("resume") or "").strip()
    kw     = (data.get("keywords") or "").strip()
    loc    = (data.get("location") or "").strip().lower()
    page   = int(data.get("page") or 1)
    size   = max(5, min(100, int(data.get("page_size") or 20)))
    type_  = (data.get("type") or "").strip().lower()

    kws = extract_keywords(resume, kw)

    jobs = read_jobs()
    cand=[]
    for j in jobs:
        if type_ and (j.get("type","").lower()!=type_): continue
        if not strict_match_filter(kws, j): continue
        s = heuristic_score(kws, j, loc_pref=loc)
        if s<30: continue
        j2=j.copy(); j2["score"]=s
        cand.append(j2)

    cand.sort(key=lambda x:(x["score"], x.get("posted_at") or ""), reverse=True)
    total=len(cand)
    start=(page-1)*size; end=start+size
    items=cand[start:end]

    top3 = ", ".join(kws[:3]) if kws else "综合背景"
    summary = f"这位候选人在「{top3}」方向具备可迁移经验。建议突出量化成果与落地成效，围绕目标岗位补强关键技能。"

    return json_response({"items":items,"total":total,"profile":{"summary":summary,"keywords":kws[:15]}})

# -------------------- 批量精评（自动刷新分数） --------------------
@app.post("/api/match_batch")
def api_match_batch():
    data=request.get_json(silent=True) or {}
    ids=list(dict.fromkeys(data.get("ids") or []))[:80]
    resume=(data.get("resume") or "").strip()
    all_jobs={j["id"]:j for j in read_jobs()}
    out=[]
    for i in ids:
        j=all_jobs.get(i); 
        if not j: continue
        base=heuristic_score(extract_keywords(resume), j)
        llm = llm_score_fit(resume, j)
        score = max(0, min(100, llm if llm>=0 else base))
        out.append({"id":i,"score":score})
        time.sleep(0.02)
    return json_response({"items":out})

# -------------------- ATS/Workday 导入 --------------------
@app.post("/api/ingest_ats")
def api_ingest_ats():
    token = request.headers.get("X-Admin-Token","").strip() or (request.get_json(silent=True) or {}).get("admin_token","").strip()
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return _json_response({"ok": False, "error":"unauthorized"}, 401)

    data=request.get_json(silent=True) or {}
    gh = [x.strip() for x in (data.get("greenhouse") or "").split(",") if x.strip()]
    lv = [x.strip() for x in (data.get("lever") or "").split(",") if x.strip()]

    imported=[]
    if gh: imported += import_greenhouse(gh)
    if lv: imported += import_lever(lv)

    wrote = append_jobs(imported) if imported else 0
    return _json_response({"ok": True, "ingested": wrote, "detail": {"greenhouse":len([1 for x in imported if x["source"]=="greenhouse"]), "lever":len([1 for x in imported if x["source"]=="lever"])}})

@app.post("/api/ingest_cn_jobs")
def api_ingest_cn():
    token = request.headers.get("X-Admin-Token","").strip() or (request.get_json(silent=True) or {}).get("admin_token","").strip()
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        return _json_response({"ok": False, "error":"unauthorized"}, 401)

    data=request.get_json(silent=True) or {}
    urls=[x.strip() for x in (data.get("workday_urls") or "").splitlines() if x.strip()]
    imported=import_workday(urls)
    wrote=append_jobs(imported) if imported else 0
    return _json_response({"ok": True, "ingested": wrote, "detail": {"workday":len(imported)}})

# -------------------- 事件追踪（可选） --------------------
def append_track(evt: dict):
    arr=[]
    if os.path.exists(TRACKER_JSON):
        try:
            with open(TRACKER_JSON,"r",encoding="utf-8") as f:
                arr=json.load(f)
        except Exception:
            arr=[]
    evt["ts"]=now_ts()
    arr.append(evt)
    with open(TRACKER_JSON,"w",encoding="utf-8") as f:
        json.dump(arr,f,ensure_ascii=False,indent=2)

@app.post("/api/track")
def api_track():
    data=request.get_json(silent=True) or {}
    append_track({"type":data.get("type","unknown"),"detail":data})
    return json_response({"ok":True})

# -------------------- 启动 --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
