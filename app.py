import os, csv, io, re, json, time, uuid, random, logging, shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
from flask import Flask, request, jsonify, send_from_directory, redirect

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(APP_ROOT, 'public')
DATA_DIR = os.path.join(APP_ROOT, 'data')
JOBS_CSV = os.path.join(DATA_DIR, 'jobs.csv')
JOBS_SEED_CSV = os.path.join(DATA_DIR, 'jobs_seed.csv')  # 可放入你离线导出的4K职位
SEED_JSON = os.path.join(DATA_DIR, 'seed_sources.json')
APPLICATIONS_CSV = os.path.join(DATA_DIR, 'applications.csv')
CANDIDATES_CSV = os.path.join(DATA_DIR, 'candidates.csv')

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('lgwork')

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='/public')

# ==== LLM（后端调用，前端不暴露）====
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY')
DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

# 更像真实浏览器，且缩短超时，避免 Render 上 gunicorn 超时
UA = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/json;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.8,zh;q=0.7'
}
REQ_TIMEOUT = 6  # 秒

# ---- 内置默认 seeds（seed_sources.json 缺失/为空时使用） ----
DEFAULT_SEEDS = {
    "greenhouse": [
        "stripe","databricks","asana","affirm","discord","coinbase","pinterest","dropbox","cloudflare",
        "lyft","instacart","reddit","brex","figma","airbnb","10xgenomics","benchling","scaleai","anduril",
        "snyk","grafana","hashicorp","vercel","samsara","datadog","okta","rubrik","squarespace","hubspot",
        "mongodb","netlify","snowflake","palantir","unity","confluent","zendesk","square","wise","revolut"
    ],
    "lever": [
        "reddit","turo","postman","benchling","sofi","niantic","loom","planet","opentable","remitly",
        "affirm","bigcommerce","hopper","xero","grab","quora","gusto","doordash","planetscale","fivetran",
        "retool","checkr","launchdarkly","airtable","figma","grammarly","carta","pilot","miro","intercom"
    ]
}

# ================== 工具函数 ==================

def strip_html(raw_html: str) -> str:
    try:
        return re.sub('<[^<]+?>', '', raw_html or '').replace('\xa0', ' ').strip()
    except Exception:
        return raw_html or ''


def save_jobs_to_csv(rows: List[Dict[str, Any]]):
    fieldnames = ['id','source','company','title','location','remote','level','salary','jd_url','description','tags','posted_at']
    with open(JOBS_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = r.copy()
            if isinstance(row.get('tags'), (list, dict)):
                row['tags'] = json.dumps(row['tags'], ensure_ascii=False)
            w.writerow({k: row.get(k, '') for k in fieldnames})


def read_jobs_from_csv() -> List[Dict[str, Any]]:
    path = JOBS_CSV if os.path.exists(JOBS_CSV) else (JOBS_SEED_CSV if os.path.exists(JOBS_SEED_CSV) else None)
    out = []
    if not path:
        return out
    with open(path, 'r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                r['tags'] = json.loads(r.get('tags') or '[]')
            except Exception:
                r['tags'] = []
            out.append(r)
    return out


def zero_exp_friendly(text: str) -> bool:
    t = (text or '').lower()
    pats = ['no prior experience','no experience required','career switch','recent graduates','fresh graduates','training provided','willingness to learn','entry level','junior role','欢迎转行','应届','可培养']
    return any(p in t for p in pats)


def detect_remote(text: str) -> bool:
    t = (text or '').lower()
    return any(p in t for p in ['remote','hybrid','在家办公','远程'])


def http_get(url: str, expect_json: bool = False):
    try:
        r = requests.get(url, headers=UA, timeout=REQ_TIMEOUT)
        if r.status_code == 200:
            return r.json() if expect_json else r.text
        logger.warning(f"GET {url} -> {r.status_code}")
    except Exception as e:
        logger.warning(f"GET {url} failed: {e}")
    return None

# ================== Greenhouse 抓取（带 EU 站 + token 自动发现） ==================

def gh_discover_token(slug: str) -> Optional[str]:
    """依次尝试 US/EU 站点，解析嵌入的 job_board?for=<token> 或 data 属性中的 apiToken。"""
    for host in ("boards.greenhouse.io", "boards.eu.greenhouse.io"):
        html = http_get(f"https://{host}/{slug}", expect_json=False)
        if not html:
            continue
        m = re.search(r'job_board\?for=([a-z0-9\-]+)', html, flags=re.I)
        if m:
            return m.group(1).lower()
        m2 = re.search(r'\"apiToken\"\s*:\s*\"([a-z0-9\-]+)\"', html, flags=re.I)
        if m2:
            return m2.group(1).lower()
    return None


def fetch_greenhouse(board: str) -> List[Dict[str, Any]]:
    def _fetch(token: str) -> List[Dict[str, Any]]:
        data = http_get(f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true", expect_json=True)
        res = []
        if not data or 'jobs' not in data:
            return res
        for j in data['jobs']:
            try:
                jd_url = j.get('absolute_url') or ''
                title = j.get('title') or ''
                location = (j.get('location') or {}).get('name','')
                desc = strip_html(j.get('content') or '')
                tags = []
                if zero_exp_friendly(desc): tags.append('zero_exp_friendly')
                if detect_remote(desc) or 'remote' in (location or '').lower(): tags.append('remote')
                res.append({
                    'id': f"gh-{j.get('id')}", 'source':'greenhouse', 'company': token, 'title': title,
                    'location': location, 'remote': 'remote' in tags, 'level':'', 'salary':'', 'jd_url': jd_url,
                    'description': desc[:5000], 'tags': tags, 'posted_at': j.get('updated_at') or j.get('created_at') or ''
                })
            except Exception:
                continue
        return res

    rows = _fetch(board)
    if rows:
        return rows
    token = gh_discover_token(board)
    if token and token != board:
        logger.info(f"Greenhouse token discovered: {board} -> {token}")
        return _fetch(token)
    return []

# ================== Lever 抓取 ==================

def fetch_lever(company: str) -> List[Dict[str, Any]]:
    data = http_get(f"https://api.lever.co/v0/postings/{company}?mode=json", expect_json=True)
    res = []
    if not data or (isinstance(data, dict) and data.get('error')):
        return res
    if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
        iterable = data['data']
    else:
        iterable = data if isinstance(data, list) else []
    for j in iterable:
        try:
            jd_url = j.get('hostedUrl') or j.get('applyUrl') or ''
            title = j.get('text') or ''
            location = (j.get('categories') or {}).get('location','')
            desc = strip_html(j.get('listsPlain') or j.get('descriptionPlain') or j.get('description') or '')
            tags = []
            if zero_exp_friendly(desc): tags.append('zero_exp_friendly')
            if detect_remote(desc) or 'remote' in (location or '').lower(): tags.append('remote')
            res.append({
                'id': f"lv-{j.get('id')}", 'source':'lever', 'company': company, 'title': title,
                'location': location, 'remote': 'remote' in tags,
                'level': (j.get('categories') or {}).get('commitment',''), 'salary':'', 'jd_url': jd_url,
                'description': desc[:5000], 'tags': tags, 'posted_at': j.get('createdAt') or ''
            })
        except Exception:
            continue
    return res

# ================== 种子与刷新（分片） ==================

def load_seeds() -> Dict[str, List[str]]:
    try:
        if os.path.exists(SEED_JSON):
            with open(SEED_JSON, 'r', encoding='utf-8') as f:
                txt = f.read().strip()
                if txt:
                    data = json.loads(txt)
                    if isinstance(data, dict) and (data.get('greenhouse') or data.get('lever')):
                        return data
                logger.warning('seed_sources.json 为空或无效，使用内置 DEFAULT_SEEDS')
    except Exception as e:
        logger.warning(f'read seeds error: {e}, use DEFAULT_SEEDS')
    return DEFAULT_SEEDS


def refresh_chunk(source: str = 'both', start: int = 0, count: int = 20) -> Dict[str, Any]:
    seeds = load_seeds()
    keys = []
    if source in ('greenhouse','both'):
        keys += [('greenhouse', x) for x in seeds.get('greenhouse', [])]
    if source in ('lever','both'):
        keys += [('lever', x) for x in seeds.get('lever', [])]

    total_keys = len(keys)
    part = keys[start: start + count]

    existing = read_jobs_from_csv()
    bykey = {(r.get('jd_url') or r.get('id')): r for r in existing}

    used = []
    for typ, key in part:
        try:
            rows = fetch_greenhouse(key) if typ == 'greenhouse' else fetch_lever(key)
            if rows:
                used.append({"type": typ, "key": key, "count": len(rows)})
                for r in rows:
                    k = r.get('jd_url') or r.get('id')
                    if k: bykey[k] = r
        except Exception as e:
            logger.warning(f"refresh_chunk error {typ}:{key} -> {e}")

    all_rows = list(bykey.values())
    random.shuffle(all_rows)
    save_jobs_to_csv(all_rows)
    return {
        "ok": True,
        "processed": len(part),
        "start": start,
        "count": count,
        "total_keys": total_keys,
        "sources": used,
        "total_jobs": len(all_rows),
        "done": (start + count) >= total_keys
    }

# ================== 解析 / 匹配 / 生成 ==================

def _try_imports():
    mods = {}
    try:
        import PyPDF2; mods['PyPDF2'] = PyPDF2
    except Exception: mods['PyPDF2'] = None
    try:
        import docx; mods['docx'] = docx
    except Exception: mods['docx'] = None
    return mods

IMPORTS = _try_imports()


def extract_text_from_file(fs) -> str:
    if not fs:
        return ''
    name = (fs.filename or '').lower()
    data = fs.read()
    fs.stream.seek(0)
    if name.endswith('.pdf') and IMPORTS['PyPDF2']:
        try:
            reader = IMPORTS['PyPDF2'].PdfReader(io.BytesIO(data))
            return '\n'.join([(p.extract_text() or '') for p in reader.pages])
        except Exception:
            pass
    if name.endswith('.docx') and IMPORTS['docx']:
        try:
            doc = IMPORTS['docx'].Document(io.BytesIO(data))
            return '\n'.join(p.text for p in doc.paragraphs)
        except Exception:
            pass
    try:
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def call_llm(system_prompt: str, user_prompt: str, temperature=0.2, max_tokens=1200) -> str:
    if not DEEPSEEK_API_KEY:
        return ''
    try:
        url = f"{DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
        payload = {"model": DEEPSEEK_MODEL, "messages": [{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            j = r.json(); return (j.get('choices',[{}])[0].get('message',{}).get('content') or '').strip()
        logger.warning(f"LLM non-200: {r.status_code} {r.text[:180]}")
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
    return ''


def summarize_resume(raw: str, lang: str = 'zh') -> str:
    if not raw:
        return ''
    sys_p = "You analyze resumes and return a concise, ATS-friendly profile with bullet points."
    usr_p = (f"LANG={lang}. Extract: 1) Professional Summary (<=120 words). 2) Key Skills (<=15). 3) Experience Highlights (<=8). 4) Education.\n\nResume:\n{raw[:15000]}")
    ans = call_llm(sys_p, usr_p)
    return ans or raw[:600]


def optimize_resume(profile: str, lang: str='zh') -> str:
    if not profile:
        return ''
    sys_p = "You are a resume coach. Improve wording, quantify results, add keywords for ATS, keep it concise and professional."
    usr_p = f"LANG={lang}. Improve the following resume content and return an optimized version.\n\n{profile[:8000]}"
    ans = call_llm(sys_p, usr_p, temperature=0.3, max_tokens=1500)
    return ans or profile


def generate_ats_resume(resume_profile: str, job_desc: str, lang: str='zh', template: str='1') -> str:
    sys_p = "You produce an ATS-friendly resume tailored to the job, with quantifiable bullets, clean sections, and keywords matching the JD."
    usr_p = (f"LANG={lang}. TEMPLATE={template}. Based on the candidate profile and the job description, write a tailored resume.\n\nPROFILE:\n{resume_profile[:6000]}\n\nJD:\n{job_desc[:6000]}\n\nReturn a complete resume in plain text sections: SUMMARY, SKILLS, EXPERIENCE, EDUCATION, PROJECTS (optional).")
    ans = call_llm(sys_p, usr_p, temperature=0.35, max_tokens=1600)
    return ans or (resume_profile[:1200] if resume_profile else '')


def match_rule_overlap(a: str, b: str) -> int:
    def tok(t):
        t = (t or '').lower(); t = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]+"," ",t)
        return set([w for w in t.split() if len(w)>2])
    return len(tok(a) & tok(b))


def match_with_llm(profile: str, jd: str, title: str='') -> Dict[str, Any]:
    base = min(60, match_rule_overlap(profile, jd + ' ' + title))
    sys_p = "You are a job-match evaluator. Score 0-100 with reasoning JSON."
    usr_p = (f"Job Title: {title}\nJD:\n{jd[:6000]}\n\nCandidate Profile:\n{profile[:6000]}\n\nReturn JSON keys: score(0-100), top_matches(<=5), gaps(<=3), advice(<=3).")
    llm = call_llm(sys_p, usr_p)
    score2, analysis = 0, {}
    if llm:
        try:
            import json as _json
            m = re.search(r"\{[\s\S]*\}", llm)
            if m:
                analysis = _json.loads(m.group(0))
                score2 = int(analysis.get('score', 0))
            else:
                analysis = {"advice":[llm[:400]]}
        except Exception:
            analysis = {"advice":[llm[:400]]}
    final = int(0.6*base + 0.4*(score2 or base))
    return {"score": max(1, min(100, final)), "rule_overlap": base, "analysis": analysis}


def topk_match(profile: str, jobs: List[Dict[str,Any]], k: int=8) -> List[Dict[str,Any]]:
    if not profile:
        return []
    jobs0 = sorted(jobs, key=lambda r: match_rule_overlap(profile, (r.get('title','')+' '+r.get('description',''))), reverse=True)[:20]
    scored = []
    for r in jobs0:
        res = match_with_llm(profile, r.get('description',''), r.get('title',''))
        scored.append((res['score'], r, res))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, r, res in scored[:k]:
        item = r.copy(); item['match'] = res; out.append(item)
    return out

# =============== 候选人池（企业端 Demo） ==================

def _ensure_candidates_header():
    if not os.path.exists(CANDIDATES_CSV):
        with open(CANDIDATES_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['id','name','email','tags','profile','ts'])
            w.writeheader()


def read_candidates() -> List[Dict[str,Any]]:
    if not os.path.exists(CANDIDATES_CSV):
        return []
    out = []
    with open(CANDIDATES_CSV, 'r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                r['tags'] = json.loads(r.get('tags') or '[]')
            except Exception:
                r['tags'] = []
            out.append(r)
    return out

# ================== 路由 ==================
@app.route('/')
def root_index():
    return redirect('/public/index.html', code=302)

@app.route('/public/<path:p>')
def serve_public(p):
    return send_from_directory(PUBLIC_DIR, p)

# Jobs 列表
@app.route('/api/jobs/list')
def api_jobs_list():
    rows = read_jobs_from_csv()
    q = (request.args.get('q') or '').lower()
    location = (request.args.get('location') or '').lower()
    remote = request.args.get('remote')
    zero = request.args.get('zero_exp')
    limit = int(request.args.get('limit') or 300)

    def hit(r):
        txt = ' '.join([r.get('title',''), r.get('company',''), r.get('location',''), r.get('description','')]).lower()
        if q and q not in txt: return False
        if location and location not in (r.get('location','') or '').lower(): return False
        if remote in ('true','false'):
            flag = str(r.get('remote','')).lower() in ('true','1','yes')
            if (remote=='true' and not flag) or (remote=='false' and flag): return False
        if zero in ('true','false'):
            has = 'zero_exp_friendly' in (r.get('tags') or [])
            if (zero=='true' and not has) or (zero=='false' and has): return False
        return True

    rows = [r for r in rows if hit(r)]
    return jsonify({"ok": True, "count": len(rows), "items": rows[:limit]})

# Jobs 刷新（一次性，可能超时）
@app.route('/api/jobs/refresh', methods=['POST'])
def api_jobs_refresh():
    # 兼容老接口，内部走分片直到完成，但每片很小，尽量避免超时
    start = 0
    total = 0
    used_all = []
    while True:
        info = refresh_chunk('both', start=start, count=15)
        total = info.get('total_jobs', total)
        used_all += info.get('sources', [])
        if info.get('done'): break
        start += 15
        # 防止单请求过长
        if start >= 120: break
    return jsonify({"ok": True, "count": total, "sources": used_all})

# Jobs 分片刷新（推荐前端循环调用）
@app.route('/api/jobs/refresh_chunk', methods=['POST'])
def api_jobs_refresh_chunk():
    source = request.args.get('source','both')
    start = int(request.args.get('start') or 0)
    count = max(1, min(int(request.args.get('count') or 20), 50))
    info = refresh_chunk(source, start, count)
    return jsonify(info)

# Jobs 导入离线种子（jobs_seed.csv -> jobs.csv）
@app.route('/api/jobs/import_static', methods=['POST'])
def api_jobs_import_static():
    if not os.path.exists(JOBS_SEED_CSV):
        return jsonify({"ok": False, "error": "data/jobs_seed.csv 不存在"}), 400
    shutil.copyfile(JOBS_SEED_CSV, JOBS_CSV)
    rows = read_jobs_from_csv()
    return jsonify({"ok": True, "count": len(rows)})

# Resume parse & optimize & generate
@app.route('/api/resume/parse', methods=['POST'])
def api_resume_parse():
    lang = (request.form.get('lang') or 'zh').lower()
    file = request.files.get('file')
    raw = extract_text_from_file(file) if file else (request.form.get('text') or '')
    profile = summarize_resume(raw, lang=lang)
    return jsonify({"ok": True, "raw_len": len(raw), "profile": profile})

@app.route('/api/resume/optimize', methods=['POST'])
def api_resume_optimize():
    data = request.get_json(force=True)
    lang = (data.get('lang') or 'zh').lower()
    text = data.get('text') or ''
    out = optimize_resume(text, lang=lang)
    return jsonify({"ok": True, "text": out})

@app.route('/api/resume/generate', methods=['POST'])
def api_resume_generate():
    data = request.get_json(force=True)
    resume_profile = data.get('resume_profile') or ''
    job_desc = data.get('job_desc') or ''
    lang = (data.get('lang') or 'zh').lower()
    tpl = str(data.get('template') or '1')
    out = generate_ats_resume(resume_profile, job_desc, lang=lang, template=tpl)
    return jsonify({"ok": True, "text": out})

# JD parse & generate（B 端）
@app.route('/api/jd/parse', methods=['POST'])
def api_jd_parse():
    lang = (request.form.get('lang') or 'zh').lower()
    file = request.files.get('file')
    raw = extract_text_from_file(file) if file else (request.form.get('text') or '')
    sys_p = "You are a JD analyst. Extract role, responsibilities, must-have, nice-to-have, location, level, salary range (if any). Return concise bullets."
    usr_p = f"LANG={lang}. Analyze this JD and return the structured summary:\n\n{raw[:12000]}"
    out = call_llm(sys_p, usr_p)
    return jsonify({"ok": True, "jd_summary": out or (raw[:800] if raw else '')})

@app.route('/api/jd/generate', methods=['POST'])
def api_jd_generate():
    data = request.get_json(force=True)
    lang = (data.get('lang') or 'zh').lower()
    need = data.get('need') or ''
    sys_p = "You are a hiring manager assistant. Write a clear, ATS-friendly JD with sections: About the Role, Responsibilities (6-10 bullets), Requirements (must-have vs nice-to-have), Location/Remote, Preferred background."
    usr_p = f"LANG={lang}. Based on this need, write a JD.\n\n{need[:6000]}"
    out = call_llm(sys_p, usr_p, temperature=0.35, max_tokens=1400)
    return jsonify({"ok": True, "jd": out or need})

# TopK 匹配
@app.route('/api/match/topk', methods=['POST'])
def api_match_topk():
    data = request.get_json(force=True)
    profile = data.get('profile') or ''
    k = int(data.get('k') or 8)
    jobs = read_jobs_from_csv()
    out = topk_match(profile, jobs, k=k)
    return jsonify({"ok": True, "items": out, "k": k})

# 候选人：上传 & 列表 & 匹配 JD
@app.route('/api/candidates/upload', methods=['POST'])
def api_candidates_upload():
    _ensure_candidates_header()
    name = request.form.get('name') or ''
    email = request.form.get('email') or ''
    tags = request.form.get('tags') or ''
    file = request.files.get('file')
    raw = extract_text_from_file(file) if file else (request.form.get('text') or '')
    prof = summarize_resume(raw, lang=(request.form.get('lang') or 'zh'))
    row = {
        'id': str(uuid.uuid4()),
        'name': name,
        'email': email,
        'tags': json.dumps([t.strip() for t in tags.split(',') if t.strip()], ensure_ascii=False),
        'profile': prof,
        'ts': datetime.utcnow().isoformat()+ 'Z'
    }
    need_header = not os.path.exists(CANDIDATES_CSV) or os.path.getsize(CANDIDATES_CSV) == 0
    with open(CANDIDATES_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id','name','email','tags','profile','ts'])
        if need_header: w.writeheader()
        w.writerow(row)
    return jsonify({"ok": True, "item": row})

@app.route('/api/candidates/list')
def api_candidates_list():
    return jsonify({"ok": True, "items": read_candidates()})

@app.route('/api/candidates/match', methods=['POST'])
def api_candidates_match():
    data = request.get_json(force=True)
    jd_text = data.get('jd_text') or ''
    k = int(data.get('k') or 8)
    cands = read_candidates()
    scored = []
    for c in cands:
        res = match_with_llm(c.get('profile',''), jd_text, '')
        scored.append((res['score'], c, res))
    scored.sort(key=lambda x: x[0], reverse=True)
    items = []
    for s, c, res in scored[:k]:
        it = c.copy(); it['match'] = res; items.append(it)
    return jsonify({"ok": True, "items": items})

# 手动投递记录
@app.route('/api/apply/log', methods=['POST'])
def api_apply_log():
    d = request.get_json(force=True)
    row = { 'id': str(uuid.uuid4()), 'user': d.get('user') or 'demo', 'job_url': d.get('job_url') or '', 'title': d.get('title') or '', 'company': d.get('company') or '', 'ts': datetime.utcnow().isoformat()+'Z', 'note': d.get('note') or 'manual' }
    need = not os.path.exists(APPLICATIONS_CSV)
    with open(APPLICATIONS_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if need: w.writeheader()
        w.writerow(row)
    return jsonify({"ok": True, "item": row})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
