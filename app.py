import os
import csv
import io
import re
import json
import time
import html
import uuid
import queue
import string
import random
import logging
from datetime import datetime
from typing import List, Dict, Any

import requests
from flask import Flask, request, jsonify, send_from_directory, redirect, Response

# =============== 基础配置 ===============
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(APP_ROOT, 'public')
DATA_DIR = os.path.join(APP_ROOT, 'data')
JOBS_CSV = os.path.join(DATA_DIR, 'jobs.csv')
SEED_JSON = os.path.join(DATA_DIR, 'seed_sources.json')
APPLICATIONS_CSV = os.path.join(DATA_DIR, 'applications.csv')

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lgwork")

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path='/public')

# LLM 环境（不在前端暴露）
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or os.getenv('LLM_API_KEY')
DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')

# =============== 工具函数 ===============

def strip_html(raw_html: str) -> str:
    try:
        # 粗糙去标签，够用
        return re.sub('<[^<]+?>', '', raw_html or '').replace('\xa0', ' ').strip()
    except Exception:
        return raw_html or ''


def save_jobs_to_csv(rows: List[Dict[str, Any]]):
    fieldnames = [
        'id', 'source', 'company', 'title', 'location', 'remote', 'level',
        'salary', 'jd_url', 'description', 'tags', 'posted_at'
    ]
    with open(JOBS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r = r.copy()
            if isinstance(r.get('tags'), (list, dict)):
                r['tags'] = json.dumps(r['tags'], ensure_ascii=False)
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def read_jobs_from_csv() -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(JOBS_CSV):
        return rows
    with open(JOBS_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                tags = r.get('tags')
                if tags:
                    try:
                        r['tags'] = json.loads(tags)
                    except Exception:
                        r['tags'] = []
                else:
                    r['tags'] = []
            except Exception:
                r['tags'] = []
            rows.append(r)
    return rows


def zero_exp_friendly(text: str) -> bool:
    t = (text or '').lower()
    patterns = [
        'no prior experience', 'no experience required', 'career switch',
        'recent graduates welcome', 'fresh graduates', 'training provided',
        'willingness to learn', 'entry level', 'junior role', '欢迎转行', '应届', '可培养'
    ]
    return any(p in t for p in patterns)


def detect_remote(text: str) -> bool:
    t = (text or '').lower()
    return any(p in t for p in ['remote', 'hybrid', '在家办公', '远程'])


def safe_get(url: str, timeout=20):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        logger.warning(f"GET {url} -> {r.status_code}")
    except Exception as e:
        logger.warning(f"GET {url} failed: {e}")
    return None


# =============== 抓取 Greenhouse / Lever ===============

def fetch_greenhouse(board: str) -> List[Dict[str, Any]]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
    data = safe_get(url)
    results = []
    if not data or 'jobs' not in data:
        return results
    for j in data['jobs']:
        try:
            jd_url = j.get('absolute_url') or ''
            title = j.get('title') or ''
            company = board
            location = (j.get('location') or {}).get('name', '')
            desc = strip_html(j.get('content') or '')
            posted = j.get('updated_at') or j.get('created_at') or ''
            tags = []
            if zero_exp_friendly(desc):
                tags.append('zero_exp_friendly')
            if detect_remote(desc) or 'remote' in (location or '').lower():
                tags.append('remote')
            results.append({
                'id': f"gh-{j.get('id')}",
                'source': 'greenhouse',
                'company': company,
                'title': title,
                'location': location,
                'remote': 'remote' in tags,
                'level': '',
                'salary': '',
                'jd_url': jd_url,
                'description': desc[:5000],
                'tags': tags,
                'posted_at': posted,
            })
        except Exception:
            continue
    return results


def fetch_lever(company: str) -> List[Dict[str, Any]]:
    url = f"https://api.lever.co/v0/postings/{company}?mode=json"
    data = safe_get(url)
    results = []
    if not data:
        return results
    for j in data:
        try:
            jd_url = j.get('hostedUrl') or j.get('applyUrl') or ''
            title = j.get('text') or ''
            company_name = company
            location = ''
            if j.get('categories'):
                location = j['categories'].get('location') or ''
            desc = strip_html(j.get('listsPlain') or j.get('descriptionPlain') or j.get('description') or '')
            posted = j.get('createdAt') or ''
            tags = []
            if zero_exp_friendly(desc):
                tags.append('zero_exp_friendly')
            if detect_remote(desc) or 'remote' in (location or '').lower():
                tags.append('remote')
            results.append({
                'id': f"lv-{j.get('id')}",
                'source': 'lever',
                'company': company_name,
                'title': title,
                'location': location,
                'remote': 'remote' in tags,
                'level': (j.get('categories') or {}).get('commitment', ''),
                'salary': '',
                'jd_url': jd_url,
                'description': desc[:5000],
                'tags': tags,
                'posted_at': posted,
            })
        except Exception:
            continue
    return results


def refresh_jobs_from_seed() -> Dict[str, Any]:
    if not os.path.exists(SEED_JSON):
        return {"count": 0, "sources": []}
    with open(SEED_JSON, 'r', encoding='utf-8') as f:
        seeds = json.load(f)
    all_rows = []
    seen = set()
    sources_used = []

    for gh in seeds.get('greenhouse', []):
        rows = fetch_greenhouse(gh)
        if rows:
            sources_used.append({"type": "greenhouse", "board": gh, "count": len(rows)})
        for r in rows:
            key = r.get('jd_url') or r.get('id')
            if not key or key in seen:
                continue
            seen.add(key)
            all_rows.append(r)

    for lv in seeds.get('lever', []):
        rows = fetch_lever(lv)
        if rows:
            sources_used.append({"type": "lever", "company": lv, "count": len(rows)})
        for r in rows:
            key = r.get('jd_url') or r.get('id')
            if not key or key in seen:
                continue
            seen.add(key)
            all_rows.append(r)

    random.shuffle(all_rows)
    save_jobs_to_csv(all_rows)
    return {"count": len(all_rows), "sources": sources_used}


# =============== 简历解析 & 匹配（LLM 可选） ===============

def _try_imports():
    mods = {}
    try:
        import PyPDF2  # type: ignore
        mods['PyPDF2'] = PyPDF2
    except Exception:
        mods['PyPDF2'] = None
    try:
        import docx  # python-docx
        mods['docx'] = docx
    except Exception:
        mods['docx'] = None
    return mods


IMPORTS = _try_imports()


def extract_text_from_file(file_storage) -> str:
    filename = (file_storage.filename or '').lower()
    data = file_storage.read()
    file_storage.stream.seek(0)

    # PDF
    if filename.endswith('.pdf') and IMPORTS.get('PyPDF2') is not None:
        try:
            reader = IMPORTS['PyPDF2'].PdfReader(io.BytesIO(data))
            text = []
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or '')
                except Exception:
                    pass
            return '\n'.join(text).strip()
        except Exception:
            pass

    # DOCX
    if filename.endswith('.docx') and IMPORTS.get('docx') is not None:
        try:
            doc = IMPORTS['docx'].Document(io.BytesIO(data))
            return '\n'.join([p.text for p in doc.paragraphs]).strip()
        except Exception:
            pass

    # 兜底：尝试按 utf-8 解码
    try:
        return data.decode('utf-8', errors='ignore')
    except Exception:
        return ''


def call_deepseek_chat(system_prompt: str, user_prompt: str, temperature=0.2, max_tokens=1200) -> str:
    if not DEEPSEEK_API_KEY:
        return ''
    try:
        url = f"{DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            return (data.get('choices', [{}])[0].get('message', {}).get('content') or '').strip()
        logger.warning(f"DeepSeek non-200: {r.status_code} {r.text[:200]}")
    except Exception as e:
        logger.warning(f"DeepSeek call failed: {e}")
    return ''


def summarize_resume(text: str, lang: str = 'zh') -> Dict[str, Any]:
    # 规则先给个基础画像
    base = {
        'summary': text[:600],
        'skills': list(sorted(set(re.findall(r"[A-Za-z#\+\.]{2,}[A-Za-z0-9#\+\.-]*", text)) ))[:40],
        'education': '',
        'experience_points': []
    }
    sys_prompt = (
        "You are a precise resume analyst. Extract key skills, education, achievements, and quantifiable results. "
        "Return concise bullet points. Keep it ATS-friendly."
    )
    user_prompt = (
        f"LANG={lang}. Given the raw resume text below, produce a concise ATS-friendly profile with sections: "
        f"1) Professional Summary (<=120 words). 2) Key Skills (<=15 items). 3) Experience Highlights (<=8 bullets). "
        f"4) Education (degree/school).\n\nResume Raw Text:\n{text[:15000]}"
    )
    llm = call_deepseek_chat(sys_prompt, user_prompt)
    if llm:
        return {
            'summary': llm,
            'skills': base['skills'],
            'education': '',
            'experience_points': []
        }
    return base


def match_resume_to_job(resume_profile: str, job_desc: str, job_title: str = '') -> Dict[str, Any]:
    # 规则分：技能词/关键词重叠
    def tokenize(t):
        t = (t or '').lower()
        t = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]+", " ", t)
        return set(w for w in t.split() if len(w) > 2)

    res_toks = tokenize(resume_profile)
    jd_toks = tokenize(job_desc + ' ' + job_title)
    overlap = len(res_toks & jd_toks)
    base_score = min(60, overlap)  # 0-60 分

    # LLM 语义补分
    sys_prompt = "You are a job-match evaluator. Score 0-100 with reasoning."
    user_prompt = (
        f"Job Title: {job_title}\nJD:\n{job_desc[:6000]}\n\nCandidate Profile:\n{resume_profile[:6000]}\n\n"
        f"Return JSON with keys: score(0-100), top_matches(<=5 bullets), gaps(<=3 bullets), advice(<=3 bullets)."
    )
    llm = call_deepseek_chat(sys_prompt, user_prompt)
    llm_score = 0
    analysis = {}
    if llm:
        try:
            # 尝试找一个 JSON 块
            json_match = re.search(r"\{[\s\S]*\}", llm)
            if json_match:
                analysis = json.loads(json_match.group(0))
                llm_score = int(analysis.get('score', 0))
            else:
                analysis = {"advice": [llm[:400]]}
        except Exception:
            analysis = {"advice": [llm[:400]]}

    final_score = int(0.6 * base_score + 0.4 * (llm_score or base_score))
    return {
        'score': max(1, min(100, final_score)),
        'rule_overlap': overlap,
        'llm_raw': llm[:1200] if llm else '',
        'analysis': analysis or {}
    }


def generate_ats_resume(resume_profile: str, job_desc: str, lang='zh', template='1') -> str:
    sys_prompt = (
        "You produce ATS-optimized resumes tailored to a specific JD. Use simple formatting, clear bullet points, "
        "and keyword-rich phrasing."
    )
    user_prompt = (
        f"LANG={lang}. Create a one-page ATS resume tailored to this JD. Keep it concise and achievement-oriented.\n\n"
        f"JD:\n{job_desc[:6000]}\n\nCandidate Profile:\n{resume_profile[:6000]}"
    )
    llm = call_deepseek_chat(sys_prompt, user_prompt, temperature=0.3, max_tokens=1500)
    if llm:
        return llm
    # 兜底模板
    return (
        ("【ATS 简历（模板{t}）】\n\n职业概述\n- …\n\n核心技能\n- …\n\n工作经历\n- …\n\n教育背景\n- …\n").replace('{t}', template)
        if lang == 'zh' else
        ("ATS Resume (Template {t})\n\nSummary\n- …\n\nKey Skills\n- …\n\nExperience\n- …\n\nEducation\n- …\n").replace('{t}', template)
    )


# =============== Flask 路由 ===============

@app.route('/')
def root_index():
    return redirect('/public/agent.html', code=302)


@app.route('/public/<path:path>')
def serve_public(path):
    return send_from_directory(PUBLIC_DIR, path)


# ---- Jobs ----
@app.route('/api/jobs/refresh', methods=['POST'])
def api_jobs_refresh():
    info = refresh_jobs_from_seed()
    return jsonify({"ok": True, **info})


@app.route('/api/jobs/list', methods=['GET'])
def api_jobs_list():
    rows = read_jobs_from_csv()
    q = (request.args.get('q') or '').lower()
    location = (request.args.get('location') or '').lower()
    remote = request.args.get('remote')  # 'true'/'false'/None
    level = (request.args.get('level') or '').lower()
    zero_exp = request.args.get('zero_exp')  # 'true'/'false'/None
    limit = int(request.args.get('limit') or 300)

    def match(row):
        text = ' '.join([
            row.get('title', ''), row.get('company', ''), row.get('location', ''), row.get('description', '')
        ]).lower()
        if q and q not in text:
            return False
        if location and location not in (row.get('location', '') or '').lower():
            return False
        if level and level not in (row.get('level', '') or '').lower():
            return False
        if remote in ('true', 'false'):
            flag = str(row.get('remote', '')).lower() in ('true', '1', 'yes')
            if (remote == 'true' and not flag) or (remote == 'false' and flag):
                return False
        if zero_exp in ('true', 'false'):
            has = 'zero_exp_friendly' in (row.get('tags') or [])
            if (zero_exp == 'true' and not has) or (zero_exp == 'false' and has):
                return False
        return True

    rows = [r for r in rows if match(r)]
    return jsonify({"ok": True, "count": len(rows), "items": rows[:limit]})


# ---- Resume Parse ----
@app.route('/api/resume/parse', methods=['POST'])
def api_resume_parse():
    lang = (request.form.get('lang') or 'zh').lower()
    file = request.files.get('file')
    if not file:
        return jsonify({"ok": False, "error": "no_file"}), 400
    raw_text = extract_text_from_file(file)
    profile = summarize_resume(raw_text, lang=lang)
    return jsonify({"ok": True, "raw_len": len(raw_text), "profile": profile})


# ---- Match ----
@app.route('/api/match', methods=['POST'])
def api_match():
    data = request.get_json(force=True)
    resume_profile = data.get('resume_profile') or ''
    job_desc = data.get('job_desc') or ''
    job_title = data.get('job_title') or ''
    res = match_resume_to_job(resume_profile, job_desc, job_title)
    return jsonify({"ok": True, "result": res})


# ---- Generate ATS resume ----
@app.route('/api/resume/generate', methods=['POST'])
def api_resume_generate():
    data = request.get_json(force=True)
    resume_profile = data.get('resume_profile') or ''
    job_desc = data.get('job_desc') or ''
    lang = (data.get('lang') or 'zh').lower()
    template = data.get('template') or '1'
    text = generate_ats_resume(resume_profile, job_desc, lang=lang, template=template)
    return jsonify({"ok": True, "text": text})


# ---- Apply Log (手动) ----
@app.route('/api/apply/log', methods=['POST'])
def api_apply_log():
    data = request.get_json(force=True)
    row = {
        'id': str(uuid.uuid4()),
        'user': data.get('user') or 'demo',
        'job_url': data.get('job_url') or '',
        'title': data.get('title') or '',
        'company': data.get('company') or '',
        'ts': datetime.utcnow().isoformat() + 'Z',
        'note': data.get('note') or 'manual'
    }
    need_header = not os.path.exists(APPLICATIONS_CSV)
    with open(APPLICATIONS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if need_header:
            writer.writeheader()
        writer.writerow(row)
    return jsonify({"ok": True, "item": row})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
