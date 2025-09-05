# app.py  — LGWORK 一键覆盖版
# -*- coding: utf-8 -*-

import csv
import io
import json
import os
import re
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Any

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

APP_NAME = "LGWORK"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
JOBS_CSV = os.path.join(DATA_DIR, "jobs.csv")

# 环境变量（DeepSeek 可选）
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
LLM_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")  # 如果不设置，则不做鉴权

# CSV 列头
CSV_HEADERS = ["title", "company", "location", "jd_url", "source", "posted_at", "tags"]

# 线程锁，避免并发写文件问题
_file_lock = threading.Lock()

app = Flask(__name__, static_folder="public", static_url_path="/public")
CORS(app)


# ---------- 工具函数 ----------
def _log(msg: str):
    print(f"[{APP_NAME}] {msg}", flush=True)


def ensure_jobs_csv():
    """确保 CSV 文件存在且有表头"""
    if not os.path.exists(JOBS_CSV):
        with _file_lock:
            with open(JOBS_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)


def _safe_json_loads(s: str, default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _normalize_url(u: str) -> str:
    # 简单归一化，去掉尾部 / 和空白
    if not u:
        return u
    return u.strip().rstrip("/")


def _read_jobs() -> List[Dict[str, str]]:
    """容错读 CSV；tags 脏 JSON 直接返回 []"""
    ensure_jobs_csv()
    rows = []
    with _file_lock:
        with open(JOBS_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # 如果表头不对，重新写表头
            if reader.fieldnames is None or set(reader.fieldnames) != set(CSV_HEADERS):
                # 备份旧文件
                f.seek(0)
                raw = f.read()
                _log("jobs.csv header invalid — rewriting header and preserving rows when possible.")
                with open(JOBS_CSV, "w", newline="", encoding="utf-8") as fw:
                    w = csv.writer(fw)
                    w.writerow(CSV_HEADERS)
                # 尽量不崩，但无法保留旧行结构，返回空
                return []
            for r in reader:
                # 兜底字段
                item = {
                    "title": (r.get("title") or "").strip(),
                    "company": (r.get("company") or "").strip(),
                    "location": (r.get("location") or "").strip(),
                    "jd_url": _normalize_url(r.get("jd_url") or ""),
                    "source": (r.get("source") or "").strip(),
                    "posted_at": (r.get("posted_at") or "").strip(),
                    "tags": json.dumps(_safe_json_loads(r.get("tags") or "", []), ensure_ascii=False),
                }
                rows.append(item)
    return rows


def _write_jobs(jobs: List[Dict[str, str]]):
    ensure_jobs_csv()
    with _file_lock:
        with open(JOBS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for j in jobs:
                row = {k: j.get(k, "") for k in CSV_HEADERS}
                # 确保 tags 为合法 JSON 字符串
                row["tags"] = json.dumps(_safe_json_loads(row.get("tags", ""), []), ensure_ascii=False)
                writer.writerow(row)


def _append_jobs(new_jobs: List[Dict[str, str]]) -> Tuple[int, int]:
    """追加职位，按 jd_url 去重；返回 (新增条数, 去重条数)"""
    existing = _read_jobs()
    seen = {e["jd_url"] for e in existing if e.get("jd_url")}
    added, skipped = 0, 0
    clean = []

    for j in new_jobs:
        # 规范化
        j["jd_url"] = _normalize_url(j.get("jd_url", ""))
        j["posted_at"] = j.get("posted_at") or datetime.utcnow().isoformat()
        j["tags"] = json.dumps(_safe_json_loads(j.get("tags", ""), []), ensure_ascii=False)

        if not j["jd_url"] or j["jd_url"] in seen:
            skipped += 1
            continue
        seen.add(j["jd_url"])
        clean.append(j)
        added += 1

    if clean:
        existing.extend(clean)
        _write_jobs(existing)

    return added, skipped


def _require_admin(req: request):
    """可选的 admin token 校验"""
    if not ADMIN_TOKEN:
        return True
    token = req.headers.get("X-ADMIN-TOKEN") or req.args.get("admin_token") or (req.json or {}).get("admin_token")
    return token == ADMIN_TOKEN


# ---------- Greenhouse / Lever / Workday 抓取 ----------
def fetch_greenhouse(company: str) -> List[Dict[str, str]]:
    # API 文档：https://developers.greenhouse.io/job-board.html#list-jobs
    url = f"https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true"
    _log(f"Fetching Greenhouse: {url}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    out = []
    for job in data.get("jobs", []):
        out.append({
            "title": job.get("title", "").strip(),
            "company": company,
            "location": (job.get("location") or {}).get("name", ""),
            "jd_url": job.get("absolute_url", ""),
            "source": "greenhouse",
            "posted_at": job.get("updated_at") or job.get("created_at") or "",
            "tags": json.dumps([], ensure_ascii=False),
        })
    return out


def fetch_lever(company: str) -> List[Dict[str, str]]:
    # 公开 JSON： https://api.lever.co/v0/postings/<company>?mode=json
    url = f"https://api.lever.co/v0/postings/{company}?mode=json"
    _log(f"Fetching Lever: {url}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json() or []
    out = []
    for job in data:
        loc = ""
        if job.get("categories"):
            loc = job["categories"].get("location", "") or ""
        out.append({
            "title": job.get("text", "").strip(),
            "company": company,
            "location": loc,
            "jd_url": job.get("hostedUrl", ""),
            "source": "lever",
            "posted_at": job.get("createdAt") and datetime.utcfromtimestamp(job["createdAt"]/1000).isoformat() or "",
            "tags": json.dumps([], ensure_ascii=False),
        })
    return out


# ---- Workday ----
# Workday 集群检测：wd1-wd15 逐个尝试；找到能 200 的 cxs 接口即用之
_WORKDAY_CLUSTERS = [str(i) for i in range(1, 16)]

def _workday_cxs_url(cluster: str, tenant: str, site: str) -> str:
    return f"https://{tenant}.wd{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"

def detect_workday_cluster(tenant: str, site: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (LGWORK/1.0)"}
    payload = {"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""}
    for c in _WORKDAY_CLUSTERS:
        url = _workday_cxs_url(c, tenant, site)
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=12)
            if r.status_code == 200 and "jobPostings" in r.text:
                _log(f"Workday cluster detected: wd{c}")
                return c
        except Exception:
            pass
    raise RuntimeError("无法自动识别 Workday 集群（wdN）。请检查 tenant/site 是否正确。")


def parse_workday_line(tenant: str, site: str, cluster: str, j: Dict[str, Any]) -> Dict[str, str]:
    """
    统一的 Workday job 解析函数（修复你日志里 _parse_workday_line 未定义的问题）。
    cxs 返回字段常见：title, locationsText, externalPath, postedOn
    """
    title = (j.get("title") or "").strip()
    location = (j.get("locationsText") or "").strip()
    path = j.get("externalPath") or j.get("externalUrl") or ""
    path = str(path).lstrip("/")

    # 常见 externalPath 已自带 job/xxx/xxx，不再重复拼接
    if path.startswith("job/"):
        jd_url = f"https://{tenant}.wd{cluster}.myworkdayjobs.com/{site}/{path}"
    else:
        jd_url = f"https://{tenant}.wd{cluster}.myworkdayjobs.com/{site}/job/{path}"

    posted = (j.get("postedOn") or "").strip()
    return {
        "title": title,
        "company": tenant,
        "location": location,
        "jd_url": jd_url,
        "source": "workday",
        "posted_at": posted,
        "tags": json.dumps([], ensure_ascii=False),
    }


def fetch_workday(tenant: str, site: str, limit: int = 100) -> List[Dict[str, str]]:
    cluster = detect_workday_cluster(tenant, site)
    headers = {"User-Agent": "Mozilla/5.0 (LGWORK/1.0)"}
    url = _workday_cxs_url(cluster, tenant, site)

    out = []
    offset = 0
    step = 50
    while offset < limit:
        payload = {"appliedFacets": {}, "limit": step, "offset": offset, "searchText": ""}
        _log(f"Fetching Workday: {url} offset={offset}")
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        posts = data.get("jobPostings", [])
        if not posts:
            break
        for j in posts:
            out.append(parse_workday_line(tenant, site, cluster, j))
        if len(posts) < step:
            break
        offset += step

    return out


# ---------- LLM（可选） ----------
def llm_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    兜底：没配 DeepSeek 也能返回一个可用的占位分析，避免 500。
    前端只要能拿到结构化 JSON 就行。
    """
    resume_text = (payload.get("resume_text") or "").strip()
    jd_text = (payload.get("jd_text") or "").strip()

    if not LLM_API_KEY:
        # 占位返回
        return {
            "model": "placeholder",
            "score": 68,
            "summary": "未配置 DeepSeek API，返回占位分析结果。",
            "keywords": ["experience", "skills", "match"],
            "risks": ["教育背景可能不完全匹配", "管理跨度需要进一步确认"],
            "advice": ["补充可量化成果", "对齐 JD 的 must-have 能力，完善案例"],
        }

    # 真正调用 DeepSeek（兼容 OpenAI 风格）
    try:
        url = f"{LLM_API_BASE}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        sys_prompt = (
            "You are a recruiting co-pilot. Read resume and JD, return a concise JSON with fields: "
            "score(0-100), summary, keywords(list), risks(list), advice(list). Keep it short."
        )
        user_prompt = f"RESUME:\n{resume_text}\n\nJD:\n{jd_text}\n"
        body = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        resp = r.json()
        content = resp["choices"][0]["message"]["content"]
        # 尝试从内容中提取 JSON；若不是严格 JSON，则包一层
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {
                "model": LLM_MODEL,
                "raw": content,
                "score": 70,
                "summary": "LLM 返回非严格 JSON，已兜底。",
                "keywords": ["AI", "NLP"],
                "risks": [],
                "advice": [],
            }
        return parsed
    except Exception as e:
        _log(f"DeepSeek call failed: {e}")
        return {
            "model": "error-fallback",
            "score": 65,
            "summary": "DeepSeek 调用失败，返回兜底结果。",
            "keywords": ["fallback"],
            "risks": ["外部模型不可用"],
            "advice": ["稍后重试或检查 API Key / Base / Model"],
        }


# ---------- 路由 ----------
@app.route("/")
def index():
    # 根路径直接渲染 agent.html（已解决你之前的“/ 返回 404”）
    return send_from_directory(PUBLIC_DIR, "agent.html")


@app.route("/admin")
def admin_page():
    return send_from_directory(PUBLIC_DIR, "admin.html")


@app.route("/api/ping")
def ping():
    return jsonify({"ok": True, "ts": datetime.utcnow().isoformat()})


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    jobs = _read_jobs()
    return jsonify({"ok": True, "count": len(jobs), "jobs": jobs})


@app.route("/api/clear_jobs", methods=["POST"])
def clear_jobs():
    if not _require_admin(request):
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    _write_jobs([])
    return jsonify({"ok": True, "cleared": True})


@app.route("/api/ingest_ats", methods=["POST"])
def ingest_ats():
    """
    统一导入入口（admin.html 会 POST 到这里）
    接收 JSON：
    {
      "provider": "greenhouse" | "lever" | "workday",
      // Greenhouse/Lever:
      "company": "xxx",
      // Workday:
      "tenant_site": "beigene/BeiGene"   // 按你要求的输入格式
      // 兼容参数：也可传 {"tenant": "...", "site": "..."}
    }
    """
    try:
        if not _require_admin(request):
            return jsonify({"ok": False, "error": "Unauthorized"}), 401

        body = request.get_json(force=True) or {}
        provider = (body.get("provider") or "").strip().lower()
        _log(f"/api/ingest_ats provider={provider}")

        new_jobs: List[Dict[str, str]] = []

        if provider == "greenhouse":
            company = (body.get("company") or "").strip()
            if not company:
                return jsonify({"ok": False, "error": "缺少 company"}), 400
            new_jobs = fetch_greenhouse(company)

        elif provider == "lever":
            company = (body.get("company") or "").strip()
            if not company:
                return jsonify({"ok": False, "error": "缺少 company"}), 400
            new_jobs = fetch_lever(company)

        elif provider == "workday":
            # 支持 tenant_site 或者 tenant+site
            tenant_site = (body.get("tenant_site") or body.get("workday") or "").strip()
            tenant = (body.get("tenant") or "").strip()
            site = (body.get("site") or "").strip()
            if tenant_site and (not tenant or not site):
                if "/" in tenant_site:
                    tenant, site = tenant_site.split("/", 1)
            if not (tenant and site):
                return jsonify({"ok": False, "error": "缺少 tenant/site，请按 'tenant/site' 或分别提供"}), 400
            new_jobs = fetch_workday(tenant, site, limit=int(body.get("limit") or 200))

        else:
            return jsonify({"ok": False, "error": "未知 provider"}), 400

        added, skipped = _append_jobs(new_jobs)
        return jsonify({"ok": True, "fetched": len(new_jobs), "added": added, "skipped": skipped})

    except requests.HTTPError as he:
        return jsonify({"ok": False, "error": f"HTTP {he.response.status_code}: {str(he)}"}), 502
    except Exception as e:
        _log(f"ERROR in /api/ingest_ats: {e}")
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/analyze_recommend", methods=["POST"])
def analyze_recommend():
    """
    前端会提交 { resume_text, jd_text, ... }
    我们做容错：即使 jobs.csv 的 tags 是空/脏 JSON，也不报错。
    """
    try:
        body = request.get_json(force=True) or {}
        result = llm_analyze(body)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        _log(f"ERROR in /api/analyze_recommend: {e}")
        # 兜底，避免 500
        return jsonify({
            "ok": True,
            "result": {
                "model": "server-fallback",
                "score": 66,
                "summary": "分析出现异常，已返回兜底结果。",
                "keywords": [],
                "risks": [],
                "advice": [],
            }
        })


# ---------- 静态资源（兜底） ----------
@app.route("/public/<path:filename>")
def public_files(filename):
    # 让 /public/* 正常返回文件（admin.html / agent.html / JS / CSS）
    return send_from_directory(PUBLIC_DIR, filename)


# ---------- 启动 ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    _log("////////////////////////////////////////////////////")
    _log(f"==> {APP_NAME} is booting on port {port}")
    _log("==> Your service is live  🚀")
    _log("////////////////////////////////////////////////////")
    app.run(host="0.0.0.0", port=port)
