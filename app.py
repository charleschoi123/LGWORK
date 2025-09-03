import os
import csv
import json
import uuid
import time
import threading
from datetime import datetime
from urllib.parse import unquote_plus

from flask import Flask, request, jsonify, make_response, send_from_directory

import requests  # 用于调用 DeepSeek

app = Flask(__name__)

# -----------------------------
# 配置（用环境变量即可，不暴露在前端）
# -----------------------------
JOBS_CSV_PATH = os.environ.get("JOBS_CSV_PATH", "data/jobs.csv")
TRACKER_JSON_PATH = os.environ.get("TRACKER_JSON_PATH", "data/tracker.json")

DEEPSEEK_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")  # 也可用 deepseek-reasoner

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
    # 尝试 ISO
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
            # 标准化字段
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
            tags = [t.strip() for sep in [";", ","] for t in raw_tags.split(sep)]
            item["tags"] = sorted(list({t for t in tags if t}))

            # 排序辅助时间戳
            dt = _parse_date(item["posted_at"])
            item["_posted_ts"] = int(dt.timestamp()) if dt else 0

            jobs.append(item)
    return jobs


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


# -----------------------------
# /api/jobs 公开职位列表 + 过滤
# -----------------------------
@app.route("/api/jobs", methods=["GET", "OPTIONS"])
def api_jobs():
    if request.method == "OPTIONS":
        return _json_response({"ok": True})

    jobs = _read_jobs()

    # 过滤参数
    q = (request.args.get("q") or "").strip().lower()
    location = (request.args.get("location") or "").strip().lower()
    type_ = (request.args.get("type") or "").strip().lower()       # full/part/project
    work_mode = (request.args.get("work_mode") or "").strip().lower()  # onsite/hybrid/remote
    tag = (request.args.get("tag") or "").strip().lower()          # 单 tag，逗号分隔也可
    min_salary = _safe_float(request.args.get("min_salary"))
    max_salary = _safe_float(request.args.get("max_salary"))
    currency = (request.args.get("currency") or "").strip().upper()

    sort = (request.args.get("sort") or "posted_desc").strip().lower()
    page = max(1, int(request.args.get("page", "1") or 1))
    page_size = min(100, max(1, int(request.args.get("page_size", "20") or 20)))

    # 搜索与过滤
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

        if location and location not in (item.get("location", "").lower()):
            return False

        if type_ and item.get("type") != type_:
            return False

        if work_mode and item.get("work_mode") != work_mode:
            return False

        if currency and (item.get("currency") or "") != currency:
            return False

        if tag:
            # 支持逗号OR
            want = {t.strip() for t in tag.split(",") if t.strip()}
            have = {t.lower() for t in (item.get("tags") or [])}
            if not (want & have):
                return False

        # 薪资区间过滤（若职位没填薪资，则不剔除）
        if min_salary is not None or max_salary is not None:
            smin = item.get("salary_min")
            smax = item.get("salary_max")
            if smin is not None or smax is not None:
                lo = smin if smin is not None else smax
                hi = smax if smax is not None else smin
                # 区间重叠判断
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

    # 排序
    if sort == "posted_asc":
        filtered.sort(key=lambda x: x.get("_posted_ts", 0))
    else:  # 默认 desc
        filtered.sort(key=lambda x: x.get("_posted_ts", 0), reverse=True)

    # 分页
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    data = filtered[start:end]

    # 输出时移除内部字段
    for it in data:
        if "_posted_ts" in it:
            it.pop("_posted_ts", None)

    return _json_response({
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": data
    })


# -----------------------------
# /api/match 深度匹配（DeepSeek）
# -----------------------------
def _extract_json(text):
    """从 LLM 返回中鲁棒提取 JSON（去掉```包裹等）"""
    if not text:
        return None
    text = text.strip()
    # 去掉代码块围栏
    if text.startswith("```"):
        # 可能是 ```json \n ... \n ```
        text = text.strip("`")
        # 去掉可能的json标记
        text = text.replace("json\n", "", 1).replace("json\r\n", "", 1)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # 尝试定位第一个 { 与最后一个 }
        try:
            l = text.find("{")
            r = text.rfind("}")
            if l != -1 and r != -1 and r > l:
                return json.loads(text[l:r+1])
        except Exception:
            return None
    return None


def _call_deepseek_match(resume_text: str, jd_text: str):
    if not DEEPSEEK_API_KEY:
        return {
            "error": "Missing DEEPSEEK_API_KEY. Please set it in Render/Env first."
        }

    url = f"{DEEPSEEK_API_BASE.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are an expert job-matching assistant. Carefully read the JD (must-have & nice-to-have) "
        "and the candidate resume, then return STRICT JSON with fields: "
        "{match_score:int(0-100), must_have_hits:[], must_have_misses:[], "
        "nice_to_have_hits:[], nice_to_have_misses:[], gap_advice:[], "
        "resume_bullets:[]}. Keep answers concise and actionable."
    )

    user_msg = (
        "JD:\n"
        f"{jd_text}\n"
        "---------------------\n"
        "RESUME:\n"
        f"{resume_text}\n\n"
        "Return JSON only. No commentary."
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        # 有些 DeepSeek 兼容 JSON 输出；若不支持也不影响（我们做二次解析）
        # "response_format": {"type": "json_object"},
    }

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        # OpenAI 兼容结构
        content = ""
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "") or ""
        parsed = _extract_json(content)
        if parsed is None:
            return {"error": "LLM returned non-JSON or empty result.", "raw": content}
        # 兜底：确保字段齐全
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
# 轻量投递/跟进：/api/track
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

    # POST 新增或更新
    body = request.get_json(silent=True) or {}
    # 需要字段：job_id, status
    job_id = (body.get("job_id") or "").strip()
    status_ = (body.get("status") or "").strip()  # 准备中/已投/面试中/Offer/关闭
    notes = (body.get("notes") or "").strip()

    if not job_id or not status_:
        return _json_response({"error": "job_id and status are required."}, 400)

    data = _load_tracker()
    now_iso = datetime.utcnow().isoformat()

    # 若 body 带有 id 则更新，否则新增
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
# 健康检查/版本
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return _json_response({"status": "ok", "time": int(time.time())})


@app.route("/version", methods=["GET"])
def version():
    return _json_response({
        "name": "LGWORK API",
        "version": "0.1.0",
        "jobs_csv": JOBS_CSV_PATH,
        "model": DEEPSEEK_MODEL
    })


# -----------------------------
# 本地启动
# -----------------------------
@app.route("/public/<path:filename>", methods=["GET"])
def public_files(filename):
    return send_from_directory("public", filename)

@app.route("/", methods=["GET"])
def home():
    return send_from_directory("public", "index.html")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(JOBS_CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TRACKER_JSON_PATH), exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
