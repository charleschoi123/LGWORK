import os, re
# --- 文本清洗 ---
def clean_text(t: str) -> str:
if not t:
return ""
t = re.sub(r"\s+", " ", t)
return t.strip()


# --- TF-IDF 匹配 ---
def rank_by_similarity(resume: str, jds: List[JD]) -> List[JD]:
if not resume or not jds:
return jds
docs = [resume] + [jd.desc for jd in jds]
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
X = vectorizer.fit_transform(docs)
sims = cosine_similarity(X[0], X[1:]).flatten()
for jd, s in zip(jds, sims):
jd.score = round(float(s) * 100, 2)
jds.sort(key=lambda x: x.score, reverse=True)
return jds


# --- 解析前端的 JD 文本：一行一个：标题 | 链接 | 描述 ---
def parse_jds(text: str) -> List[JD]:
items: List[JD] = []
for line in text.splitlines():
if not line.strip():
continue
parts = [p.strip() for p in line.split('|')]
if len(parts) >= 3:
items.append(JD(title=parts[0], url=parts[1], desc='|'.join(parts[2:])))
return items[:MAX_JD_PER_RUN]


@app.route('/', methods=['GET', 'POST'])
def index():
if request.method == 'POST':
resume_text = clean_text(request.form.get('resume_text', ''))
jd_text = request.form.get('jd_text', '')
jds = parse_jds(jd_text)
ranked = rank_by_similarity(resume_text, jds)


# 取 Top-1 生成详细建议，控制成本
match_analysis = ""
if resume_text and ranked:
top_jd = ranked[0]
try:
msg = MATCH_PROMPT.format(resume=resume_text[:12000], jd=(top_jd.title + "\n" + top_jd.desc)[:8000])
match_analysis = chat([
{"role": "user", "content": msg}
], system_prompt=BASE_SYSTEM_PROMPT)
except LLMError as e:
match_analysis = f"[模型调用失败] {e}"


# 简单规则生成“简历优化提示”（可与 LLM 互补）
tips = []
all_desc = ' '.join([jd.desc.lower() for jd in jds])
if 'sql' in all_desc and 'sql' not in resume_text.lower():
tips.append('JD 提到 SQL，建议在简历中补充项目/查询优化/指标结果。')
if 'python' in all_desc and 'python' not in resume_text.lower():
tips.append('JD 提到 Python，建议列出具体库（pandas/sklearn）与产出。')
if any(k in all_desc for k in ['机器学习','ml','深度学习','dl']):
if not any(k in resume_text.lower() for k in ['pytorch','tensorflow','sklearn','xgboost']):
tips.append('JD 涉及 ML/DL，建议补充模型与评估指标（AUC、F1、召回等）。')


return render_template('results.html', jds=ranked, analysis=match_analysis, tips=tips)


return render_template('index.html')


if __name__ == '__main__':
    port = int(os.getenv("PORT", "10000"))
    app.run(host='0.0.0.0', port=port, debug=False)
