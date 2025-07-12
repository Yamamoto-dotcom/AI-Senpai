# ─────────── app.py  (完全差し替え用) ───────────
import os, json, time, logging, importlib
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# --- FAISS が入らなければ自動フォールバック -----------------
USE_FAISS = importlib.util.find_spec("faiss") is not None
if USE_FAISS:
    import faiss

# ---------------- CONFIG -------------------------------------
EMBED_MODEL   = "models/embedding-001"
CHAT_MODEL    = "models/gemini-1.5-flash-latest"
COS_THRESHOLD = 0.30     # FAQ を採用するコサイン閾値
TOP_K         = 3        # FAQ 上位何件を調べるか

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG = LOG_DIR / "chat_logs.jsonl"
UNANS_LOG = LOG_DIR / "unanswered.jsonl"

st.set_page_config("AI先輩 FAQ Bot", "🤖")
st.title("🎓 AI先輩 – FAQチャットボット")

# ---------------- UTIL ---------------------------------------
def safe_jsonl(path: Path, data: dict):
    def safer(x):                    # 数値・真偽値・None 以外は str()
        return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: safer(v) for k, v in data.items()},
                           ensure_ascii=False) + "\n")

# ---------------- ENV ----------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Secrets または .env に GOOGLE_API_KEY がありません"); st.stop()
genai.configure(api_key=API_KEY)

# ---------------- FAQ LOAD & INDEX ---------------------------
@st.cache_resource(show_spinner="FAQ を読み込み中…")
def load_faq(csv="faq.csv"):
    if not Path(csv).exists():
        st.error(f"{csv} が見つかりません"); st.stop()

    df = pd.read_csv(csv)
    if not {"question", "answer"}.issubset(df.columns):
        st.error("faq.csv に 'question','answer' 列が必要"); st.stop()

    vecs = [genai.embed_content(model=EMBED_MODEL,
                                content=q,
                                task_type="retrieval_query")["embedding"]
            for q in df["question"]]
    vecs = np.asarray(vecs, dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # 正規化

    if USE_FAISS:
        index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)
    else:      # NumPy fallback
        index = vecs

    return df, index

faq_df, faq_index = load_faq()

# ---------------- SIDEBAR ------------------------------------
with st.sidebar:
    st.write(f"📄 FAQ 件数: {len(faq_df)}")
    if st.button("FAQ 先頭 5 件を見る"):
        st.dataframe(faq_df.head())
    COS_THRESHOLD = st.slider("FAQ マッチ閾値 (cos)", 0.0, 1.0,
                              COS_THRESHOLD, 0.01)

# ---------------- CHAT HISTORY -------------------------------
if "hist" not in st.session_state: st.session_state.hist = []
for role, txt in st.session_state.hist:
    st.chat_message(role).markdown(txt)

# ---------------- MAIN LOOP ----------------------------------
def cosine_search(qvec):
    if USE_FAISS:
        q = qvec / np.linalg.norm(qvec)
        D, I = faq_index.search(q[None], TOP_K)
        return D[0], I[0]
    else:      # NumPy fallback
        q = qvec / np.linalg.norm(qvec)
        sims = faq_index @ q
        idx = sims.argsort()[-TOP_K:][::-1]
        return sims[idx], idx

if user_q := st.chat_input("質問をどうぞ"):
    st.session_state.hist.append(("user", user_q))
    st.chat_message("user").markdown(user_q)

    # --- FAQ 検索 -------------------------------------------
    qvec = genai.embed_content(model=EMBED_MODEL,
                               content=user_q,
                               task_type="retrieval_query")["embedding"]
    qvec = np.asarray(qvec, dtype="float32")
    sims, idxs = cosine_search(qvec)
    best_sim, best_idx = float(sims[0]), int(idxs[0])

    use_faq = best_sim >= COS_THRESHOLD
    src_q, answer = "", ""

    if use_faq:
        row = faq_df.iloc[best_idx]
        src_q, answer = row["question"], row["answer"]
    else:
        ctx = ""
        if best_sim > 0.1:
            row = faq_df.iloc[best_idx]
            ctx = f"参考FAQ:\nQ: {row['question']}\nA: {row['answer']}\n---\n"

        sys_prompt = ('あなたは大学の先輩チャットボット "AI先輩" です。'
                      '質問に日本語で端的かつ丁寧に答えてください。')
        full_prompt = f"{ctx}ユーザーの質問: {user_q}"

        try:
            rsp = genai.generate_content(
                model=CHAT_MODEL,
                contents=[
                    {"role": "system", "parts": [{"text": sys_prompt}]},
                    {"role": "user",   "parts": [{"text": full_prompt}]},
                ],
            )
            answer = rsp.candidates[0].content.parts[0].text
        except Exception as e:
            answer = "回答生成に失敗しました。時間をおいて再度お試しください。"
            logging.exception(e)

    # --- 表示 & 履歴 ----------------------------------------
    st.chat_message("assistant").markdown(answer)
    st.session_state.hist.append(("assistant", answer))
    st.caption(f"コサイン類似度: {best_sim:.2f} / FAQマッチ: {use_faq}")

    # --- ログ ----------------------------------------------
    entry = {
        "ts": time.time(),
        "question": user_q,
        "answer"  : answer,
        "faq_hit" : use_faq,
        "similarity": best_sim,
        "faq_question": src_q,
    }
    safe_jsonl(CHAT_LOG, entry)
    if not use_faq: safe_jsonl(UNANS_LOG, entry)
# ─────────────────────────────────────────────
