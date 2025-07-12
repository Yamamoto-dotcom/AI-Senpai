# ─────────────────  app.py  ─────────────────
import os, json, time, logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

# ----------------- CONFIG ------------------ #
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL  = "models/gemini-1.5-flash-latest"
COS_THRESHOLD = 0.30          # コサイン類似度の閾値
TOP_K = 3                     # FAQ 上位何件を見るか

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG_FILE      = LOG_DIR / "chat_logs.jsonl"
UNANSWERED_LOGFILE = LOG_DIR / "unanswered.jsonl"

st.set_page_config("AI先輩 FAQ Bot", "🤖")
st.title("🎓 AI先輩 – FAQチャットボット")

# -------------- UTIL（安全ログ） ------------- #
def append_jsonl(path: Path, data: dict) -> None:
    def to_safe(x):
        return x if isinstance(x, (str, int, float, bool, type(None))) else str(x)
    safe = {k: to_safe(v) for k, v in data.items()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")

# -------------- 環境変数 --------------------- #
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("`.env` か Secrets に GOOGLE_API_KEY を設定してください"); st.stop()
genai.configure(api_key=API_KEY)

# -------------- FAQ 読み込み＆インデックス --- #
@st.cache_resource(show_spinner="FAQ を読み込み中 …")
def load_faq(csv_path="faq.csv"):
    if not Path(csv_path).exists():
        st.error(f"{csv_path} がありません"); st.stop()

    df = pd.read_csv(csv_path)
    if not {"question", "answer"}.issubset(df.columns):
        st.error("faq.csv に 'question','answer' 列が必要"); st.stop()

    # Embedding & Cosine 用前処理
    vecs = [genai.embed_content(model=EMBED_MODEL,
                                content=q,
                                task_type="retrieval_query")["embedding"]
            for q in df["question"]]

    vecs = np.array(vecs).astype("float32")
    faiss.normalize_L2(vecs)              # ★ 正規化
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return df, index

faq_df, faq_index = load_faq()

# -------------- SIDEBAR --------------------- #
with st.sidebar:
    st.write(f"📄 FAQ 件数 : {len(faq_df)}")
    if st.button("最初の5件を見る"):
        st.dataframe(faq_df.head())
    COS_THRESHOLD = st.slider("コサイン閾値", 0.0, 1.0, COS_THRESHOLD, 0.01)

# -------------- チャット履歴 ----------------- #
if "history" not in st.session_state: st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)

# -------------- ユーザ入力 ------------------- #
if user_q := st.chat_input("質問をどうぞ"):
    st.session_state.history.append(("user", user_q))
    st.chat_message("user").markdown(user_q)

    # --- Embedding & 検索（コサイン） -------- #
    uvec = genai.embed_content(model=EMBED_MODEL,
                               content=user_q,
                               task_type="retrieval_query")["embedding"]
    uvec = np.asarray(uvec, dtype="float32"); faiss.normalize_L2(uvec)
    D, I = faq_index.search(uvec[None], TOP_K)   # D はコサイン類似度
    best_sim, best_idx = float(D[0][0]), int(I[0][0])

    use_faq = best_sim >= COS_THRESHOLD
    answer, src_q = "", ""

    if use_faq:
        row = faq_df.iloc[best_idx]
        src_q, answer = row["question"], row["answer"]
    else:
        # ---- Gemini 生成 -------------------- #
        context = ""
        if best_sim > 0.1:
            row = faq_df.iloc[best_idx]
            context = f"参考FAQ:\nQ: {row['question']}\nA: {row['answer']}\n---\n"
        system_prompt = (
            'あなたは大学の先輩チャットボット "AI先輩" です。'
            '質問に日本語で端的かつ丁寧に答えてください。'
        )
        full_prompt = f"{context}ユーザーの質問: {user_q}"
        try:
            resp = genai.generate_content(
                model=CHAT_MODEL,
                contents=[
                    {"role": "system", "parts": [{"text": system_prompt}]},
                    {"role": "user",   "parts": [{"text": full_prompt}]},
                ],
            )
            answer = resp.candidates[0].content.parts[0].text
        except Exception as e:
            answer = "回答生成でエラーが発生しました。時間を置いて再度お試しください。"
            logging.exception(e)

    # ---- 画面表示 --------------------------- #
    st.chat_message("assistant").markdown(answer)
    st.session_state.history.append(("assistant", answer))
    st.caption(f"コサイン類似度: {best_sim:.2f} / FAQマッチ: {use_faq}")

    # ---- ログ書き込み ----------------------- #
    log = {
        "ts"       : time.time(),
        "question" : user_q,
        "answered_by_faq": use_faq,
        "similarity": best_sim,
        "faq_question": src_q,
        "answer"    : answer,
    }
    append_jsonl(CHAT_LOG_FILE, log)
    if not use_faq: append_jsonl(UNANSWERED_LOGFILE, log)
# ─────────────────────────────────────────────
        st.caption(f"（類似度: {top_similarity:.2f}, 未回答: {is_unanswered}）")
