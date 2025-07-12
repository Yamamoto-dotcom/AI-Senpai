
import os
import json
import time
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ CONFIG ------------------ #
EMBED_MODEL = "models/embedding-001"      # Gemini embeddingモデル (Flashと互換)
CHAT_MODEL  = "models/gemini-1.5-flash-latest"
FAISS_THRESHOLD = 0.80                    # FAQマッチ判定用

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG_FILE = LOG_DIR / "chat_logs.jsonl"
UNANSWERED_FILE = LOG_DIR / "unanswered.jsonl"

st.set_page_config(page_title="AI先輩 FAQ Bot", page_icon="🤖", layout="centered")
st.title("🎓 AI先輩 – FAQチャットボット")

# ------------------  UTIL ------------------ #
def append_jsonl(path: Path, data: dict) -> None:
    """JSON Lines 形式で1行追記"""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# ------------------  LOAD ENV ------------------ #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("`.env` に GOOGLE_API_KEY が設定されていません。")
    st.stop()
genai.configure(api_key=GOOGLE_API_KEY)

# ------------------  LOAD FAQ & BUILD INDEX ------------------ #
@st.cache_resource(show_spinner="FAQ を読み込み中 ...")  # 再起動までキャッシュ
def load_faq_index(csv_path: str = "faq.csv"):
    if not Path(csv_path).exists():
        st.error(f"{csv_path} が見つかりません。app.py と同じフォルダに配置してください。")
        st.stop()

    df = pd.read_csv(csv_path)
    if not {"question", "answer"}.issubset(df.columns):
        st.error("`faq.csv` には 'question' と 'answer' の列が必要です。")
        st.stop()

    # Embedding
    embeddings = []
    for q in df["question"].tolist():
        try:
            emb = genai.embed_content(model=EMBED_MODEL, content=q, task_type="retrieval_query")["embedding"]
            embeddings.append(emb)
        except Exception as e:
            st.error(f"Embedding 失敗: {e}")
            st.stop()

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return df, index, embeddings

faq_df, faq_index, faq_embeddings = load_faq_index()

# ------------------  SIDEBAR ------------------ #
with st.sidebar:
    st.subheader("📄 FAQ データ")
    st.write(f"件数: {len(faq_df)}")
    if st.button("FAQ 先頭5件を見る"):
        st.dataframe(faq_df.head())

    st.markdown("---")
    st.markdown("**閾値 (類似度)**")
    FAISS_THRESHOLD = st.slider("FAQ マッチ閾値", 0.0, 1.0, FAISS_THRESHOLD, 0.01)

# ------------------  CHAT LOOP ------------------ #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render previous messages
for role, msg in st.session_state.chat_history:
    box = st.chat_message(role)
    box.markdown(msg)

# User input
if prompt := st.chat_input("質問をどうぞ"):
    st.session_state.chat_history.append(("user", prompt))
    user_box = st.chat_message("user")
    user_box.markdown(prompt)

    # Embedding & search
    try:
        user_emb = genai.embed_content(model=EMBED_MODEL, content=prompt, task_type="retrieval_query")["embedding"]
    except Exception as e:
        st.error(f"Embedding 失敗: {e}")
        st.stop()

    D, I = faq_index.search(np.array([user_emb]).astype("float32"), k=1)
    similarity = 1 - D[0][0] / 2  # L2->cos 適当変換 (0~1ぐらいの目安)

    answered_by_faq = similarity >= FAISS_THRESHOLD
    response = ""
    source_question = ""
    if answered_by_faq:
        source_question = faq_df.iloc[I[0][0]]["question"]
        response = faq_df.iloc[I[0][0]]["answer"]
    else:
        # 検索結果をコンテキストとして Gemini に投げる
        context = ""
        if similarity > 0.2:  # 一応近いものがあれば
            context = (
                f"参考になりそうな FAQ:\n"
                f"Q: {faq_df.iloc[I[0][0]]['question']}\n"
                f"A: {faq_df.iloc[I[0][0]]['answer']}\n---"
            )

        system_prompt = (
            'あなたは大学の先輩チャットボット "AI先輩" です。'
            'ユーザーの質問に日本語で端的かつ丁寧に答えてください。'
        )
        full_prompt = f"{context}\nユーザーの質問: {prompt}"

        try:
            gen_response = genai.generate_content(
                model=CHAT_MODEL,
                contents=[
                    {"role": "system", "parts": [{"text": system_prompt}]},
                    {"role": "user",   "parts": [{"text": full_prompt}]},
                ],
                safety_settings={
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
            )
            response = gen_response.candidates[0].content.parts[0].text
        except Exception as e:
            response = (
                "申し訳ありません、回答生成中にエラーが発生しました。"
                "後でもう一度お試しください。"
            )
            logging.exception(e)

    # Display assistant response
    assistant_box = st.chat_message("assistant")
    assistant_box.markdown(response)
    st.session_state.chat_history.append(("assistant", response))

    # Caption for debug
    assistant_box.caption(f"FAQ 類似度: {similarity:.2f} / マッチ: {answered_by_faq}")

    # --- JSON に安全に変換できるように保険 ------------------
if not isinstance(response, str):
    response = str(response)
if not isinstance(similarity, float):
    similarity = float(similarity)
# -----------------------------------------------------------
    # Logging
    log_entry = {
        "ts": time.time(),
        "question": prompt,
        "answered_by_faq": answered_by_faq,
        "similarity": similarity,
        "faq_question": source_question,
        "answer": response,
    }
    append_jsonl(CHAT_LOG_FILE, log_entry)

    if not answered_by_faq:
        append_jsonl(UNANSWERED_FILE, log_entry)

        st.caption(f"（類似度: {top_similarity:.2f}, 未回答: {is_unanswered}）")
