# ─────────── app.py  (cost 無視・Flash 全投げ版) ───────────
import os, json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ────── CONFIG ──────
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL  = "models/gemini-1.5-flash-latest"
TOP_K       = 10         # Flash に渡す FAQ 件数
LOG_DIR     = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG    = LOG_DIR / "chat_logs.jsonl"

# ────── UTIL ──────
def jl_append(path: Path, row: dict):
    safe = {k:(v if isinstance(v,(str,int,float,bool,type(None))) else str(v)) for k,v in row.items()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")

# ────── API KEY ──────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY を .env か Secrets で設定してください"); st.stop()
genai.configure(api_key=api_key)
MODEL = genai.GenerativeModel(CHAT_MODEL)

# ────── FAQ 読み込み & 埋め込み ──────
@st.cache_resource(show_spinner="FAQ を読み込み中…")
def load_faq(csv="faq.csv"):
    df = pd.read_csv(csv)
    if not {"question","answer"}.issubset(df.columns):
        st.error("faq.csv に question / answer 列が必要"); st.stop()

    doc_vecs = [
        genai.embed_content(
            model=EMBED_MODEL,
            content=row["question"],
            task_type="retrieval_document"
        )["embedding"]
        for _, row in df.iterrows()
    ]
    doc_vecs = np.asarray(doc_vecs, dtype="float32")
    doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return df, doc_vecs
faq_df, faq_vecs = load_faq()

def top_k(q_vec, k=TOP_K):
    q = q_vec / np.linalg.norm(q_vec)
    sims = faq_vecs @ q
    idxs = sims.argsort()[-k:][::-1]
    return sims[idxs], idxs

# ────── Streamlit UI ──────
st.set_page_config("AI先輩 FAQ Bot", "🤖")
st.title("🎓 AI先輩 – FAQチャットボット")
if "hist" not in st.session_state: st.session_state.hist=[]
for r,t in st.session_state.hist: st.chat_message(r).markdown(t)

# ────── メインループ ──────
if q := st.chat_input("質問をどうぞ"):
    st.chat_message("user").markdown(q)
    st.session_state.hist.append(("user", q))

    # 1) 質問を query 埋め込み
    q_vec = genai.embed_content(
        model=EMBED_MODEL,
        content=q,
        task_type="retrieval_query"
    )["embedding"]
    q_vec = np.asarray(q_vec, dtype="float32")

    # 2) FAQ 上位 k 件
    sims, idxs = top_k(q_vec)
    faq_block = "\n\n".join(
        f"{i+1}. (cos={sims[i]:.2f})\nQ:{faq_df.iloc[idx]['question']}\nA:{faq_df.iloc[idx]['answer']}"
        for i,idx in enumerate(idxs)
    )

    # 3) Flash へ丸投げ
    prompt = (
        "あなたは大学の頼れる先輩チャットボットです。\n"
        "以下の候補 FAQ を読み、質問に最も合致する回答があれば「その回答だけ」を返してください。\n"
        "該当が無い場合は **NO_ANSWER** とだけ返してください。\n\n"
        f"【質問】\n{q}\n\n【候補FAQ】\n{faq_block}"
    )
    try:
        resp    = MODEL.generate_content(prompt)
        raw_ans = resp.text.strip()
    except Exception as e:
        raw_ans = "NO_ANSWER"
        st.error(f"Gemini API エラー: {type(e).__name__} – {e}")

    # 4) 出力判定
    if raw_ans == "NO_ANSWER":
        final = "ごめん、今はその情報を持っていないんだ。他の先輩にも聞いてみてね。"
        hit   = False
    else:
        final = f"{raw_ans}\n\nまた何かあったら遠慮なく聞いてね！"
        hit   = True

    # 5) 表示 & ログ
    st.chat_message("assistant").markdown(final)
    st.session_state.hist.append(("assistant", final))
    jl_append(CHAT_LOG, {"ts":time.time(),"q":q,"a":final,"faq_hit":hit})
