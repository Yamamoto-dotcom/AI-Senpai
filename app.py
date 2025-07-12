import os, json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- CONFIG ----------
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL  = "models/gemini-1.5-flash-latest"

TOP_K       = 5      # FAQ を渡す上位件数
SIM_FLOOR   = 0.15   # これ未満は FAQ を無理に渡さない
LOG_DIR     = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG    = LOG_DIR / "chat_logs.jsonl"

# ---------- UTILS -----------
def save_jsonl(path: Path, data: dict):
    safe = {k: (v if isinstance(v,(str,int,float,bool,type(None))) else str(v))
            for k,v in data.items()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")

# ---------- API KEY ----------
load_dotenv(); key = os.getenv("GOOGLE_API_KEY")
if not key: st.error("GOOGLE_API_KEY が未設定です"); st.stop()
genai.configure(api_key=key)

# ---------- FAQ LOAD ---------
@st.cache_resource(show_spinner="FAQ を読み込み中…")
def load_faq(csv="faq.csv"):
    df = pd.read_csv(csv)
    vecs = [
        genai.embed_content(model=EMBED_MODEL,
                            content=q,
                            task_type="retrieval_query")["embedding"]
        for q in df["question"]
    ]
    vecs = np.asarray(vecs, dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return df, vecs
faq_df, faq_vecs = load_faq()

def top_k(q_vec, k=TOP_K):
    q = q_vec / np.linalg.norm(q_vec)
    sims = faq_vecs @ q
    idx  = sims.argsort()[-k:][::-1]
    return sims[idx], idx

# ---------- STREAMLIT UI -----
st.set_page_config("AI先輩 FAQ Bot", "🤖")
st.title("🎓 AI先輩 – FAQチャットボット")

if "hist" not in st.session_state: st.session_state.hist=[]
for r,t in st.session_state.hist: st.chat_message(r).markdown(t)

if q := st.chat_input("質問をどうぞ"):
    st.chat_message("user").markdown(q)
    st.session_state.hist.append(("user", q))

    q_vec = genai.embed_content(model=EMBED_MODEL,
                                content=q,
                                task_type="retrieval_query")["embedding"]
    q_vec = np.asarray(q_vec, dtype="float32")
    sims, idxs = top_k(q_vec)

    # ---- プロンプト構築 ----------------------
    faqs = [
        f"{i+1}. Q: {faq_df.iloc[idx]['question']}\n   A: {faq_df.iloc[idx]['answer']}"
        for i,idx in enumerate(idxs) if sims[i] >= SIM_FLOOR
    ]
    prompt = (
        "あなたは大学の頼れる先輩チャットボットです。\n"
        "以下の候補 FAQ を読んで，質問に最も合う回答だけを選び，"
        "その回答文だけを返答として出力してください。\n"
        "もし該当するものが一つも無い場合は **NO_ANSWER** とだけ返してください。\n\n"
        f"【質問】\n{q}\n\n【候補FAQ】\n" + "\n".join(faqs)
    )

    # ---- Gemini 呼び出し --------------------
    try:
        rsp = genai.generate_content(
            model=CHAT_MODEL,
            contents=[{"role":"user","parts":[{"text":prompt}]}],
        )
        ans_raw = rsp.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        ans_raw = "システムエラーで回答できませんでした。時間をおいて試してください。"

    if ans_raw == "NO_ANSWER":
        answer = "ごめん、今はその情報を持っていないんだ。ほかの先輩にも聞いてみてね。"
        faq_hit = False
    else:
        answer = f"{ans_raw}\n\n困ったらいつでもまた聞いてね。"
        faq_hit = True

    st.chat_message("assistant").markdown(answer)
    st.session_state.hist.append(("assistant", answer))

    save_jsonl(CHAT_LOG, {"ts":time.time(),"q":q,"a":answer,"faq_hit":faq_hit})
# ────────────────────────────────────────────
# ─────────────────────────────────────────────
