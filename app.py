import os, json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ───── CONFIG ─────
EMBED_MODEL = "models/embedding-001"
HIGH_TH     = 0.80   # 完全一致
MID_TH      = 0.30   # 参考になる
LOG_DIR     = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG    = LOG_DIR / "chat_logs.jsonl"

# ───── UTILS ─────
def log_line(path: Path, row: dict):
    safe = {k: (v if isinstance(v,(str,int,float,bool,type(None))) else str(v))
            for k,v in row.items()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")

# ───── API KEY ─────
load_dotenv(); key = os.getenv("GOOGLE_API_KEY")
if not key:  st.error("GOOGLE_API_KEY を設定してください"); st.stop()
genai.configure(api_key=key)

# ───── FAQ 読込 ─────
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

def best_match(q_vec):
    q = q_vec / np.linalg.norm(q_vec)
    sims = faq_vecs @ q
    idx  = int(sims.argmax())
    return float(sims[idx]), idx

# ───── UI ─────
st.set_page_config("AI先輩 FAQ Bot", "🤖")
st.title("🎓 AI先輩 – FAQチャットボット")
if "hist" not in st.session_state: st.session_state.hist=[]
for r,t in st.session_state.hist: st.chat_message(r).markdown(t)

# ───── LOOP ─────
if q := st.chat_input("質問をどうぞ"):
    st.chat_message("user").markdown(q)
    st.session_state.hist.append(("user", q))

    q_vec = genai.embed_content(model=EMBED_MODEL,
                                content=q,
                                task_type="retrieval_query")["embedding"]
    q_vec = np.asarray(q_vec, dtype="float32")
    sim, idx = best_match(q_vec)
    faq_q, faq_a = faq_df.iloc[idx][["question","answer"]]

    if sim >= HIGH_TH:                  # ① 完全一致
        ans = f"{faq_a}\n\nまた何かあったら遠慮なく聞いてね！"
    elif sim >= MID_TH:                 # ② 参考になる
        ans = (f"参考になりそうな情報だよ：\n\n{faq_a}\n\n"
               "念のため他の先輩にも確認してみて！")
    else:                               # ③ 分からない
        ans = "ごめん、いまはその情報を持っていないんだ。他の先輩にも聞いてみてね。"

    st.chat_message("assistant").markdown(ans)
    st.session_state.hist.append(("assistant", ans))
    st.caption(f"コサイン類似度: {sim:.2f}")

    log_line(CHAT_LOG, {"ts":time.time(),"q":q,"a":ans,"sim":sim,"faq_q":faq_q if sim>=MID_TH else ""})
