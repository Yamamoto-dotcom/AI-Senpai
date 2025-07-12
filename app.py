import os, json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ───────── CONFIG ─────────
EMBED_MODEL   = "models/embedding-001"
CHAT_MODEL    = "models/gemini-1.5-flash-latest"
TOP_K         = 5            # LLM に渡す FAQ 件数
SIM_CUTOFF    = 0.15         # これ未満は FAQ を渡さない
HIGH_TH, MID_TH = 0.80, 0.30 # メタ情報表示用

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG = LOG_DIR / "chat_logs.jsonl"

# ───────── UTIL ──────────
def log_jsonl(path: Path, row: dict):
    norm = lambda v: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: norm(v) for k, v in row.items()},
                           ensure_ascii=False) + "\n")

# ───────── API KEY ────────
load_dotenv(); key = os.getenv("GOOGLE_API_KEY")
if not key:
    st.error("GOOGLE_API_KEY を .env か Secrets に設定してください"); st.stop()
genai.configure(api_key=key)

# ───────── FAQ LOAD ───────
@st.cache_resource(show_spinner="FAQ を読み込み中...")
def load_faq(csv="faq.csv"):
    df = pd.read_csv(csv)
    vecs = [
        genai.embed_content(model=EMBED_MODEL,
                            content=q,
                            task_type="retrieval_query")["embedding"]
        for q in df["question"]
    ]
    vecs = np.asarray(vecs, dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)   # コサイン用正規化
    return df, vecs
faq_df, faq_vecs = load_faq()

def top_k_cos(q_vec, k=TOP_K):
    q = q_vec / np.linalg.norm(q_vec)
    sims = faq_vecs @ q
    idx  = sims.argsort()[-k:][::-1]
    return sims[idx], idx

# ───────── STREAMLIT UI ───
st.set_page_config("AI先輩 FAQ Bot", "🤖")
st.title("🎓 AI先輩 – FAQチャットボット")

if "hist" not in st.session_state: st.session_state.hist=[]
for role, txt in st.session_state.hist:
    st.chat_message(role).markdown(txt)

# ───────── CHAT LOOP ──────
if user_q := st.chat_input("質問をどうぞ"):
    st.chat_message("user").markdown(user_q)
    st.session_state.hist.append(("user", user_q))

    # 埋め込み & 類似検索
    q_vec = genai.embed_content(model=EMBED_MODEL,
                                content=user_q,
                                task_type="retrieval_query")["embedding"]
    q_vec = np.asarray(q_vec, dtype="float32")
    sims, idxs = top_k_cos(q_vec)

    # FAQ が十分近いものだけを LLM に渡す
    faq_lines = [
        f"{i+1}. Q: {faq_df.iloc[idx]['question']}\n   A: {faq_df.iloc[idx]['answer']}"
        for i, idx in enumerate(idxs) if sims[i] >= SIM_CUTOFF
    ]

    prompt = (
        "あなたは大学の頼れる先輩チャットボットです。\n"
        "以下の候補 FAQ を参考に、質問に最も合う回答だけを返答してください。\n"
        "該当する回答が無い場合は **NO_ANSWER** とだけ返してください。\n\n"
        f"【質問】\n{user_q}\n\n【候補FAQ】\n" + ("\n".join(faq_lines) or "（該当なし）")
    )

    # LLM 呼び出し
    try:
        rsp = genai.generate_content(
            model=CHAT_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        raw_ans = rsp.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        raw_ans = "NO_ANSWER"
        st.error(f"Gemini API エラー: {type(e).__name__}")
    
    # 3 段階の最終返答整形
    if raw_ans == "NO_ANSWER":
        answer = "ごめん、いまはその情報を持っていないんだ。他の先輩にも聞いてみてね。"
        faq_hit = False
    elif sims[0] >= HIGH_TH:
        answer = f"{raw_ans}\n\nまた困ったらいつでも聞いてね！"
        faq_hit = True
    else:
        answer = f"{raw_ans}\n\n念のため他の先輩にも確認してみて！"
        faq_hit = True

    st.chat_message("assistant").markdown(answer)
    st.session_state.hist.append(("assistant", answer))

    log_jsonl(CHAT_LOG, {
        "ts": time.time(),
        "question": user_q,
        "answer": answer,
        "faq_hit": faq_hit,
        "top_similarity": float(sims[0] if sims.size else 0)
    })
# ─────────────────────────────────────────────
