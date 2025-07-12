import os, json, time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# ────────── CONFIG ──────────
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL  = "models/gemini-1.5-flash-latest"

TOP_K      = 5      # LLM に渡す FAQ 件数
SIM_CUTOFF = 0.15   # これ未満は FAQ を渡さない
HIGH_TH    = 0.80   # 「完全一致」とみなす境界
MID_TH     = 0.30   # 「参考になる」境界

LOG_DIR  = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG = LOG_DIR / "chat_logs.jsonl"

# ────────── UTILS ──────────
def jsonl_append(path: Path, row: dict):
    safe = {
        k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
        for k, v in row.items()
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")

# ────────── ENV & API KEY ──────────
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY を .env または Secrets に設定してください。")
    st.stop()
genai.configure(api_key=api_key)

# モデルは configure 後に生成
MODEL = genai.GenerativeModel(CHAT_MODEL)

# ────────── FAQ 読み込み ──────────
@st.cache_resource(show_spinner="FAQ を読み込み中…")
def load_faq(csv_path: str = "faq.csv"):
    if not Path(csv_path).exists():
        st.error(f"{csv_path} が見つかりません"); st.stop()

    df = pd.read_csv(csv_path)
    if not {"question", "answer"}.issubset(df.columns):
        st.error("faq.csv に 'question','answer' 列が必要です"); st.stop()

    vecs = [
        genai.embed_content(
            model=EMBED_MODEL,
            content=q,
            task_type="retrieval_query"
        )["embedding"]
        for q in df["question"]
    ]
    vecs = np.asarray(vecs, dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # L2 正規化
    return df, vecs

faq_df, faq_vecs = load_faq()

def top_k_cosine(q_vec, k: int = TOP_K):
    q = q_vec / np.linalg.norm(q_vec)
    sims = faq_vecs @ q
    idxs = sims.argsort()[-k:][::-1]
    return sims[idxs], idxs

# ────────── STREAMLIT UI ──────────
st.set_page_config("AI先輩 FAQ Bot", "🤖")
st.title("🎓 AI先輩 – FAQチャットボット")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# ────────── MAIN LOOP ──────────
if user_q := st.chat_input("質問をどうぞ"):
    st.chat_message("user").markdown(user_q)
    st.session_state.history.append(("user", user_q))

    # 1) 埋め込み
    q_vec = genai.embed_content(
        model=EMBED_MODEL,
        content=user_q,
        task_type="retrieval_query"
    )["embedding"]
    q_vec = np.asarray(q_vec, dtype="float32")

    # 2) 類似 FAQ を取得
    sims, idxs = top_k_cosine(q_vec)

    # 3) プロンプト生成
    faq_lines = [
        f"{i+1}. Q: {faq_df.iloc[idx]['question']}\n   A: {faq_df.iloc[idx]['answer']}"
        for i, idx in enumerate(idxs)
        if sims[i] >= SIM_CUTOFF
    ]
    prompt = (
        "あなたは大学の頼れる先輩チャットボットです。\n"
        "次の候補 FAQ を参考に、質問に最も合う回答だけを返答してください。\n"
        "該当するものが無ければ **NO_ANSWER** とだけ返してください。\n\n"
        f"【質問】\n{user_q}\n\n【候補FAQ】\n"
        + ("\n".join(faq_lines) if faq_lines else "（候補なし）")
    )

    # 4) Gemini Flash で回答選択
    try:
        resp   = MODEL.generate_content(prompt)
        raw_ans = resp.text.strip()
    except Exception as e:
        raw_ans = "NO_ANSWER"
        st.error(f"Gemini API エラー: {type(e).__name__} – {e}")

    # 5) 3 段階で最終整形
    if raw_ans == "NO_ANSWER":
        answer  = "ごめん、いまはその情報を持っていないんだ。他の先輩にも聞いてみてね。"
        faq_hit = False
    elif sims[0] >= HIGH_TH:
        answer  = f"{raw_ans}\n\nまた何かあったら遠慮なく聞いてね！"
        faq_hit = True
    else:
        answer  = f"{raw_ans}\n\n念のため他の先輩にも確認してみて！"
        faq_hit = True

    # 6) 表示＆履歴
    st.chat_message("assistant").markdown(answer)
    st.session_state.history.append(("assistant", answer))

    # 7) ログ保存
    jsonl_append = jsonl_append if 'jsonl_append' in globals() else log_jsonl
    jsonl_append(
        CHAT_LOG,
        {
            "ts": time.time(),
            "question": user_q,
            "answer": answer,
            "faq_hit": faq_hit,
            "top_similarity": float(sims[0] if sims.size else 0),
        },
    )
