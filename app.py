
import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# --- 1. 環境変数の読み込み ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("エラー: .env ファイルに GOOGLE_API_KEY が設定されていません。")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- 2. Gemini モデルの設定 ---
GENERATION_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# --- 3. FAQデータの読み込みとFAISSインデックスの作成 ---
FAQ_FILE = "faq.csv"
FAISS_THRESHOLD = 0.8  # 回答なしと判断する類似度閾値

@st.cache_resource
def load_faq_and_create_faiss_index():
    try:
        df_faq = pd.read_csv(FAQ_FILE)
        st.success(f"{FAQ_FILE} を読み込みました。質問数: {len(df_faq)}")
    except FileNotFoundError:
        st.error(f"エラー: {FAQ_FILE} が見つかりません。app.py と同じディレクトリに配置してください。")
        st.stop()
    except Exception as e:
        st.error(f"エラー: {FAQ_FILE} の読み込み中に問題が発生しました: {e}")
        st.stop()

    questions = df_faq["question"].tolist()
    answers = df_faq["answer"].tolist()

    st.write("FAQの埋め込みを生成中...")
    embeddings = []
    try:
        # バッチ処理でエンベディングを生成
        for i in range(0, len(questions), 10):
            batch = questions[i : i + 10]
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="RETRIEVAL_QUERY"
            )
            batch_embeds = [item["embedding"] if isinstance(item, dict) else item for item in response]
            embeddings.extend(batch_embeds)
            st.write(f"  - {min(i + 10, len(questions))} / {len(questions)} 件の埋め込みを生成...")

        st.success("FAQの埋め込み生成が完了しました。")
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.success("FAISSインデックスを作成しました。")
        return index, questions, answers

    except Exception as e:
        st.error(f"エラー: エンベディング生成またはFAISSインデックス作成中に問題が発生しました: {e}")
        st.stop()

faiss_index, faq_questions, faq_answers = load_faq_and_create_faiss_index()

# --- 4. ログファイルの準備 ---
CHAT_LOG_FILE = "chat_logs.csv"
UNANSWERED_LOG_FILE = "unanswered.csv"

def initialize_log_files():
    if not os.path.exists(CHAT_LOG_FILE):
        pd.DataFrame(columns=["timestamp", "user_question", "bot_answer", "is_unanswered", "source_faq_question"]).to_csv(CHAT_LOG_FILE, index=False)
    if not os.path.exists(UNANSWERED_LOG_FILE):
        pd.DataFrame(columns=["timestamp", "user_question"]).to_csv(UNANSWERED_LOG_FILE, index=False)

initialize_log_files()

def append_log(file_path, data):
    df = pd.DataFrame([data])
    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path) or pd.read_csv(file_path).empty, index=False)

# --- 5. Streamlit UI とチャットロジック ---
st.set_page_config(page_title="AI先輩", page_icon="🎓")
st.title("🎓 AI先輩")
st.caption("履修登録やキャンパス情報、生活情報に答えるチャットボットです。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI先輩が考え中..."):
            user_embedding = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=prompt,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]
            user_embedding_np = np.array(user_embedding).astype('float32').reshape(1, -1)

            D, I = faiss_index.search(user_embedding_np, 1)
            top_similarity = 1 - (D[0][0] / 2)

            if top_similarity >= FAISS_THRESHOLD:
                retrieved_question = faq_questions[I[0][0]]
                retrieved_answer = faq_answers[I[0][0]]
                context_text = f"参考情報:
質問: {retrieved_question}
回答: {retrieved_answer}
---"
                is_unanswered = False
                source_faq = retrieved_question
            else:
                context_text = "参考情報: 該当する情報は見つかりませんでした。
---"
                is_unanswered = True
                source_faq = "N/A (unanswered)"

            system_prompt = '''
            あなたは大学の学生支援チャットボット「AI先輩」です。
            以下の「参考情報」を基に、ユーザーの質問に日本語で丁寧に回答してください。
            参考情報に該当する情報が全くない場合や関連性が低い場合は、
            「申し訳ありません、その情報については現在持ち合わせておりません。」と回答してください。
            '''
            try:
                model = genai.GenerativeModel(GENERATION_MODEL)
                response = model.generate_content(f"{system_prompt}
{context_text}
ユーザーからの質問:
{prompt}")
                bot_response = response.text

                if "申し訳ありません、その情報については現在持ち合わせておりません" in bot_response:
                    is_unanswered = True
            except Exception as e:
                bot_response = f"回答生成中にエラーが発生しました。({e})"
                is_unanswered = True

        st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": current_time,
            "user_question": prompt,
            "bot_answer": bot_response,
            "is_unanswered": is_unanswered,
            "source_faq_question": source_faq
        }
        append_log(CHAT_LOG_FILE, log_data)

        if is_unanswered:
            unanswered_data = {
                "timestamp": current_time,
                "user_question": prompt
            }
            append_log(UNANSWERED_LOG_FILE, unanswered_data)

        st.caption(f"（類似度: {top_similarity:.2f}, 未回答: {is_unanswered}）")
