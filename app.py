import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# .envからのAPIキー読み込み
load_dotenv()

# Streamlit SecretsからAPIキー取得
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# LLM実行関数
def get_llm_response(user_text, expert_type):

    # 専門家ごとのシステムメッセージ
    system_messages = {
        "健康アドバイザー": "あなたは健康管理の専門家です。生活習慣や健康維持についてわかりやすくアドバイスしてください。",
        "スポーツインストラクター": "あなたはスポーツトレーニングの専門家です。運動方法やトレーニングについてわかりやすくアドバイスしてください。",
        "栄養アドバイザー": "あなたは栄養学の専門家です。食事や栄養バランスについてわかりやすくアドバイスしてください。"
    }

    system_message = system_messages.get(expert_type)

    # プロンプト
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    # LangChain実行
    chain = prompt | llm

    result = chain.invoke({"question": user_text})

    return result.content


# -------------------------
# Streamlit UI
# -------------------------

st.title("AI健康専門家アドバイザー")

st.write("### アプリ概要")
st.write("""
このアプリでは健康、運動、栄養に関する質問に対して生成AIが専門家として回答するアプリです。

【使い方】

1. 専門家の種類を選択  
2. 質問を入力  
3. 実行ボタンを押す  

AIが選択された専門家としてアドバイスを行います。
""")

st.divider()

# 専門家選択
selected_expert = st.radio(
    "専門家の種類を選択してください",
    ["健康アドバイザー (全般)", "スポーツインストラクター (運動に関する質問)", "栄養アドバイザー (食事に関する質問)"]
)

# 入力フォーム
input_text = st.text_input("質問を入力してください")

st.divider()

# 実行ボタン
if st.button("実行"):

    if input_text:

        with st.spinner("AIが回答を生成しています..."):

            answer = get_llm_response(input_text, selected_expert)

        st.divider()

        st.write("### AIからの回答")
        st.write(answer)

    else:
        st.error("質問を入力してください")