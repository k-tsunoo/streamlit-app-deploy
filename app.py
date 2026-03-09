import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# APIキー読み込み
load_dotenv()

# LLM呼び出し関数
def get_llm_response(user_text, expert_type):

    # 専門家ごとのシステムメッセージ
    system_messages = {
        "健康アドバイザー": "あなたは健康管理の専門家です。健康維持や生活習慣についてわかりやすくアドバイスしてください。",
        "スポーツインストラクター": "あなたはスポーツ指導の専門家です。運動方法やトレーニングについてわかりやすくアドバイスしてください。",
        "栄養アドバイザー": "あなたは栄養学の専門家です。食事や栄養バランスについてわかりやすくアドバイスしてください。"
    }

    system_message = system_messages.get(expert_type)

    # プロンプト作成
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])

    # LLM設定
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # LangChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # LLM実行
    result = chain.run(question=user_text)

    return result


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("サンプルアプリ③: LLMアドバイザー")

st.write("### アプリ概要")
st.write("""
このアプリでは、生成AIに専門家としてアドバイスをしてもらうことができます。

以下の手順で利用してください。

1. 専門家の種類を選択します  
2. 質問内容を入力します  
3. 「実行」ボタンを押します  

AIが選択した専門家として回答します。
""")

st.divider()

# 専門家選択
selected_expert = st.radio(
    "専門家の種類を選択してください。",
    ["健康アドバイザー", "スポーツインストラクター", "栄養アドバイザー"]
)

# 入力フォーム
input_text = st.text_input("質問内容を入力してください。")

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
        st.error("質問内容を入力してください。")