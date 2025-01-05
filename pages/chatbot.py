import streamlit as st
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="🤗💬 Help Care bot")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hai selamat datang di help care chat"
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def generate_response(question: str):
    api_key = os.getenv("GENAI_APIKEY")
    model_selected = os.getenv("GENAI_MODEL")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_selected)

    join_question = question.join(
        " please provide your anwear with bahasa and make it easy to understand"
    )

    response = model.generate_content(join_question)
    return response.text


if prompt := st.chat_input("ketik pertanyaan mu disini"):
    pegawai_name = ""

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()

            answer = ""

            for ans in response:
                answer += ans
                placeholder.markdown(answer)

                time.sleep(0.005)

    message = {
        "role": "assistant",
        "content": response
    }

    st.session_state.messages.append(message)
