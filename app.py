import streamlit as st
import os
from prompt import OpenAIConfig

api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("Please check the OPENAI_API_KEY.")

ai = OpenAIConfig(api_key=api_key)

def AI_insurance_assistance():
    st.title("AI Insurance Assistance")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask anything...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = ai.get_response(user_input, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    AI_insurance_assistance()