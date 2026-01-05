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

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask anything...")
    if user_input:
        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream assistant response
        reply = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_stream = ai.client.chat.completions.create(
                model=ai.model,
                messages=ai.system_prompt + st.session_state.messages,
                temperature=1,
                stream=True
            )
            for chunk in response_stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    reply += delta.content
                    placeholder.markdown(reply + "▌")  # typing effect
            placeholder.markdown(reply)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    AI_insurance_assistance()