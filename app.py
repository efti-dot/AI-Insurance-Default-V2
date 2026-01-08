import streamlit as st
import time
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

        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            reply_placeholder = st.empty()

            dots = ""
            thinking_placeholder.markdown(f"**Thinking:** Please wait{dots}")
            start_time = time.time()
            last_update = start_time

            full_reply = ""
            first_token_received = False

            for delta_content in ai.get_stream_response(user_input, st.session_state.messages):
                if not first_token_received:
                    now = time.time()
                    if now - last_update >= 0.3:
                        dots = "." * (((int((now - start_time) / 0.3)) % 3) + 1)
                        thinking_placeholder.markdown(f"**Thinking:** Please wait{dots}")
                        last_update = now

                if delta_content:
                    if not first_token_received:
                        thinking_placeholder.empty()
                        first_token_received = True

                    full_reply += delta_content
                    reply_placeholder.markdown(full_reply + " ▌")

            if not first_token_received:
                thinking_placeholder.empty()
            reply_placeholder.markdown(full_reply)

        st.session_state.messages.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    AI_insurance_assistance()