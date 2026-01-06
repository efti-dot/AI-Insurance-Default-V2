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

        # Assistant message container
        with st.chat_message("assistant"):
            # Two placeholders: one for the thinking indicator, one for the streamed content
            thinking_placeholder = st.empty()
            reply_placeholder = st.empty()

            # Show a subtle animated "thinking..." indicator until first token arrives
            # (heartbeat dots to indicate the model is thinking)
            dots = ""
            thinking_placeholder.markdown(f"**Thinking:** Please wait{dots}")
            start_time = time.time()
            last_update = start_time

            # Kick off the stream
            response_stream = ai.client.chat.completions.create(
                model=ai.model,
                messages=ai.system_prompt + st.session_state.messages,
                temperature=1,
                stream=True
            )

            full_reply = ""
            first_token_received = False

            for chunk in response_stream:
                delta = chunk.choices[0].delta

                # While waiting for the first token, animate the thinking indicator
                if not first_token_received:
                    now = time.time()
                    if now - last_update >= 0.3:
                        dots = "." * (((int((now - start_time) / 0.3)) % 3) + 1)
                        thinking_placeholder.markdown(f"**Thinking:** Please wait{dots}")
                        last_update = now

                if delta.content:
                    # First token received: remove thinking indicator and start streaming
                    if not first_token_received:
                        thinking_placeholder.empty()
                        first_token_received = True

                    full_reply += delta.content
                    reply_placeholder.markdown(full_reply + " ▌")  # typing cursor

            # Finalize the streamed message
            if not first_token_received:
                # In rare cases with empty content (errors/timeouts), clear thinking
                thinking_placeholder.empty()
            reply_placeholder.markdown(full_reply)

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    AI_insurance_assistance()