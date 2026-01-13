import streamlit as st
import time
from prompt import OpenAIConfig

api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Please check the OPENAI_API_KEY.")

ai = OpenAIConfig(api_key=api_key)

def naive_bar():
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox("Select an option", ["General", "Premium"])
    return page

def general():
    st.title("AI Insurance Assistance (General)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask anything...")
    if user_input:
        # Persist user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant streaming with thinking indicator
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

        # Persist assistant reply
        st.session_state.messages.append({"role": "assistant", "content": full_reply})

def premium():
    st.title("AI Insurance Assistance (Premium)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = None

    # Render history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Attachment input (optional per message)
    attached_file = st.file_uploader(
        "Attach a file to this message",
        type=["pdf", "docx", "pptx", "ppt", "png", "jpg", "jpeg"]
    )

    # If a file is loaded, process it immediately when user submits a question
    user_input = st.chat_input("Ask anything (documents improve answers)...")
    processing_status_placeholder = st.empty()

    if user_input:
        # Save & show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process attachment (if provided)
        if attached_file is not None:
            # Prevent re-processing the same attachment name back-to-back
            if st.session_state.last_uploaded_name != attached_file.name:
                status_msg = ai.add_attachment(attached_file)
                st.session_state.last_uploaded_name = attached_file.name
            else:
                status_msg = f"✓ {attached_file.name} already added"
            processing_status_placeholder.info(status_msg)

        # Assistant streaming with KB context and thinking indicator
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            reply_placeholder = st.empty()

            dots = ""
            thinking_placeholder.markdown(f"**Thinking:** Please wait{dots}")
            start_time = time.time()
            last_update = start_time

            full_reply = ""
            first_token_received = False

            for delta_content in ai.get_premium_stream_response(user_input, st.session_state.messages):
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

        # Persist assistant reply
        st.session_state.messages.append({"role": "assistant", "content": full_reply})

if __name__ == "__main__":
    page = naive_bar()
    if page == "General":
        general()
    elif page == "Premium":
        premium()