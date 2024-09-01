import os
import yaml
import streamlit as st
from llm_chains import load_normal_chain
from streamlit_mic_recorder import mic_recorder
from audio_handler import transcribe_audio
from image_handler import handle_image
from langchain.memory import StreamlitChatMessageHistory
from pdf_handler import add_documents_to_db
from utils import save_chat_history_json, get_timestamp, load_chat_history_json


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_chain(chat_history):
    """
        Loads a conversational chain with the provided chat history.

        This function acts as a wrapper that loads a normal conversational chain
        using the provided chat history. It delegates the task to the `load_normal_chain`
        function, which is expected to handle the specifics of loading the chain.

        Args:
            chat_history (list): A list representing the chat history, typically containing
            a sequence of messages exchanged in the conversation.

        Returns:
            The output of the `load_normal_chain` function, which is presumably a
            conversational chain object that includes the provided chat history.
    """
    return load_normal_chain(chat_history)


def clear_input_field():
    """
        Clears the user input field in the Streamlit app.

        This function transfers the content of `st.session_state.user_input` to
        `st.session_state.user_question` and then clears the `user_input` field.
        It is typically used after processing the user's input to reset the input field.

        Side Effects:
            - `st.session_state.user_question` is set to the current value of `st.session_state.user_input`.
            - `st.session_state.user_input` is cleared (set to an empty string).
    """
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""


def set_send_input():
    """
        Sets the send input flag and clears the input field in the Streamlit app.

        This function sets `st.session_state.send_input` to `True`, indicating that
        the user's input should be sent or processed. It then calls `clear_input_field()`
        to clear the input field after setting the flag.

        Side Effects:
            - `st.session_state.send_input` is set to `True`.
            - Clears `st.session_state.user_input` by calling `clear_input_field()`.
    """
    st.session_state.send_input = True
    clear_input_field()


def track_index():
    """
        Tracks the session index by updating the session state in the Streamlit app.

        This function assigns the value of `st.session_state.session_key` to
        `st.session_state.session_index_tracker`. It is used to track or update the
        session index within the Streamlit app's session state.

        Side Effects:
            - `st.session_state.session_index_tracker` is updated with the value of
              `st.session_state.session_key`.
    """
    st.session_state.session_index_tracker = st.session_state.session_key


def save_chat_history():
    """
        Saves the current chat history to a JSON file.

        This function checks if there is any chat history stored in `st.session_state.history`.
        If chat history exists, it determines whether the session is new or ongoing.
        For a new session, it generates a new session key using the current timestamp
        and saves the chat history to a new JSON file. For an ongoing session, it appends
        the chat history to an existing JSON file using the existing session key.

        Side Effects:
            - Updates `st.session_state.new_session_key` with a new session key if the
              current session is new.
            - Saves the chat history as a JSON file in the specified `chat_history_path`.

        Notes:
            - The path for saving the chat history is determined by the `config["chat_history_path"]` setting.
    """
    if st.session_state.history:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] +
                                   st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config["chat_history_path"] +
                                   st.session_state.session_key)


def main():
    st.title("MultiGPT :computer:")
    chat_container = st.container()
    st.sidebar.title("Chat Sessions")
    chat_sessions = ['new_session'] + os.listdir(config['chat_history_path'])

    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.session_key = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key is not None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index, on_change=track_index)

    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key='history')
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Type your message here", key="user_input", on_change=set_send_input)
    voice_recording_column, send_button_column = st.columns(2)
    with voice_recording_column:
        voice_recording = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            just_once=True
        )

    with send_button_column:
        send_button = st.button("Enter", key="send_button", on_click=clear_input_field)

    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png'])
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file", accept_multiple_files=True, key="pdf_upload", type=['pdf'])

    if uploaded_pdf:
        with st.spinner("Processing PDF..."):
            add_documents_to_db(uploaded_pdf)

    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        llm_chain.run("Summarize this text: " + transcribed_audio)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording['bytes'])
        llm_chain.run(transcribed_audio)

    if send_button or st.session_state.send_input:
        if uploaded_image:
            with st.spinner("Processing Image..."):
                user_message = "Describe this image in detail please."
                if st.session_state.user_question != "":
                    user_message = handle_image(uploaded_image.getvalue(), st.session_state.user_question)
                    st.session_state.user_question = ""
                llm_answer = handle_image(uploaded_image.getvalue(), st.session_state.user_question)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer)
        if st.session_state.user_question != "":
            # st.chat_message("user").write(st.session_state.user_question)
            llm_response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ""

    if chat_history.messages:
        with chat_container:
            st.write("Chat History:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history()


if __name__ == '__main__':
    main()
