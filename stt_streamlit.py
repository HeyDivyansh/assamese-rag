import io
import re
import uuid
import base64
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from voice_component import voice_component

def clean_answer(text):
    if not text:
        return ""

    # Remove source citations such as [s1], [s1, s2], or standalone s1.
    text = re.sub(r"\[\s*s\d+(?:\s*,\s*s\d+)*\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)s\d+(?!\w)", "", text, flags=re.IGNORECASE)

    # Remove leftover empty citation brackets such as [] or [   ].
    text = re.sub(r"\[\s*\]", "", text)

    # Remove extra spaces created after removing citations.
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Clean spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"

LANGUAGES = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Kannada": "kn-IN",
    "Assamese": "as-IN",
}


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multilingual RAG Voice Assistant",
    page_icon="🎤",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Language Selection
# ---------------------------------------------------------------------------

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Assamese": "as",
}

if "language" not in st.session_state:
    st.session_state["language"] = "English"

st.subheader("🌐 Select Language")

selected_language = st.selectbox(
    "Choose the language for your conversation",
    options=list(LANGUAGES.keys()),
    index=list(LANGUAGES.keys()).index(st.session_state["language"]),
    key="language_selector",
)

selected_code = LANGUAGES[selected_language]

# Detect language change
if st.session_state["language"] != selected_language:

    st.session_state["language"] = selected_language

    # Clear previous conversation/results when language changes
    keys_to_clear = [
        "transcript",
        "stt_data",
        "voice_result",
        "text_answer",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    st.rerun()


# ===========================================================================
# SELECTED LANGUAGE
# ===========================================================================
selected_language = st.session_state.get("language", "English")
language_code = LANGUAGES[selected_language]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎤 Multilingual RAG Voice Assistant")

st.caption(
    f"Current language: **{selected_language}** ({language_code})"
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

st.sidebar.header("Configuration")

user_id = st.sidebar.text_input(
    "X-User-Id",
    value="550e8400-e29b-41d4-a716-446655440000",
)

request_id = st.sidebar.text_input(
    "X-Request-Id",
    value=str(uuid.uuid4()),
)

st.sidebar.write("API:")
st.sidebar.code(API_BASE_URL)

st.sidebar.divider()

st.sidebar.write("Selected Language:")
st.sidebar.success(
    f"{selected_language} ({language_code})"
)

if st.sidebar.button("🌐 Change Language"):
    st.session_state["language"] = None

    # Clear previous language-specific results
    for key in [
        "transcript",
        "stt_data",
        "voice_result",
    ]:
        st.session_state.pop(key, None)

    st.rerun()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_headers():
    return {
        "X-User-Id": user_id,
        "X-Request-Id": request_id,

        # Selected language
        "X-Language": language_code,
    }


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_upload, tab_continuous_voice, tab_voice, tab_text = st.tabs(
    [
        "📄 Upload Document",
        "🚀 Intelligent Voice AI",
        "🎤 Voice AI",
        "💬 Text Chat",
    ]
)


# ===========================================================================
# DOCUMENT UPLOAD TAB
# ===========================================================================

with tab_upload:

    st.header("📄 Upload Document")

    st.write(
        "Upload a PDF document. The existing backend ingestion pipeline "
        "will handle parsing, chunking, embeddings, and indexing."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    if uploaded_file:

        st.success(
            f"Selected: {uploaded_file.name}"
        )

        st.write(
            f"File size: {uploaded_file.size:,} bytes"
        )

        if st.button(
            "📤 Upload & Ingest",
            type="primary",
            key="upload_document_button",
        ):

            with st.spinner("Uploading and starting ingestion..."):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/documents",
                        headers=get_headers(),
                        files=files,
                        timeout=300,
                    )

                    if response.ok:

                        data = response.json()

                        st.success(
                            "✅ Document uploaded successfully!"
                        )

                        st.subheader("Ingestion Response")

                        st.json(data)

                    else:

                        st.error(
                            f"❌ Upload failed ({response.status_code})"
                        )

                        st.code(response.text)

                except Exception as e:

                    st.error(
                        f"❌ Request failed: {e}"
                    )
            # ---------------------------------------------------------------
            # We will connect your existing Swagger ingestion endpoint here.
            #
            # DO NOT change the backend ingestion logic.
            # ---------------------------------------------------------------

            # Example structure:
            #
            # files = {
            #     "file": (
            #         uploaded_file.name,
            #         uploaded_file.getvalue(),
            #         "application/pdf",
            #     )
            # }
            #
            # response = requests.post(
            #     f"{API_BASE_URL}/YOUR_INGEST_ENDPOINT",
            #     headers=get_headers(),
            #     files=files,
            #     timeout=300,
            # )


# ===========================================================================
# INTELLIGENT VOICE AI TAB
# ===========================================================================

with tab_continuous_voice:
    st.header("🚀 Intelligent Voice AI")
    st.write(
        "This mode supports continuous listening and interruption (barge-in). "
        "When the assistant is speaking, you can start talking to interrupt it."
    )
    
    # Render the custom voice component
    voice_component()


# ===========================================================================
# VOICE AI TAB
# ===========================================================================

with tab_voice:

    st.header("Voice AI")

    st.write(
        f"Record your question in **{selected_language}**."
    )

    # -----------------------------------------------------------------------
    # Recorder
    # -----------------------------------------------------------------------

    audio_bytes = audio_recorder(
        text="Click to record",
        # Keep recording through normal pauses; click the mic to stop it.
        pause_threshold=3600,
        recording_color="#ff4b4b",
        neutral_color="#6c757d",
        icon_name="microphone",
        icon_size="2x",
        key="voice_ai_recorder",
    )

    if audio_bytes:

        st.success(
            f"Recording captured — {len(audio_bytes):,} bytes"
        )

        # -------------------------------------------------------------------
        # Play recording
        # -------------------------------------------------------------------

        st.subheader("Your Recording")

        st.audio(
            audio_bytes,
            format="audio/wav",
        )

        # -------------------------------------------------------------------
        # STT
        # -------------------------------------------------------------------

        st.subheader("1️⃣ Speech-to-Text")

        if st.button(
            "Transcribe Audio",
            key="transcribe_button",
            type="primary",
        ):

            with st.spinner(
                f"Transcribing in {selected_language}..."
            ):

                try:

                    files = {
                        "file": (
                            "recording.wav",
                            io.BytesIO(audio_bytes),
                            "audio/wav",
                        )
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/chat/voice/transcribe",
                        headers=get_headers(),
                        files=files,
                        timeout=120,
                    )

                    if response.ok:

                        data = response.json()

                        st.session_state["transcript"] = (
                            data.get("transcript", "")
                        )

                        st.session_state["stt_data"] = data

                    else:

                        st.error(
                            f"STT failed ({response.status_code})"
                        )

                        st.code(response.text)

                except Exception as e:

                    st.error(
                        f"Request failed: {e}"
                    )

        # -------------------------------------------------------------------
        # Display transcript
        # -------------------------------------------------------------------

        if "transcript" in st.session_state:

            st.subheader("📝 Transcript")

            st.text_area(
                "STT Output",
                value=st.session_state["transcript"],
                height=120,
                disabled=True,
            )

            st.write(
                st.session_state.get(
                    "stt_data",
                    {},
                )
            )

        # -------------------------------------------------------------------
        # Full voice RAG
        # -------------------------------------------------------------------

        st.subheader("2️⃣ Voice → RAG → Answer")

        if st.button(
            "Ask RAG Using This Audio",
            key="voice_rag_button",
        ):

            with st.spinner(
                "Running STT → Retrieval → LLM..."
            ):

                try:

                    files = {
                        "file": (
                            "recording.wav",
                            io.BytesIO(audio_bytes),
                            "audio/wav",
                        )
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/chat/voice",
                        headers=get_headers(),
                        files=files,
                        timeout=180,
                    )

                    if response.ok:

                        data = response.json()

                        st.session_state["voice_result"] = data

                    else:

                        st.error(
                            f"Voice RAG failed ({response.status_code})"
                        )

                        st.code(response.text)

                except Exception as e:

                    st.error(
                        f"Request failed: {e}"
                    )

        # -------------------------------------------------------------------
        # Display RAG result
        # -------------------------------------------------------------------

        if "voice_result" in st.session_state:

            result = st.session_state["voice_result"]

            st.subheader("📝 Transcribed Question")

            st.text_area(
                "Question",
                value=result.get("transcript", ""),
                height=100,
                disabled=True,
            )

            st.subheader("🤖 RAG Answer")

            answer = clean_answer(result.get("answer", ""))

            st.write(answer)

            # ---------------------------------------------------------------
            # TTS Audio
            # ---------------------------------------------------------------

            audio_base64 = result.get(
                "audio_base64"
            )

            if audio_base64:

                try:

                    audio_bytes = base64.b64decode(
                        audio_base64
                    )

                    st.subheader(
                        "🔊 Assistant Voice"
                    )

                    st.audio(
                        audio_bytes,
                        format="audio/wav",
                    )

                except Exception as e:

                    st.error(
                        f"Could not play assistant audio: {e}"
                    )

            # ---------------------------------------------------------------
            # Sources
            # ---------------------------------------------------------------

            sources = result.get(
                "sources",
                [],
            )

            if sources:

                with st.expander(
                    f"📚 Sources ({len(sources)})"
                ):

                    for i, source in enumerate(
                        sources,
                        start=1,
                    ):

                        st.markdown(
                            f"**Source {i}**"
                        )

                        st.write(
                            source
                        )


# ===========================================================================
# TEXT TAB
# ===========================================================================

with tab_text:

    st.header("Text Chat")

    st.write(
        f"Ask questions only in **{selected_language}**."
    )

    text_question = st.text_area(
        "Enter your question",
        placeholder=(
            f"Type your question in {selected_language}..."
        ),
        height=120,
    )

    if st.button(
        "Ask Question",
        key="text_button",
        type="primary",
    ):

        if not text_question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Running RAG..."
            ):

                try:

                    payload = {
                        "message": text_question,
                        
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/chat/text",
                        headers=get_headers(),
                        json=payload,
                        timeout=180,
                    )

                    if response.ok:

                        data = response.json()

                        st.subheader(
                            "🤖 Answer"
                        )

                        answer = clean_answer(data.get("answer", ""))

                        st.write(answer)

                        sources = data.get(
                            "sources",
                            [],
                        )

                        if sources:

                            with st.expander(
                                f"📚 Sources ({len(sources)})"
                            ):

                                for i, source in enumerate(
                                    sources,
                                    start=1,
                                ):

                                    st.markdown(
                                        f"**Source {i}**"
                                    )

                                    st.write(
                                        source
                                    )

                    else:

                        st.error(
                            f"Request failed ({response.status_code})"
                        )

                        st.code(
                            response.text
                        )

                except Exception as e:

                    st.error(
                        f"Request failed: {e}"
                    )