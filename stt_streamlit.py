import io
import uuid
import base64
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Assamese RAG Voice Assistant",
    page_icon="🎤",
    layout="wide",
)

st.title("🎤 Assamese RAG Voice Assistant")

st.caption(
    "Microphone → STT → RAG → Answer"
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_headers():
    return {
        "X-User-Id": user_id,
        "X-Request-Id": request_id,
    }


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_voice, tab_text = st.tabs(
    ["🎤 Voice", "💬 Text Chat"]
)


# ===========================================================================
# VOICE TAB
# ===========================================================================

with tab_voice:

    st.header("Voice Testing")

    st.write(
        "Record your question and test the STT transcription first."
    )

    # -----------------------------------------------------------------------
    # Recorder
    # -----------------------------------------------------------------------

    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#ff4b4b",
        neutral_color="#6c757d",
        icon_name="microphone",
        icon_size="2x",
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

            with st.spinner("Sending audio to Saaras v3..."):

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

                    st.error(f"Request failed: {e}")

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
                st.session_state.get("stt_data", {})
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

                    st.error(f"Request failed: {e}")

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

            st.write(
                result.get("answer", "")
            )
            # ---------------------------------------------------------------
            # TTS Audio
            # ---------------------------------------------------------------

            audio_base64 = result.get("audio_base64")

            if audio_base64:
                try:
                    audio_bytes = base64.b64decode(audio_base64)

                    st.subheader("🔊 Assistant Voice")

                    st.audio(
                        audio_bytes,
                        format="audio/wav",
                    )

                except Exception as e:
                    st.error(f"Could not play assistant audio: {e}")

            # ---------------------------------------------------------------
            # Sources
            # ---------------------------------------------------------------

            sources = result.get("sources", [])

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

    text_question = st.text_area(
        "Enter your question",
        placeholder="Ask something about the uploaded documents...",
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

                        st.write(
                            data.get("answer", "")
                        )

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