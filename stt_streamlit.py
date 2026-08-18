import io
import uuid
import base64
import requests
import streamlit as st
import threading
import queue
import numpy as np
import av
import time
import wave

from streamlit_webrtc import webrtc_streamer

import sys
from pathlib import Path

ANC_PATH = (
    Path(__file__).resolve().parent
    / "tools"
    / "Active-Noise-Cancelling"
)

if str(ANC_PATH) not in sys.path:
    sys.path.insert(0, str(ANC_PATH))

from dsp import NoiseSuppressor

def denoise_audio(audio_bytes: bytes) -> bytes:
    """
    Apply Active-Noise-Cancelling DSP to recorded WAV audio.
    Converts stereo input to mono and returns denoised WAV bytes.
    """
    import io
    import wave
    import numpy as np

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        wav_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM audio, got sample width={sample_width}"
        )

    audio = np.frombuffer(
        frames,
        dtype=np.int16,
    ).astype(np.float32)

    # ---------------------------------------------------------
    # Convert stereo/multi-channel audio to mono
    # ---------------------------------------------------------

    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = audio.mean(axis=1)

    audio /= 32768.0

    # ---------------------------------------------------------
    # Noise suppressor
    # ---------------------------------------------------------

    suppressor = NoiseSuppressor(
        sr=wav_rate,
        frame_ms=20,
        beta=1.0,
        noise_floor=0.02,
        ema_alpha=0.96,
        gain_smooth=0.8,
        highpass_hz=80.0,
    )

    hop = suppressor.hop

    # Pad to complete hop
    padding = (-len(audio)) % hop

    if padding:
        audio = np.pad(
            audio,
            (0, padding),
        )

    denoised_chunks = []

    for start in range(0, len(audio), hop):
        chunk = audio[start:start + hop].astype(np.float32)

        denoised = suppressor.process(chunk)

        denoised_chunks.append(denoised)

    denoised_audio = np.concatenate(denoised_chunks)

    denoised_audio = np.clip(
        denoised_audio,
        -1.0,
        1.0,
    )

    output = (
        denoised_audio * 32767.0
    ).astype(np.int16)

    output_buffer = io.BytesIO()

    with wave.open(output_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(wav_rate)
        wav.writeframes(output.tobytes())

    return output_buffer.getvalue()

# ---------------------------------------------------------------------------
# Live microphone state
# ---------------------------------------------------------------------------

def create_audio_state():
    return {
        "lock": threading.Lock(),

        # Current recording buffer
        "buffer": [],

        # Recording state
        "recording": False,
        "recording_started": 0.0,

        # First recording = exactly 5 seconds
        "initial_recording": True,

        # After initial recording, wait for speech
        "waiting_for_speech": True,

        # Completed audio waiting for processing
        "completed_audio": None,

        # Audio properties
        "sample_rate": 48000,

        # VAD
        "noise_floor": 0.005,
        "speech_active": False,
        "last_voice_time": 0.0,

        # TTS interruption
        "interrupt": False,

        # Cancel
        "cancel": False,
    }


def create_audio_callback(state):
    
    INITIAL_RECORD_SECONDS = 5.0
    SILENCE_SECONDS = 0.8
    SPEECH_THRESHOLD = 0.015

    def audio_frame_callback(frame: av.AudioFrame):

        try:
            samples = frame.to_ndarray()

            if samples.ndim > 1:
                samples = np.mean(
                    samples,
                    axis=0,
                )

            samples = samples.astype(np.float32)

            if np.max(np.abs(samples)) > 1.5:
                samples /= 32768.0

            now = time.monotonic()

            rms = float(
                np.sqrt(
                    np.mean(samples * samples)
                )
            )

            with state["lock"]:

                state["sample_rate"] = (
                    frame.sample_rate or 48000
                )

                # -------------------------------------------------
                # Cancel
                # -------------------------------------------------

                if state["cancel"]:
                    state["buffer"] = []
                    state["recording"] = False
                    state["recording_started"] = 0.0
                    state["speech_active"] = False
                    state["cancel"] = False

                # -------------------------------------------------
                # INITIAL QUERY
                # Record exactly 5 seconds.
                # -------------------------------------------------

                if state["initial_recording"]:

                    if not state["recording"]:

                        state["recording"] = True
                        state["recording_started"] = now
                        state["buffer"] = []

                        print(
                            "DEBUG: Initial 5-second recording started"
                        )

                    state["buffer"].append(
                        samples.copy()
                    )

                    if (
                        now
                        - state["recording_started"]
                        >= INITIAL_RECORD_SECONDS
                    ):

                        state["completed_audio"] = (
                            np.concatenate(
                                state["buffer"]
                            ),
                            state["sample_rate"],
                        )

                        state["buffer"] = []
                        state["recording"] = False
                        state["initial_recording"] = False
                        state["waiting_for_speech"] = True

                        print(
                            "DEBUG: Initial 5-second recording completed"
                        )

                # -------------------------------------------------
                # AFTER INITIAL QUERY
                # Wait for speech, then record until silence.
                # -------------------------------------------------

                elif state["waiting_for_speech"]:

                    threshold = max(
                        SPEECH_THRESHOLD,
                        state["noise_floor"] * 3.0,
                    )

                    if rms > threshold:

                        state["waiting_for_speech"] = False
                        state["recording"] = True
                        state["speech_active"] = True
                        state["recording_started"] = now
                        state["last_voice_time"] = now
                        state["buffer"] = [
                            samples.copy()
                        ]

                        # Tell Streamlit to stop TTS.
                        state["interrupt"] = True

                        print(
                            "DEBUG: Interruption speech detected"
                        )

                    elif rms < 0.02:

                        state["noise_floor"] = (
                            0.98 * state["noise_floor"]
                            + 0.02 * rms
                        )

                # -------------------------------------------------
                # INTERRUPTION RECORDING
                # No 5-second limit here.
                # -------------------------------------------------

                elif state["recording"]:

                    state["buffer"].append(
                        samples.copy()
                    )

                    threshold = max(
                        SPEECH_THRESHOLD,
                        state["noise_floor"] * 3.0,
                    )

                    if rms > threshold:

                        state["last_voice_time"] = now
                        state["speech_active"] = True

                    elif state["speech_active"]:

                        if (
                            now
                            - state["last_voice_time"]
                            >= SILENCE_SECONDS
                        ):

                            state["completed_audio"] = (
                                np.concatenate(
                                    state["buffer"]
                                ),
                                state["sample_rate"],
                            )

                            state["buffer"] = []
                            state["recording"] = False
                            state["speech_active"] = False
                            state["waiting_for_speech"] = True

                            print(
                                "DEBUG: Interruption recording completed"
                            )

            # Do not send microphone audio back to browser.
            return None

        except Exception as exc:

            print(
                "WARNING: audio callback error:",
                exc,
            )

            return None

    return audio_frame_callback
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

st.title("🎤Assamese RAG Voice Assistant")

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

    st.header("🎤 Voice Assistant")

    st.caption(
        "Speak naturally with the assistant."
    )

    # -----------------------------------------------------------------------
    # Conversation state
    # -----------------------------------------------------------------------

    if "voice_conversation_id" not in st.session_state:
        st.session_state["voice_conversation_id"] = None

    if "voice_messages" not in st.session_state:
        st.session_state["voice_messages"] = []

    if "assistant_audio" not in st.session_state:
        st.session_state["assistant_audio"] = None

    if "last_audio_hash" not in st.session_state:
        st.session_state["last_audio_hash"] = None

    # -----------------------------------------------------------------------
    # New conversation
    # -----------------------------------------------------------------------

    if st.button(
        "ðŸ”„ New Conversation",
        key="new_voice_conversation",
    ):
        st.session_state["voice_conversation_id"] = None
        st.session_state["voice_messages"] = []
        st.session_state["assistant_audio"] = None
        st.session_state["last_audio_hash"] = None
        st.rerun()

    # -----------------------------------------------------------------------
    # Always-on microphone
    # -----------------------------------------------------------------------
    if "audio_state" not in st.session_state:
        st.session_state["audio_state"] = create_audio_state()

    audio_state = st.session_state["audio_state"]

    audio_callback = create_audio_callback(
        audio_state
    )
    webrtc_ctx = webrtc_streamer(
    key="voice_microphone",
    desired_playing_state=True,
    media_stream_constraints={
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
        "video": False,
    },
    audio_frame_callback=audio_callback,
    media_toggle_controls=False,
)
   


# -----------------------------------------------------------------------
# Microphone status indicator
# -----------------------------------------------------------------------

tts_until = st.session_state.get(
    "tts_until",
    0.0,
)

assistant_speaking = (
    time.monotonic() < tts_until
)

if assistant_speaking:
    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            margin:8px 0;
            font-size:16px;
            font-weight:600;
        ">
            <span style="
                width:14px;
                height:14px;
                border-radius:50%;
                background:#888;
                display:inline-block;
            "></span>
            Assistant is speaking
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            margin:8px 0;
            font-size:16px;
            font-weight:600;
        ">
            <span style="
                width:14px;
                height:14px;
                border-radius:50%;
                background:red;
                display:inline-block;
                box-shadow:0 0 8px rgba(255,0,0,0.7);
            "></span>
            Listening
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------
# Voice status + transcription + Cancel
# -----------------------------------------------------------------------

if "voice_transcript" not in st.session_state:
    st.session_state["voice_transcript"] = ""

if "voice_processing" not in st.session_state:
    st.session_state["voice_processing"] = False

if st.session_state["voice_processing"]:

    st.info(
        "🎙️ Processing your voice..."
    )

elif st.session_state["voice_transcript"]:

    st.markdown("### Transcription")

    st.info(
        st.session_state["voice_transcript"]
    )

else:

    st.caption(
        "🎙️ Listening for your voice..."
    )

if st.button(
    "Cancel",
    key="cancel_voice",
):

    with audio_state["lock"]:

        audio_state["cancel"] = True
        audio_state["buffer"] = []
        audio_state["recording"] = False
        audio_state["speech_active"] = False
        audio_state["completed_audio"] = None

        audio_state["initial_recording"] = True
        audio_state["waiting_for_speech"] = True

    st.session_state[
        "voice_processing"
    ] = False

    st.session_state[
        "assistant_audio"
    ] = None

    st.session_state[
        "voice_transcript"
    ] = ""

    st.session_state[
        "tts_until"
    ] = 0.0

    st.rerun()

# -----------------------------------------------------------------------
# Live microphone monitor
# -----------------------------------------------------------------------

@st.fragment(run_every="100ms")
def monitor_microphone():

    state = st.session_state["audio_state"]

    interrupt = False
    completed = None

    # ---------------------------------------------------------------
    # Read state from WebRTC callback
    # ---------------------------------------------------------------

    with state["lock"]:

        if state["interrupt"]:

            interrupt = True

            state["interrupt"] = False

        if state["completed_audio"] is not None:

            completed = state["completed_audio"]

            state["completed_audio"] = None

    # ---------------------------------------------------------------
    # User started speaking during TTS
    # ---------------------------------------------------------------

    if interrupt:

        print(
            "DEBUG: User speech detected - "
            "interrupting assistant"
        )

        st.session_state[
            "assistant_audio"
        ] = None

        st.session_state[
            "tts_until"
        ] = 0.0

        st.session_state[
            "voice_processing"
        ] = True

        st.rerun()

    # ---------------------------------------------------------------
    # No completed utterance yet
    # ---------------------------------------------------------------

    if completed is None:
        return

    # ---------------------------------------------------------------
    # Completed user audio
    # ---------------------------------------------------------------

    samples, sample_rate = completed

    print(
        "DEBUG: User utterance captured:",
        len(samples),
        "samples @",
        sample_rate,
        "Hz",
    )

    st.session_state[
        "voice_processing"
    ] = True

    # ---------------------------------------------------------------
    # Convert to mono int16 WAV
    # ---------------------------------------------------------------

    samples = np.clip(
        samples,
        -1.0,
        1.0,
    )

    pcm16 = (
        samples * 32767
    ).astype(np.int16)

    wav_buffer = io.BytesIO()

    with wave.open(
        wav_buffer,
        "wb",
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(
            int(sample_rate)
        )
        wav.writeframes(
            pcm16.tobytes()
        )

    audio_bytes = wav_buffer.getvalue()

    # ---------------------------------------------------------------
    # Noise suppression
    # ---------------------------------------------------------------

    try:

        audio_bytes = denoise_audio(
            audio_bytes
        )

        print(
            "DEBUG: Denoised audio:",
            len(audio_bytes),
            "bytes",
        )

    except Exception as exc:

        print(
            "WARNING: Noise suppression failed:",
            exc,
        )

    # ---------------------------------------------------------------
    # Send to FastAPI
    # ---------------------------------------------------------------

    try:

        files = {
            "file": (
                "recording.wav",
                io.BytesIO(audio_bytes),
                "audio/wav",
            )
        }

        data = {}

        conversation_id = (
            st.session_state.get(
                "voice_conversation_id"
            )
        )

        if conversation_id:

            data["conversation_id"] = (
                conversation_id
            )

        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat/voice",
            headers=get_headers(),
            files=files,
            data=data,
            timeout=180,
        )

        if not response.ok:

            st.session_state[
                "voice_processing"
            ] = False

            st.error(
                f"Voice request failed "
                f"({response.status_code})"
            )

            st.code(
                response.text
            )

            return

        result = response.json()

        # -----------------------------------------------------------
        # Show transcription
        # -----------------------------------------------------------

        transcript = (
            result.get("transcript")
            or ""
        )

        st.session_state[
            "voice_transcript"
        ] = transcript

        print(
            "DEBUG: Transcript:",
            transcript,
        )

        # -----------------------------------------------------------
        # Conversation ID
        # -----------------------------------------------------------

        new_conversation_id = (
            result.get(
                "conversation_id"
            )
        )

        if new_conversation_id:

            st.session_state[
                "voice_conversation_id"
            ] = new_conversation_id

        # -----------------------------------------------------------
        # TTS
        # -----------------------------------------------------------

        audio_base64 = (
            result.get("audio_base64")
        )

        if not audio_base64:

            st.session_state[
                "voice_processing"
            ] = False

            st.error(
                "Backend did not return TTS audio."
            )

            return

        audio_bytes = base64.b64decode(
            audio_base64
        )

        st.session_state[
            "assistant_audio"
        ] = audio_bytes

        # -----------------------------------------------------------
        # Calculate TTS duration
        # -----------------------------------------------------------

        try:

            with wave.open(
                io.BytesIO(audio_bytes),
                "rb",
            ) as wav:

                frames = wav.getnframes()

                sample_rate = (
                    wav.getframerate()
                )

                duration = (
                    frames / sample_rate
                    if sample_rate
                    else 0
                )

            st.session_state[
                "tts_until"
            ] = (
                time.monotonic()
                + duration
            )

        except Exception as exc:

            print(
                "WARNING: Could not calculate "
                f"TTS duration: {exc}"
            )

            st.session_state[
                "tts_until"
            ] = (
                time.monotonic()
                + 10.0
            )

        st.session_state[
            "voice_processing"
        ] = False

        st.rerun()

    except Exception as exc:

        st.session_state[
            "voice_processing"
        ] = False

        st.error(
            f"Voice request failed: {exc}"
        )


monitor_microphone()
    

# -----------------------------------------------------------------------
# TTS PLAYBACK
# -----------------------------------------------------------------------

if st.session_state.get("assistant_audio"):

    audio_b64 = base64.b64encode(
        st.session_state["assistant_audio"]
    ).decode("utf-8")

    st.components.v1.html(
        f"""
        <audio
            id="assistantAudio"
            autoplay
        >
            <source
                src="data:audio/wav;base64,{audio_b64}"
                type="audio/wav"
            >
        </audio>
        """,
        height=0,
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
                            "ðŸ¤– Answer"
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
                                f"ðŸ“š Sources ({len(sources)})"
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
