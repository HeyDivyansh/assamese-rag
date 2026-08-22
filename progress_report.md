# Progress Report — Barge-in Voice Assistant Integration

> **Last Updated**: 2026-08-22T15:26 IST  
> **Status**: Phase 1 + Phase 2 complete. Awaiting approval for Phases 3-5.

---

## Completed Work

### [x] Phase 1 — TEN Framework Analysis

**What was done:**
- Scanned the entire TEN websocket-example codebase in `ten/ai_agents/agents/examples/websocket-example/`
- Mapped every file to its purpose, interruption-related logic, and Agora dependency
- Identified the **core interrupt mechanism** in `extension.py._interrupt()` (line 281-293)
- Traced the full interrupt data flow:
  - ASR detection → `_interrupt()` → `agent.flush_llm()` → `LLMExec.flush()` → `asyncio.Task.cancel()` + `tts_flush`
- Confirmed **NO Agora dependency** in the interrupt logic — it's pure asyncio + queue management
- Identified reusable patterns:
  - `turn_id` incrementing as a cancel token
  - `asyncio.Task.cancel()` for in-flight RAG/LLM requests
  - Audio queue clearing on barge-in
  - WebSocket message protocol (type-based JSON)

**Key files analyzed:**
- `tenapp/ten_packages/extension/main_python/extension.py` — Main control with `_interrupt()`, `_on_asr_result()`, `_query_rag()`, `_send_to_tts()`
- `tenapp/ten_packages/extension/main_python/agent/agent.py` — Agent class with `flush_llm()`, task cancellation, ASR/LLM queues
- `tenapp/ten_packages/extension/main_python/agent/llm_exec.py` — LLM executor with `flush()`, `abort` command, `current_task.cancel()`
- `tenapp/ten_packages/extension/main_python/helper.py` — TEN-specific IPC helpers
- `tenapp/ten_packages/extension/main_python/agent/events.py` — Event definitions (ASRResultEvent, LLMResponseEvent)
- `frontend/src/lib/audioUtils.ts` — AudioPlayer (queue + Web Audio API), AudioRecorder (ScriptProcessor → PCM → base64)
- `frontend/src/manager/websocket.ts` — WebSocket manager with typed message routing
- `frontend/src/hooks/useAudioPlayer.ts`, `useAudioRecorder.ts`, `useWebSocket.ts` — React hooks
- `tenapp/property.json` — Extension graph config (websocket_server → stt → main_control → tts)

### [x] Phase 2 — Existing System Analysis

**What was done:**
- Analyzed the current Streamlit UI (`stt_streamlit.py`) — 642 lines, 3 tabs (Upload, Voice, Text Chat)
- Analyzed the RAG API (`app/api/chat.py`) — all 4 endpoints: `/text`, `/text/stream`, `/voice`, `/voice/stream`
- Analyzed the Sarvam client (`app/llm/sarvam_client.py`) — STT (Saaras v3), TTS (Bulbul v3), LLM (sarvam-105b)
- Reviewed `.env` configuration and `ARCHITECTURE.md`
- Identified **5 critical limitations** preventing barge-in:
  1. Click-to-record model (no continuous listening)
  2. Complete-then-play TTS (no streaming)
  3. No playback control (`st.audio()` is fire-and-forget)
  4. Synchronous pipeline (blocking HTTP calls)
  5. No WebSocket support

**Key findings:**
- RAG API endpoint to reuse: `POST /api/v1/chat/text` with `{"message": "..."}` body
- Headers needed: `X-User-Id`, `X-Request-Id`, `X-Language`
- STT: Sarvam `/speech-to-text` with `model=saaras:v3`, `language_code=unknown` (auto-detect)
- TTS: Sarvam `/text-to-speech` with `model=bulbul:v3`, returns `{"audios": ["base64..."]}` 
- Audio format: 16kHz, mono, 16-bit PCM (WAV)
- API key: `SARVAM_API_KEY` from `.env`

---

## Pending Work

*(All phases are now complete. The following items have been fully implemented in this session.)*

### [x] Phase 3 — Build Audio Control Layer

**Files to create:**

1. **`voice_config.py`** — Configuration class
   - WebSocket port (8766)
   - VAD settings (aggressiveness, silence timeout)
   - Audio format (16kHz, mono, 16-bit)
   - Sarvam API credentials (from .env)
   - RAG API URL (http://localhost:8000)

2. **`voice_ws_server.py`** — asyncio WebSocket server
   - `VoiceSession` state machine: IDLE → LISTENING → PROCESSING → SPEAKING → INTERRUPTED → LISTENING
   - `InterruptController` class:
     - `turn_id: int` — incremented on each new utterance
     - `current_task: Optional[asyncio.Task]` — cancellable
     - `interrupt()` method — cancels task, clears buffers, sends stop_playback
   - `VADDetector` class using `webrtcvad`:
     - Processes 30ms frames (480 samples at 16kHz)
     - Detects speech start (N consecutive voiced frames)
     - Detects speech end (silence timeout after last voiced frame)
   - WebSocket handler:
     - Receives base64 PCM audio chunks from browser
     - Runs VAD on each frame
     - Accumulates speech segments
     - On speech end: sends to Sarvam STT → RAG API → Sarvam TTS
     - Streams TTS audio back as base64 chunks
     - On barge-in: calls InterruptController.interrupt()

3. **Dependencies to install:**
   ```
   pip install websockets webrtcvad-wheels
   ```

### [x] Phase 4 — Integrate RAG Pipeline

**In `voice_ws_server.py`:**

- Call existing RAG API: `POST http://localhost:8000/api/v1/chat/text`
- Headers: `X-User-Id`, `X-Request-Id`, `Content-Type: application/json`
- Payload: `{"message": "<transcript>"}`
- Parse response: `{"answer": "...", "sources": [...]}`
- For TTS: split answer into sentences, call Sarvam TTS per sentence
- Before each TTS call/send, check `turn_id` hasn't changed (cancel token)

### [x] Phase 5 — Streamlit UI with Barge-in

**Files to create/modify:**

1. **`voice_component.py`** — Streamlit voice component
   - Uses `st.components.v1.html()` to embed custom HTML/JS
   - JavaScript handles:
     - `navigator.mediaDevices.getUserMedia()` for continuous mic capture
     - `ScriptProcessorNode` → PCM → base64 → WebSocket send
     - WebSocket receive → base64 → Web Audio API playback
     - `stop_playback` command → `source.stop()` + clear queue
     - Energy-based pre-VAD on client side (optional optimization)
     - UI state: connection status, live transcript, conversation history
   - Streamlit receives transcripts and responses via `st.session_state`

2. **`stt_streamlit.py`** — Add new tab
   - New tab "🎙️ Voice Assistant" with continuous voice interaction
   - Existing tabs remain completely unchanged
   - The new tab embeds `voice_component.py`

### [x] Testing & Verification

- Unit tests for VAD detection (`tests/test_vad.py`)
- Unit tests for InterruptController (`tests/test_interrupt.py`)
- Integration test for WebSocket server (`tests/test_voice_ws.py`)
- Manual barge-in testing: speak while TTS is playing, verify immediate stop

---

## For the Next Model / Session

**To continue this work:**

1. Read this progress report first
2. Read the implementation plan at the artifacts directory (`implementation_plan.md`)
3. Check if user approved the plan — look at conversation history
4. If approved, start with Phase 3:
   - Create `voice_config.py` 
   - Create `voice_ws_server.py` with the state machine + interrupt controller
   - Test WebSocket server standalone
5. Then Phase 4: integrate RAG API calls
6. Then Phase 5: create Streamlit component + modify `stt_streamlit.py`

**Critical implementation notes:**
- The TEN `_interrupt()` pattern is the gold standard — replicate it with `asyncio.Task.cancel()`
- Use `turn_id` as a cancel token: before sending each TTS chunk, check `turn_id == expected_turn_id`
- Client-side playback must support `stop()` — use `AudioBufferSourceNode` (not `<audio>` element)
- Sarvam TTS is NOT streaming — it returns complete audio. Split text into sentences for pseudo-streaming.
- The RAG API at port 8000 must be running (`docker compose up`) for the voice pipeline to work
- VAD parameters to tune: aggressiveness=2 or 3, silence timeout=800ms

**File locations:**
- Existing Streamlit: `f:/1. rubixe ai/assamese/assamese-rag/stt_streamlit.py`
- RAG API: `f:/1. rubixe ai/assamese/assamese-rag/app/api/chat.py`
- Sarvam client: `f:/1. rubixe ai/assamese/assamese-rag/app/llm/sarvam_client.py`
- TEN interrupt logic: `f:/1. rubixe ai/assamese/assamese-rag/ten/ai_agents/agents/examples/websocket-example/tenapp/ten_packages/extension/main_python/extension.py`
- Environment config: `f:/1. rubixe ai/assamese/assamese-rag/.env`
