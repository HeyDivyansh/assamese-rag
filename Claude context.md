# Claude Context

## What Has Been Done
1. **Removed Agora Dependency**: Patched the Go server (`main.go` and `http_server.go`) so it no longer strictly requires a 32-character `AGORA_APP_ID`. It will now start in "WebSocket-only mode".
2. **Fixed STT to RAG Integration**: Updated `main_python` extension (`extension.py`) to only call the Assamese RAG API on *final* ASR transcriptions. Partial transcripts are now only used for barge-in (interruption) and live UI updates, preventing excessive API calls.
3. **Fixed RAG API URL**: Changed the RAG API URL from `host.docker.internal` to `localhost:8000` to work natively on Windows, and made it configurable via `property.json`.
4. **Cleaned up `property.json`**: Removed the `weatherapi_tool_python` (which required paid keys) and configured a clean STT -> RAG -> TTS pipeline.
5. **Fixed Frontend WebSocket Port**: Modified the Next.js frontend to use a fixed WebSocket port (`8765`) matching `property.json`, rather than generating a random dynamic port, to ensure stable connections. Updated UI titles to reflect the "Assamese Voice AI Assistant".

## What Should Be Done Next
1. **Clean Up Unrequired Files**: Remove unused examples and files from the `ten` directory to keep the project clean (as requested).
2. **Build the Go Server**: Compile the Go API server in `ten/ai_agents/server` (`go build -o api.exe`).
3. **Start the Go API Server**: Run the Go server pointing to our `tenapp` directory.
4. **Start the Frontend**: Install dependencies and run the Next.js frontend (`npm run dev`).
5. **Start the RAG API**: Ensure the FastAPI backend is running on port `8000`.
6. **End-to-End Testing**: Open the frontend, start the agent, and test the full voice pipeline (Microphone -> WebSocket -> TEN -> Sarvam STT -> FastAPI RAG -> Sarvam TTS -> WebSocket -> Speaker).

## Interruptions
- No interruptions in the current workflow.
