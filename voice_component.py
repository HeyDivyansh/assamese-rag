import streamlit.components.v1 as components

def voice_component(ws_port=8766):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 10px; }}
            .container {{ display: flex; flex-direction: column; gap: 15px; }}
            .btn {{ padding: 12px 20px; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; transition: background 0.3s; }}
            .btn-start {{ background-color: #ff4b4b; color: white; }}
            .btn-start:hover {{ background-color: #ff3333; }}
            .btn-stop {{ background-color: #6c757d; color: white; }}
            .btn-stop:hover {{ background-color: #5a6268; }}
            .status {{ font-size: 14px; color: #555; display: flex; align-items: center; gap: 5px; }}
            .status-dot {{ width: 10px; height: 10px; border-radius: 50%; background-color: #dc3545; display: inline-block; }}
            .status-dot.active {{ background-color: #28a745; animation: pulse 1.5s infinite; }}
            @keyframes pulse {{
                0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }}
                70% {{ transform: scale(1); box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }}
                100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }}
            }}
            .transcript, .assistant {{ 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                min-height: 60px; 
                border: 1px solid #e9ecef;
                font-size: 15px;
                line-height: 1.5;
            }}
            .assistant {{ background: #e8f0fe; border-color: #d2e3fc; }}
            .title {{ font-size: 14px; font-weight: 600; margin-bottom: 5px; color: #495057; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; align-items: center; gap: 15px;">
                <button id="toggleBtn" class="btn btn-start">🎙️ Start Voice Assistant</button>
                <div class="status">
                    <span id="statusDot" class="status-dot"></span>
                    <span id="statusText">Disconnected</span>
                </div>
            </div>
            
            <div>
                <div class="title">You:</div>
                <div id="transcript" class="transcript">...</div>
            </div>
            
            <div>
                <div class="title">Assistant:</div>
                <div id="assistant" class="assistant">...</div>
            </div>
        </div>

        <script>
            let ws = null;
            let audioContext = null;
            let mediaStream = null;
            let scriptProcessor = null;
            let audioQueue = [];
            let isPlaying = false;
            let currentSource = null;
            let isRecording = false;
            
            const toggleBtn = document.getElementById('toggleBtn');
            const statusText = document.getElementById('statusText');
            const statusDot = document.getElementById('statusDot');
            const transcriptEl = document.getElementById('transcript');
            const assistantEl = document.getElementById('assistant');

            function float32ToInt16(buffer) {{
                let l = buffer.length;
                let buf = new Int16Array(l);
                while (l--) {{
                    let s = Math.max(-1, Math.min(1, buffer[l]));
                    buf[l] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }}
                return buf.buffer;
            }}

            async function startRecording() {{
                try {{
                    mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: {{
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }} }});
                    
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    audioContext = new AudioContext({{ sampleRate: 16000 }});
                    const source = audioContext.createMediaStreamSource(mediaStream);
                    
                    // Use ScriptProcessor for legacy support or simpler PCM extraction
                    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                    
                    scriptProcessor.onaudioprocess = (e) => {{
                        if (ws && ws.readyState === WebSocket.OPEN) {{
                            const inputData = e.inputBuffer.getChannelData(0);
                            const pcmData = float32ToInt16(inputData);
                            ws.send(pcmData); // Send binary PCM chunk
                        }}
                    }};
                    
                    source.connect(scriptProcessor);
                    scriptProcessor.connect(audioContext.destination);
                    
                    isRecording = true;
                    toggleBtn.innerHTML = "⏹️ Stop Voice Assistant";
                    toggleBtn.className = "btn btn-stop";
                    connectWebSocket();
                }} catch (e) {{
                    console.error("Microphone access denied:", e);
                    alert("Microphone access required. Please allow microphone permissions.");
                }}
            }}

            function stopRecording() {{
                if (scriptProcessor) {{
                    scriptProcessor.disconnect();
                    scriptProcessor = null;
                }}
                if (mediaStream) {{
                    mediaStream.getTracks().forEach(t => t.stop());
                    mediaStream = null;
                }}
                if (ws) {{
                    ws.close();
                    ws = null;
                }}
                if (audioContext && audioContext.state !== 'closed') {{
                    audioContext.close();
                }}
                isRecording = false;
                toggleBtn.innerHTML = "🎙️ Start Voice Assistant";
                toggleBtn.className = "btn btn-start";
                statusText.textContent = "Disconnected";
                statusDot.className = "status-dot";
                stopPlayback();
            }}

            function connectWebSocket() {{
                ws = new WebSocket(`ws://localhost:{ws_port}`);
                ws.binaryType = 'arraybuffer';
                
                ws.onopen = () => {{
                    statusText.textContent = "Connected. Speak now.";
                    statusDot.className = "status-dot active";
                    ws.send(JSON.stringify({{ type: "cmd", name: "start" }}));
                }};
                
                ws.onmessage = async (event) => {{
                    if (typeof event.data === 'string') {{
                        const msg = JSON.parse(event.data);
                        if (msg.type === "state") {{
                            statusText.textContent = "State: " + msg.state;
                        }} else if (msg.type === "transcript") {{
                            transcriptEl.textContent = msg.text;
                            assistantEl.textContent = "..."; // Clear on new speech
                        }} else if (msg.type === "answer") {{
                            assistantEl.textContent = msg.text;
                        }} else if (msg.type === "audio") {{
                            playBase64Wav(msg.audio);
                        }} else if (msg.type === "cmd" && msg.name === "stop_playback") {{
                            stopPlayback();
                        }}
                    }}
                }};
                
                ws.onclose = () => {{
                    if (isRecording) {{
                        statusText.textContent = "Reconnecting...";
                        statusDot.className = "status-dot";
                        setTimeout(connectWebSocket, 2000);
                    }}
                }};
            }}

            function playBase64Wav(base64) {{
                const binaryString = window.atob(base64);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                audioQueue.push(bytes.buffer);
                processAudioQueue();
            }}

            async function processAudioQueue() {{
                if (isPlaying || audioQueue.length === 0 || !audioContext) return;
                isPlaying = true;
                
                const arrayBuffer = audioQueue.shift();
                try {{
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    currentSource = audioContext.createBufferSource();
                    currentSource.buffer = audioBuffer;
                    currentSource.connect(audioContext.destination);
                    currentSource.onended = () => {{
                        isPlaying = false;
                        currentSource = null;
                        processAudioQueue();
                    }};
                    currentSource.start();
                }} catch (e) {{
                    console.error("Error decoding audio:", e);
                    isPlaying = false;
                    processAudioQueue();
                }}
            }}

            function stopPlayback() {{
                audioQueue = [];
                if (currentSource) {{
                    currentSource.stop();
                    currentSource = null;
                }}
                isPlaying = false;
            }}

            toggleBtn.addEventListener('click', () => {{
                if (isRecording) {{
                    stopRecording();
                }} else {{
                    startRecording();
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=500)
