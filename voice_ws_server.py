import asyncio
import json
import logging
import wave
import io
import uuid
import base64
import re
import websockets
import aiohttp
import numpy as np
from ten_vad import TenVad
from typing import Optional

from voice_config import VoiceConfig
from app.llm.sarvam_client import text_to_speech, transcribe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("voice_ws_server")

# VAD class
class VADDetector:
    def __init__(self):
        self.sample_rate = VoiceConfig.SAMPLE_RATE
        self.hop_size = self.sample_rate * VoiceConfig.VAD_HOP_SIZE_MS // 1000
        self.bytes_per_hop = self.hop_size * VoiceConfig.SAMPLE_WIDTH
        self.prefix_window_size = (
            VoiceConfig.VAD_PREFIX_PADDING_MS // VoiceConfig.VAD_HOP_SIZE_MS
        )
        self.silence_window_size = (
            VoiceConfig.VAD_SILENCE_DURATION_MS // VoiceConfig.VAD_HOP_SIZE_MS
        )
        self.window_size = max(self.prefix_window_size, self.silence_window_size)
        self.vad = TenVad(self.hop_size)
        self.audio_buffer = bytearray()
        self.probe_window: list[float] = []
        self.recent_audio: list[bytes] = []
        self.speech_buffer = bytearray()
        self.is_speaking = False

    def process_chunk(self, chunk: bytes):
        """Apply TEN's probability-window VAD and return completed utterances."""
        self.audio_buffer.extend(chunk)
        segments = []

        while len(self.audio_buffer) >= self.bytes_per_hop:
            audio_hop = bytes(self.audio_buffer[:self.bytes_per_hop])
            del self.audio_buffer[:self.bytes_per_hop]
            probe, _flag = self.vad.process(
                np.frombuffer(audio_hop, dtype=np.int16)
            )
            self.probe_window.append(probe)
            if len(self.probe_window) > self.window_size:
                self.probe_window.pop(0)

            self.recent_audio.append(audio_hop)
            if len(self.recent_audio) > self.prefix_window_size:
                self.recent_audio.pop(0)

            if not self.is_speaking:
                if len(self.probe_window) == self.window_size:
                    prefix_probes = self.probe_window[-self.prefix_window_size:]
                    if all(probe >= VoiceConfig.VAD_THRESHOLD for probe in prefix_probes):
                        self.is_speaking = True
                        self.speech_buffer.extend(b"".join(self.recent_audio))
            elif len(self.probe_window) == self.window_size:
                self.speech_buffer.extend(audio_hop)
                silence_probes = self.probe_window[-self.silence_window_size:]
                if all(probe < VoiceConfig.VAD_THRESHOLD for probe in silence_probes):
                    self.is_speaking = False
                    if len(self.speech_buffer) >= self.sample_rate * VoiceConfig.SAMPLE_WIDTH // 2:
                        segments.append(bytes(self.speech_buffer))
                    self.speech_buffer.clear()

        return segments

    @property
    def has_confirmed_barge_in(self):
        return self.is_speaking

def pcm_to_wav(pcm_data: bytes) -> bytes:
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(VoiceConfig.CHANNELS)
            wav_file.setsampwidth(VoiceConfig.SAMPLE_WIDTH)
            wav_file.setframerate(VoiceConfig.SAMPLE_RATE)
            wav_file.writeframes(pcm_data)
        return wav_io.getvalue()
        
def split_into_sentences(text: str) -> list[str]:
    # Simple split on punctuations
    sentences = re.split(r'(?<=[.!?।।])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def clean_answer(text: str) -> str:
    """Remove source markers before displaying or speaking the RAG answer."""
    if not text:
        return ""

    text = re.sub(
        r"\[\s*s\d+(?:\s*,\s*s\d+)*\s*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<!\w)s\d+(?!\w)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()

class InterruptController:
    def __init__(self):
        self.turn_id = 0
        self.current_task: Optional[asyncio.Task] = None
        
    def new_turn(self):
        self.turn_id += 1
        self.cancel_current()
        return self.turn_id
        
    def cancel_current(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            self.current_task = None

class VoiceSession:
    def __init__(self, websocket):
        self.ws = websocket
        self.vad = VADDetector()
        self.interrupt_ctrl = InterruptController()
        self.state = "IDLE" # IDLE, LISTENING, PROCESSING, SPEAKING
        self.user_id = str(uuid.uuid4())
        self.request_id = str(uuid.uuid4())

    async def send_state(self):
        await self.ws.send(json.dumps({"type": "state", "state": self.state}))

    async def handle_audio(self, audio_data: bytes):
        if self.state == "IDLE":
            self.state = "LISTENING"
            await self.send_state()

        speech_segments = self.vad.process_chunk(audio_data)

        # Only interrupt active TTS, never the initial STT/RAG request.
        if self.state == "SPEAKING" and self.vad.has_confirmed_barge_in:
            logger.info("Barge-in detected!")
            self.interrupt_ctrl.new_turn()
            self.state = "LISTENING"
            await self.ws.send(json.dumps({"type": "cmd", "name": "stop_playback"}))
            await self.send_state()

        for segment in speech_segments:
            # Do not let background audio or a second utterance cancel STT/RAG.
            if self.state != "LISTENING":
                continue

            # We got a complete utterance
            turn_id = self.interrupt_ctrl.new_turn()
            self.state = "PROCESSING"
            await self.send_state()
            
            # Start processing pipeline as a cancellable task
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.process_pipeline(segment, turn_id))
            self.interrupt_ctrl.current_task = task
            
    async def process_pipeline(self, pcm_data: bytes, expected_turn_id: int):
        try:
            logger.info(f"Turn {expected_turn_id}: Processing {len(pcm_data)} bytes of audio")
            # 1. STT
            wav_data = pcm_to_wav(pcm_data)
            stt_res = await transcribe(wav_data, request_id=self.request_id)
            transcript = stt_res.get("transcript", "").strip()
            
            if not transcript:
                self.state = "LISTENING"
                await self.send_state()
                return
                
            logger.info(f"Turn {expected_turn_id}: Transcript: {transcript}")
            await self.ws.send(json.dumps({"type": "transcript", "text": transcript}))
            
            # 2. RAG API
            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-User-Id": self.user_id,
                    "X-Request-Id": self.request_id,
                    "Content-Type": "application/json"
                }
                async with session.post(VoiceConfig.RAG_API_URL, json={"message": transcript}, headers=headers) as resp:
                    resp.raise_for_status()
                    rag_data = await resp.json()
                    
            answer = clean_answer(rag_data.get("answer", ""))
            logger.info(f"Turn {expected_turn_id}: RAG Answer: {answer}")
            await self.ws.send(json.dumps({"type": "answer", "text": answer}))
            
            # 3. TTS (Sentence by sentence)
            sentences = split_into_sentences(answer)
            self.state = "SPEAKING"
            await self.send_state()
            
            lang_code = stt_res.get("language_code", "en-IN")
            if lang_code not in ["en-IN", "kn-IN"]:
                lang_code = "en-IN"
                
            for sentence in sentences:
                if self.interrupt_ctrl.turn_id != expected_turn_id:
                    logger.info(f"Turn {expected_turn_id}: Interrupted before TTS")
                    break
                    
                logger.info(f"Turn {expected_turn_id}: TTS for: {sentence}")
                tts_wav = await text_to_speech(sentence, lang_code)
                
                # Check turn ID again before sending
                if self.interrupt_ctrl.turn_id != expected_turn_id:
                    logger.info(f"Turn {expected_turn_id}: Interrupted before sending audio")
                    break
                    
                audio_b64 = base64.b64encode(tts_wav).decode('utf-8')
                await self.ws.send(json.dumps({
                    "type": "audio",
                    "audio": audio_b64
                }))
                
            if self.interrupt_ctrl.turn_id == expected_turn_id:
                self.state = "LISTENING"
                await self.send_state()
                
        except asyncio.CancelledError:
            logger.info(f"Turn {expected_turn_id}: Pipeline cancelled via barge-in")
        except Exception as e:
            logger.error(f"Turn {expected_turn_id}: Pipeline error: {e}", exc_info=True)
            if self.interrupt_ctrl.turn_id == expected_turn_id:
                self.state = "LISTENING"
                await self.send_state()

async def ws_handler(websocket):
    logger.info("New WebSocket connection")
    session = VoiceSession(websocket)
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary audio data
                await session.handle_audio(message)
            elif isinstance(message, str):
                # Text commands
                try:
                    data = json.loads(message)
                    if data.get("type") == "cmd" and data.get("name") == "start":
                        session.state = "LISTENING"
                        await session.send_state()
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        session.interrupt_ctrl.cancel_current()

async def main():
    logger.info(f"Voice WebSocket server listening on ws://{VoiceConfig.WS_HOST}:{VoiceConfig.WS_PORT}")
    async with websockets.serve(ws_handler, VoiceConfig.WS_HOST, VoiceConfig.WS_PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
