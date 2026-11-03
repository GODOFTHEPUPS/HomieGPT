import os
import io
import re
import json
import time
import asyncio
import tempfile
import urllib.request
import threading
import subprocess

import discord
import edge_tts
import numpy as np
import sherpa_onnx

from discord import app_commands
from discord.ext import voice_recv
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")

OLLAMA_URL = "http://localhost:11434/api/chat"
AI_MODEL = "qwen3:8b"

VOICE = "en-US-AndrewNeural"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHERPA_MODEL_DIR = os.path.join(
    BASE_DIR,
    "sherpa-onnx-streaming-zipformer-en-2023-06-26"
)

SAMPLE_RATE = 16000
DISCORD_SAMPLE_RATE = 48000

# Sherpa-ONNX streaming English Zipformer (INT8 encoder/joiner).
SHERPA_TOKENS = os.path.join(SHERPA_MODEL_DIR, "tokens.txt")
SHERPA_ENCODER = os.path.join(
    SHERPA_MODEL_DIR,
    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
)
SHERPA_DECODER = os.path.join(
    SHERPA_MODEL_DIR,
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
)
SHERPA_JOINER = os.path.join(
    SHERPA_MODEL_DIR,
    "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
)

for model_file in (
    SHERPA_TOKENS,
    SHERPA_ENCODER,
    SHERPA_DECODER,
    SHERPA_JOINER,
):
    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Sherpa-ONNX model file is missing: {model_file}\n"
            "Download and extract the English streaming Zipformer model "
            "into the HomieGPT folder."
        )

print("Loading HomieGPT Sherpa-ONNX streaming listening model...")
speech_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=SHERPA_TOKENS,
    encoder=SHERPA_ENCODER,
    decoder=SHERPA_DECODER,
    joiner=SHERPA_JOINER,
    num_threads=2,
    sample_rate=SAMPLE_RATE,
    provider="cpu",
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=0.8,
    rule2_min_trailing_silence=0.45,
    rule3_min_utterance_length=20.0,
    decoding_method="greedy_search",
)
print("Sherpa-ONNX listening model ready!")



SYSTEM_PROMPT = (
    "You are HomieGPT, Victor's personal AI voice assistant. ""The user's name is Victor. If asked his name, answer Victor confidently. ""Never say you are unsure of his name. "
    "Always address the user as Victor when using his name. "
    "Never call the user Homie, bro, boss, buddy, or similar nicknames. "
    "Your own name is HomieGPT. "
    "Be friendly, intelligent, confident, relaxed, and conversational. "
    "Speak naturally, like a real person having a conversation. "
    "Avoid forced slang, repetitive catchphrases, and unnecessary greetings. "
    "Do not sound robotic or repeatedly introduce yourself. "
    "Never begin every response with the same phrase. "
    "Use the recent conversation history to understand context. "
    "Answer naturally and directly. Keep spoken replies concise, "
    "usually 1-3 sentences unless Victor asks for more detail. "
    "Do not interrupt or ignore what Victor is telling you. "
    "Respect Victor's preferences and instructions. "
    "Be honest and never invent information. "
    "Ask a short clarifying question when necessary."
)



def ask_ollama(messages):
    data = {
        "model": AI_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 80,
            "num_ctx": 2048,
            "temperature": 0.6
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["message"]["content"]


class StreamingFFmpegSource(discord.AudioSource):
    """AudioSource that feeds streamed MP3 bytes to FFmpeg stdin."""

    def __init__(self):
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "mp3", "-i", "pipe:0",
                "-f", "s16le", "-ar", "48000", "-ac", "2", "pipe:1"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        self._cleaned = False
        self._lock = threading.Lock()

    def is_opus(self):
        return False

    def read(self):
        if self.process.stdout is None:
            return b""
        return self.process.stdout.read(3840)

    def write_audio(self, audio_bytes):
        if self.process.stdin is None or self.process.stdin.closed:
            raise RuntimeError("FFmpeg input pipe is closed.")
        self.process.stdin.write(audio_bytes)
        self.process.stdin.flush()

    def finish_input(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()

    def cleanup(self):
        with self._lock:
            if self._cleaned:
                return
            self._cleaned = True
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    if stream and not stream.closed:
                        stream.close()
                except OSError:
                    pass
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()


async def speak_streaming(voice_client, text):
    """
    Stable buffered TTS playback.

    We intentionally collect the complete MP3 before playback rather than
    piping live MP3 fragments into FFmpeg. This avoids partial PCM frames and
    pipe-lifetime races that can cause crackling or Opus encoder failures.
    """
    communicate = edge_tts.Communicate(
        text[:600],
        VOICE,
        rate="+3%",
        pitch="-2Hz",
        volume="+0%"
    )

    audio_parts = []
    audio_chunks = 0

    tts_started = time.monotonic()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            audio_parts.append(chunk["data"])
            audio_chunks += 1

    mp3_audio = b"".join(audio_parts)
    if not mp3_audio:
        raise RuntimeError("Edge TTS returned no audio data.")

    print(
        f"Generated {audio_chunks} Edge TTS chunks "
        f"in {time.monotonic() - tts_started:.1f}s; "
        f"buffered {len(mp3_audio)} bytes."
    )

    # FFmpeg reads the complete MP3 from a file-like stdin source and emits
    # Discord-compatible 48 kHz, stereo, signed 16-bit PCM frames.
    source = discord.FFmpegPCMAudio(
        io.BytesIO(mp3_audio),
        pipe=True,
        options="-vn"
    )

    loop = asyncio.get_running_loop()
    playback_done = asyncio.Event()
    playback_error = {"error": None}

    def after_playback(error):
        playback_error["error"] = error
        loop.call_soon_threadsafe(playback_done.set)

    try:
        voice_client.play(source, after=after_playback)
        await playback_done.wait()
        if playback_error["error"] is not None:
            raise RuntimeError(
                f"Discord audio playback failed: {playback_error['error']}"
            )
    except Exception:
        if voice_client.is_playing():
            voice_client.stop()
        source.cleanup()
        raise


class HomieVoiceSink(voice_recv.AudioSink):
    """Discord audio sink feeding per-user streams to Sherpa-ONNX."""

    def __init__(self, bot, loop, guild_id):
        super().__init__()
        self.bot = bot
        self.loop = loop
        self.guild_id = guild_id
        self.streams = {}
        self.lock = threading.Lock()

    def wants_opus(self):
        return False

    def write(self, user, data):
        if user is None or getattr(user, "bot", False):
            return
        if data.pcm is None:
            return

        try:
            # Discord supplies 48 kHz, stereo, signed 16-bit PCM.
            # Sherpa accepts float32 samples and resamples from 48 kHz itself.
            pcm = np.frombuffer(data.pcm, dtype=np.int16)
            if pcm.size < 2:
                return

            stereo = pcm[: (pcm.size // 2) * 2].reshape(-1, 2)
            mono_float = stereo.astype(np.float32).mean(axis=1) / 32768.0

            transcript_to_send = None
            with self.lock:
                user_id = user.id
                if user_id not in self.streams:
                    self.streams[user_id] = {
                        "stream": speech_recognizer.create_stream(),
                        "last_partial": "",
                        "last_partial_log": 0.0,
                    }

                entry = self.streams[user_id]
                stream = entry["stream"]
                stream.accept_waveform(DISCORD_SAMPLE_RATE, mono_float)

                while speech_recognizer.is_ready(stream):
                    speech_recognizer.decode_stream(stream)

                result = speech_recognizer.get_result_all(stream)
                partial_text = (result.text or "").strip()

                # Show occasional live partials so we can see what Homie hears.
                now = time.monotonic()
                if (
                    partial_text
                    and partial_text != entry["last_partial"]
                    and now - entry["last_partial_log"] >= 0.6
                ):
                    print(f"Listening (partial): {partial_text}")
                    entry["last_partial"] = partial_text
                    entry["last_partial_log"] = now

                if speech_recognizer.is_endpoint(stream):
                    final_text = partial_text
                    speech_recognizer.reset(stream)
                    entry["last_partial"] = ""
                    entry["last_partial_log"] = 0.0

                    if final_text:
                        transcript_to_send = (user_id, final_text)

            # Schedule Discord coroutine outside the recognizer lock.
            if transcript_to_send:
                user_id, transcript = transcript_to_send
                asyncio.run_coroutine_threadsafe(
                    self.bot.handle_voice_text(
                        self.guild_id,
                        user_id,
                        transcript
                    ),
                    self.loop
                )

        except Exception as error:
            print("Voice recognition error:", error)

    def cleanup(self):
        with self.lock:
            self.streams.clear()


def normalize_recognized_text(text):
    """
    Apply only a confirmed personal-name correction.
    This does not alter Sherpa's acoustic model or general vocabulary.
    """
    # Sherpa transcribed Victor's surname as "VASQUITS" in the supplied test.
    text = re.sub(
        r"\b(?:VASQUITS|VASQUIZ)\b",
        "VASQUEZ",
        text,
        flags=re.IGNORECASE
    )
    return text.strip()


class HomieGPT(discord.Client):

    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)
        # Conversation state is isolated per Discord user.
        self.conversations = {}
        self.history_message_limit = 12  # Six user/assistant exchanges.

        # Queue recognized speech so sentences aren't discarded while Homie
        # is thinking or speaking. One worker processes each user's queue.
        self.voice_queues = {}
        self.voice_workers = {}
        self.last_queued_speech = {}

    async def setup_hook(self):
        await self.tree.sync()

    async def on_ready(self):
        print(f"HomieGPT is online as {self.user}!")
        print("Hands-free Sherpa-ONNX voice mode is ready. Join with /join.")

    async def handle_voice_text(self, guild_id, user_id, transcript):
        question = normalize_recognized_text(transcript)
        if not question:
            return

        # Sherpa can occasionally emit the same final transcript twice.
        # Suppress only exact repeats arriving within a short time window;
        # similar but different sentences are preserved.
        now = time.monotonic()
        previous = self.last_queued_speech.get(user_id)
        if previous:
            previous_text, previous_time = previous
            if question.casefold() == previous_text.casefold() and now - previous_time < 1.5:
                print(f"Ignored duplicate final transcript for user {user_id}.")
                return
        self.last_queued_speech[user_id] = (question, now)

        # A larger bounded queue absorbs normal speech while Homie is
        # generating or playing a response without allowing unlimited growth.
        queue = self.voice_queues.setdefault(
            user_id, asyncio.Queue(maxsize=25)
        )

        if queue.full():
            print(
                f"Voice queue is full for user {user_id}; "
                "new sentence was not queued. Pause briefly so Homie can catch up."
            )
            return

        await queue.put((guild_id, question))
        print(
            f"Queued speech for user {user_id} "
            f"({queue.qsize()} pending)."
        )

        worker = self.voice_workers.get(user_id)
        if worker is None or worker.done():
            self.voice_workers[user_id] = asyncio.create_task(
                self.process_voice_queue(user_id),
                name=f"homie-voice-queue-{user_id}"
            )

    async def process_voice_queue(self, user_id):
        queue = self.voice_queues[user_id]

        try:
            while not queue.empty():
                guild_id, question = await queue.get()

                try:
                    print(f"Recognized speech: {question}")

                    conversation = self.conversations.setdefault(
                        user_id, {"messages": []}
                    )

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT}
                    ]
                    messages.extend(conversation["messages"])
                    messages.append({"role": "user", "content": question})

                    print("HomieGPT is thinking...")
                    ai_started = time.monotonic()
                    answer = await asyncio.to_thread(
                        ask_ollama, messages
                    )
                    print(
                        "AI response generated in "
                        f"{time.monotonic() - ai_started:.1f}s"
                    )

                    conversation["messages"].extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ])
                    conversation["messages"] = conversation["messages"][
                        -self.history_message_limit:
                    ]

                    print("HomieGPT:", answer)

                    guild = self.get_guild(guild_id)
                    voice_client = guild.voice_client if guild else None

                    if not voice_client or not voice_client.is_connected():
                        print("HomieGPT has no active voice connection.")
                        continue

                    # Wait asynchronously for any other audio to finish.
                    while voice_client.is_connected() and voice_client.is_playing():
                        await asyncio.sleep(0.1)

                    if not voice_client.is_connected():
                        print("Voice connection ended before playback.")
                        continue

                    tts_started = time.monotonic()
                    await speak_streaming(voice_client, answer)
                    print(
                        "Speech generated and played in "
                        f"{time.monotonic() - tts_started:.1f}s"
                    )

                except Exception as error:
                    print("HomieGPT conversation error:", error)

                finally:
                    queue.task_done()

        finally:
            current = asyncio.current_task()
            if self.voice_workers.get(user_id) is current:
                self.voice_workers.pop(user_id, None)

            # If speech arrived as this worker was finishing, start another.
            if not queue.empty():
                self.voice_workers[user_id] = asyncio.create_task(
                    self.process_voice_queue(user_id),
                    name=f"homie-voice-queue-{user_id}"
                )


bot = HomieGPT()


@bot.tree.command(
    name="hello",
    description="Say hello to HomieGPT"
)
async def hello(interaction: discord.Interaction):

    await interaction.response.send_message(
        "HomieGPT is online and ready."
    )


@bot.tree.command(
    name="ask",
    description="Ask HomieGPT a question"
)
@app_commands.describe(question="What do you want to ask?")
async def ask(interaction: discord.Interaction, question: str):

    await interaction.response.defer(thinking=True)

    try:
        answer = await asyncio.to_thread(
            ask_ollama, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        )

        await interaction.followup.send(answer[:2000])

    except Exception as error:
        print("Ollama error:", error)

        await interaction.followup.send(
            "I couldn't reach my local AI."
        )


@bot.tree.command(
    name="join",
    description="Join voice for hands-free conversation"
)
async def join(interaction: discord.Interaction):

    if not interaction.user.voice:
        await interaction.response.send_message(
            "Join a voice channel first!",
            ephemeral=True
        )
        return

    channel = interaction.user.voice.channel

    await interaction.response.defer()

    try:
        voice_client = interaction.guild.voice_client

        if voice_client and not isinstance(
            voice_client, voice_recv.VoiceRecvClient
        ):
            await voice_client.disconnect()
            voice_client = None

        if voice_client:
            await voice_client.move_to(channel)
        else:
            voice_client = await channel.connect(
                cls=voice_recv.VoiceRecvClient
            )

        if not voice_client.is_listening():
            voice_client.listen(
                HomieVoiceSink(
                    bot,
                    asyncio.get_running_loop(),
                    interaction.guild_id
                )
            )

        await interaction.followup.send(
            "I'm listening! Just speak naturally. 🎙️"
        )

    except Exception as error:
        print("Voice connection error:", error)

        await interaction.followup.send(
            "I couldn't connect or start listening. "
            "Check the CMD window."
        )


@bot.tree.command(
    name="speak",
    description="Make HomieGPT speak"
)
@app_commands.describe(message="What should HomieGPT say?")
async def speak(interaction: discord.Interaction, message: str):

    await interaction.response.defer(thinking=True)

    voice_client = interaction.guild.voice_client

    if not voice_client or not voice_client.is_connected():
        await interaction.followup.send(
            "Use /join first!"
        )
        return

    try:
        if voice_client.is_playing():
            voice_client.stop()

        await speak_streaming(voice_client, message)
        await interaction.followup.send("Speaking now! 🎙️")

    except Exception as error:
        print("Speech error:", error)
        await interaction.followup.send(
            "I couldn't generate speech. Check the CMD window."
        )


@bot.tree.command(
    name="leave",
    description="Make HomieGPT leave voice"
)
async def leave(interaction: discord.Interaction):

    voice_client = interaction.guild.voice_client

    if voice_client:
        await voice_client.disconnect()

        await interaction.response.send_message(
            "I'll be here when you need me."
        )
    else:
        await interaction.response.send_message(
            "I'm not in a voice channel!"
        )


if not TOKEN:
    raise RuntimeError(
        "DISCORD_TOKEN is missing from your .env file."
    )


bot.run(TOKEN)
