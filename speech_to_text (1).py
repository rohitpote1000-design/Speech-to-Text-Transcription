"""
speech_to_text.py
-----------------
Transcribe audio files or live microphone input using the SpeechRecognition library.

Install dependencies:
    pip install SpeechRecognition pyaudio pydub

For .mp3 / .ogg support:
    pip install pydub
    # Also install ffmpeg: https://ffmpeg.org/download.html
"""

import os
import sys
import time
import wave
import argparse
import tempfile
import speech_recognition as sr

# ── optional pydub for mp3/ogg conversion ──────────────────────────────────
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def convert_to_wav(filepath: str) -> str:
    """Convert mp3/ogg/flac → temporary WAV file. Returns path to WAV."""
    if not PYDUB_AVAILABLE:
        raise RuntimeError(
            "pydub is required to convert non-WAV files.\n"
            "Install it with:  pip install pydub\n"
            "Also install ffmpeg from https://ffmpeg.org"
        )
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    audio = AudioSegment.from_file(filepath, format=ext)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    return tmp.name


def get_wav_duration(filepath: str) -> float:
    """Return duration in seconds for a WAV file."""
    with wave.open(filepath, "r") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


# ───────────────────────────────────────────────────────────────────────────
# Core transcription functions
# ───────────────────────────────────────────────────────────────────────────

def transcribe_file(
    filepath: str,
    engine: str = "google",
    language: str = "en-US",
    chunk_duration: int = 60,
) -> str:
    """
    Transcribe an audio file.

    Parameters
    ----------
    filepath       : Path to .wav / .mp3 / .ogg / .flac file
    engine         : 'google' | 'sphinx' | 'whisper'
    language       : BCP-47 language tag, e.g. 'en-US', 'hi-IN', 'fr-FR'
    chunk_duration : Split long files into N-second chunks (avoids API limits)

    Returns
    -------
    Full transcript string
    """
    ext = os.path.splitext(filepath)[1].lower()
    wav_path = filepath

    # Convert non-WAV formats
    tmp_wav = None
    if ext != ".wav":
        print(f"  [convert] {ext} → WAV …")
        wav_path = convert_to_wav(filepath)
        tmp_wav = wav_path  # remember for cleanup

    try:
        duration = get_wav_duration(wav_path)
        print(f"  [info]    Duration : {duration:.1f}s  |  Engine: {engine}  |  Lang: {language}")

        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 0.8     # seconds of silence → end of phrase
        recognizer.energy_threshold = 300    # mic sensitivity (auto-adjusted below)

        # ── chunk long files ──────────────────────────────────────────────
        chunks = []
        offset = 0
        while offset < duration:
            length = min(chunk_duration, duration - offset)
            chunks.append((offset, length))
            offset += length

        transcript_parts = []

        for idx, (start, length) in enumerate(chunks, 1):
            print(f"  [chunk {idx}/{len(chunks)}]  {start:.0f}s – {start+length:.0f}s …", end=" ", flush=True)

            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio_data = recognizer.record(source, offset=start, duration=length)

            text = _recognize(recognizer, audio_data, engine, language)
            print("done")
            transcript_parts.append(text)

        return " ".join(transcript_parts).strip()

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def transcribe_microphone(
    engine: str = "google",
    language: str = "en-US",
    timeout: int = 5,
    phrase_limit: int = 30,
) -> str:
    """
    Transcribe live microphone input (one utterance).

    Parameters
    ----------
    engine       : 'google' | 'sphinx'
    language     : BCP-47 language tag
    timeout      : Seconds to wait for speech to start
    phrase_limit : Max seconds of speech to capture

    Returns
    -------
    Transcribed text string
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("  [mic]  Calibrating for ambient noise (1 s) …")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    print("  [mic]  Listening — speak now …")
    with mic as source:
        try:
            audio_data = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_limit
            )
        except sr.WaitTimeoutError:
            raise RuntimeError("No speech detected within the timeout window.")

    print("  [mic]  Processing …")
    return _recognize(recognizer, audio_data, engine, language)


def _recognize(
    recognizer: sr.Recognizer,
    audio_data: sr.AudioData,
    engine: str,
    language: str,
) -> str:
    """Dispatch to the chosen recognition engine."""
    try:
        if engine == "google":
            return recognizer.recognize_google(audio_data, language=language)

        elif engine == "sphinx":
            # Offline — no internet required; English only; lower accuracy
            return recognizer.recognize_sphinx(audio_data)

        elif engine == "whisper":
            # pip install openai-whisper
            return recognizer.recognize_whisper(audio_data, language=language.split("-")[0])

        else:
            raise ValueError(f"Unknown engine '{engine}'. Use: google | sphinx | whisper")

    except sr.UnknownValueError:
        return "[inaudible]"
    except sr.RequestError as e:
        raise RuntimeError(f"API request failed: {e}")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Speech-to-Text Transcription Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Transcribe a WAV file (Google — default)
  python speech_to_text.py audio.wav

  # Transcribe an MP3 in Hindi
  python speech_to_text.py interview.mp3 --lang hi-IN

  # Offline transcription (Sphinx engine)
  python speech_to_text.py meeting.wav --engine sphinx

  # Live microphone input
  python speech_to_text.py --mic

  # Save transcript to a file
  python speech_to_text.py lecture.wav --output lecture.txt
""",
    )
    p.add_argument("filepath", nargs="?", help="Path to audio file (.wav, .mp3, .ogg, .flac)")
    p.add_argument("--mic", action="store_true", help="Use live microphone instead of a file")
    p.add_argument("--engine", default="google", choices=["google", "sphinx", "whisper"],
                   help="Recognition engine (default: google)")
    p.add_argument("--lang", default="en-US", metavar="LANG",
                   help="BCP-47 language code, e.g. en-US, hi-IN, fr-FR (default: en-US)")
    p.add_argument("--output", metavar="FILE", help="Save transcript to this file path")
    p.add_argument("--chunk", type=int, default=60, metavar="SECS",
                   help="Chunk size in seconds for long files (default: 60)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("\n━━━  VoiceScribe — Speech-to-Text  ━━━")
    t0 = time.perf_counter()

    try:
        if args.mic:
            transcript = transcribe_microphone(engine=args.engine, language=args.lang)
        elif args.filepath:
            if not os.path.isfile(args.filepath):
                print(f"\n  [error]  File not found: {args.filepath}", file=sys.stderr)
                sys.exit(1)
            transcript = transcribe_file(
                args.filepath,
                engine=args.engine,
                language=args.lang,
                chunk_duration=args.chunk,
            )
        else:
            parser.print_help()
            sys.exit(0)

    except RuntimeError as e:
        print(f"\n  [error]  {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    words = len(transcript.split()) if transcript.strip() else 0

    print(f"\n{'─'*50}")
    print(f"  Words   : {words}")
    print(f"  Time    : {elapsed:.1f}s")
    print(f"{'─'*50}\n")
    print(transcript)
    print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"  [saved]  → {args.output}")

    return transcript


if __name__ == "__main__":
    main()
