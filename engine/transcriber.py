import os
import requests
import whisper
from pydub import AudioSegment
from engine.config import config


class SpeechTranscriberEngine:
    """Base interface for speech-to-text engines."""

    def transcribe(self, chunk_path: str) -> str:
        raise NotImplementedError


class WhisperTranscriber(SpeechTranscriberEngine):
    """Local OpenAI Whisper transcription engine."""

    def __init__(self, model_size: str = config.WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            print(f"Initializing Whisper model [{self.model_size}]...")
            self._model = whisper.load_model(self.model_size)
            print("Whisper model loaded.")
        return self._model

    def transcribe(self, chunk_path: str) -> str:
        model = self._get_model()
        result = model.transcribe(chunk_path, task="transcribe")
        return result.get("text", "").strip()


class SarvamTranscriber(SpeechTranscriberEngine):
    """Sarvam AI Cloud STT & Translation Engine for Hinglish audio."""

    def __init__(self, api_key: str = config.SARVAM_API_KEY, model: str = config.SARVAM_STT_MODEL):
        self.api_key = api_key
        self.model = model
        self.piece_seconds = config.SARVAM_PIECE_SECONDS
        self.api_url = "https://api.sarvam.ai/speech-to-text-translate"

    def _send_piece(self, piece_path: str) -> str:
        headers = {"api-subscription-key": self.api_key}
        with open(piece_path, "rb") as f:
            files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
            data = {"model": self.model, "with_diarization": "false"}
            res = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=120)

        if not res.ok:
            print(f"Sarvam API error ({res.status_code}): {res.text}")
            res.raise_for_status()

        return res.json().get("transcript", "")

    def transcribe(self, chunk_path: str) -> str:
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY missing from environment configuration.")

        audio = AudioSegment.from_wav(chunk_path)
        piece_ms = self.piece_seconds * 1000
        full_text = []

        total_pieces = (len(audio) + piece_ms - 1) // piece_ms
        for i, start in enumerate(range(0, len(audio), piece_ms)):
            piece = audio[start : start + piece_ms]
            piece_path = f"{chunk_path}_sv_{i}.wav"
            piece.export(piece_path, format="wav")
            try:
                print(f"  → Transcribing Sarvam piece {i + 1}/{total_pieces}...")
                text = self._send_piece(piece_path)
                if text:
                    full_text.append(text)
            finally:
                if os.path.exists(piece_path):
                    os.remove(piece_path)

        return " ".join(full_text)


class TranscriberFactory:
    """Factory to instantiate the appropriate transcription engine."""

    @staticmethod
    def create_engine(language: str = "english") -> SpeechTranscriberEngine:
        if language.lower() in ["hinglish", "hindi"]:
            return SarvamTranscriber()
        return WhisperTranscriber()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    engine = TranscriberFactory.create_engine(language)
    return engine.transcribe(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    engine_name = "Sarvam AI" if language.lower() in ["hinglish", "hindi"] else "Whisper"
    print(f"Starting transcription pipeline using [{engine_name}] for {len(chunks)} chunk(s)...")

    transcript_parts = []
    for i, chunk in enumerate(chunks):
        print(f"Processing audio chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        if text:
            transcript_parts.append(text)

    full_transcript = " ".join(transcript_parts)
    print("Transcription complete.")
    return full_transcript
