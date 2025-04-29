import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Centralized application configuration parameters."""

    # Models & Keys
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL", "small")
    SARVAM_STT_MODEL: str = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
    LLM_MODEL_NAME: str = "mistral-small-latest"

    # Vector DB & Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_DIR: str = "vector_db"
    COLLECTION_NAME: str = "meeting_transcript_collection"
    DEFAULT_TOP_K: int = 4

    # Audio Processing Settings
    DOWNLOAD_DIR: str = "downloades"
    TARGET_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    DEFAULT_CHUNK_MINUTES: int = 10
    SARVAM_PIECE_SECONDS: int = 25

    def validate(self) -> list:
        """Validate required configuration settings."""
        warnings = []
        if not self.MISTRAL_API_KEY:
            warnings.append("MISTRAL_API_KEY is not set. LLM features will fail without an API key.")
        if not self.SARVAM_API_KEY:
            warnings.append("SARVAM_API_KEY is not set. Hinglish transcription will be disabled.")
        return warnings


config = AppConfig()
