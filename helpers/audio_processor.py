import os
import yt_dlp
from pydub import AudioSegment
from engine.config import config

os.makedirs(config.DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download audio stream from YouTube URL as a standard WAV file."""
    output_template = os.path.join(config.DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=True)
        raw_filename = ydl.prepare_filename(info)
        base_name, _ = os.path.splitext(raw_filename)
        filename = f"{base_name}.wav"

    return filename


def convert_to_wav(input_path: str) -> str:
    """Normalize local audio or video file to 16kHz mono WAV format."""
    output_path = f"{os.path.splitext(input_path)[0]}_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(config.AUDIO_CHANNELS).set_frame_rate(config.TARGET_SAMPLE_RATE)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = config.DEFAULT_CHUNK_MINUTES) -> list:
    """Split WAV audio file into uniform time segments for downstream processing."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunk_paths = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        segment = audio[start : start + chunk_ms]
        chunk_file = f"{wav_path}_chunk_{i}.wav"
        segment.export(chunk_file, format="wav")
        chunk_paths.append(chunk_file)

    return chunk_paths


def process_input(source: str) -> list:
    """Main entry for audio acquisition, conversion, and segment chunking."""
    if source.startswith("http://") or source.startswith("https://"):
        print("Processing YouTube URL source...")
        wav_file = download_youtube_audio(source)
    else:
        print("Processing local audio/video file...")
        wav_file = convert_to_wav(source)

    print("Splitting audio into chunks...")
    chunks = chunk_audio(wav_file)
    print(f"Audio processing complete: {len(chunks)} segment chunk(s) generated.")
    return chunks
