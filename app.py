from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from version_1 import PrototypeError, format_timestamp, process_audio


st.set_page_config(page_title="Speaker Transcript")
st.title("Speaker-labeled transcript")
st.caption("Local Pyannote Community-1 + Whisper proof of concept")

uploaded_audio = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
)

if uploaded_audio is not None and st.button("Transcribe", type="primary"):
    suffix = Path(uploaded_audio.name).suffix or ".audio"

    try:
        audio_directory = Path("audio")
        audio_directory.mkdir(exist_ok=True)

        original_name = Path(uploaded_audio.name).name or f"uploaded_audio{suffix}"
        saved_audio_path = audio_directory / original_name
        if saved_audio_path.exists():
            counter = 1
            while True:
                candidate = audio_directory / (
                    f"{saved_audio_path.stem}_{counter}{saved_audio_path.suffix}"
                )
                if not candidate.exists():
                    saved_audio_path = candidate
                    break
                counter += 1
        uploaded_bytes = uploaded_audio.getvalue()
        saved_audio_path.write_bytes(uploaded_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_bytes)
            temporary_path = Path(temporary_file.name)

        with st.spinner("Running diarization and transcription..."):
            output = process_audio(temporary_path)

        Path("transcript.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        st.success("Transcript created.")
        st.download_button(
            "Download transcript.json",
            data=json.dumps(output, indent=2, ensure_ascii=True) + "\n",
            file_name="transcript.json",
            mime="application/json",
        )

        for segment in output["segments"]:
            st.markdown(
                f"**[{format_timestamp(segment['start'])} - "
                f"{format_timestamp(segment['end'])}] {segment['speaker']}**\n\n"
                f"{segment['text']}"
            )

    except PrototypeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc.__class__.__name__}: {exc}")
    finally:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
else:
    st.info("Choose an audio file, then click Transcribe.")
