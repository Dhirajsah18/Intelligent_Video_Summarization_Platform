import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.extract_audio import extract_audio
from utils.moderation import (
    ModerationConfigError,
    ModerationRejectedError,
    ModerationRuntimeError,
    run_video_moderation,
)
from utils.rag import answer_question_from_transcript, generate_suggested_questions
from utils.summarize import SUPPORTED_SUMMARY_STYLES, generate_time_key_points, summarize_text
from utils.transcribe import transcribe_audio

app = FastAPI()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _format_timestamp(seconds):
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_bool(value, default=True):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ensure_moderation_passed(moderation_result):
    if not isinstance(moderation_result, dict):
        raise ModerationRuntimeError("Moderation did not return a valid result.")

    if not moderation_result.get("moderation_passed"):
        raise ModerationRejectedError(
            "Uploaded video failed safety moderation. Please upload a different video."
        )


@app.get("/")
def index():
    return {"message": "AI Video Summarization Backend Running!"}


class VideoQuestionRequest(BaseModel):
    question: str
    transcript_text: str = ""
    transcript_segments: list[dict] = []


@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    summary_length: str = Form("medium"),
    summary_style: str = Form("general"),
    transcription_task: str = Form("translate"),
    source_language: str = Form("auto"),
    include_key_points: str = Form("true"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is missing.")

    normalized_task = (transcription_task or "translate").strip().lower()
    if normalized_task not in {"transcribe", "translate"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid transcription_task. Use 'transcribe' or 'translate'.",
        )

    normalized_summary_style = (summary_style or "general").strip().lower()
    if normalized_summary_style not in SUPPORTED_SUMMARY_STYLES:
        supported_styles = ", ".join(sorted(SUPPORTED_SUMMARY_STYLES))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid summary_style. Use one of: {supported_styles}.",
        )

    normalized_source_language = (source_language or "auto").strip().lower()
    if normalized_source_language == "auto":
        whisper_language = None
    else:
        if not normalized_source_language.isalpha() or len(normalized_source_language) != 2:
            raise HTTPException(
                status_code=400,
                detail="Invalid source_language. Use 'auto' or a 2-letter language code like 'fr'.",
            )
        whisper_language = normalized_source_language
    include_key_points_enabled = _parse_bool(include_key_points, default=True)

    safe_name = Path(file.filename).name
    request_id = uuid4().hex
    video_path = UPLOAD_DIR / f"{request_id}_{safe_name}"
    audio_path = OUTPUT_DIR / f"{request_id}.wav"

    try:
        with video_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        moderation_result = run_video_moderation(
            video_path=str(video_path),
            enabled=os.getenv("CONTENT_MODERATION_ENABLED", "false"),
            api_url=os.getenv("CONTENT_MODERATION_API_URL", "").strip(),
            api_key=os.getenv("CONTENT_MODERATION_API_KEY", "").strip() or None,
            threshold=os.getenv("CONTENT_MODERATION_THRESHOLD", "0.65"),
            frame_interval_seconds=os.getenv("CONTENT_MODERATION_FRAME_INTERVAL_SECONDS", "4"),
            max_frames=os.getenv("CONTENT_MODERATION_MAX_FRAMES", "6"),
            timeout_seconds=os.getenv("CONTENT_MODERATION_TIMEOUT_SECONDS", "20"),
        )
        _ensure_moderation_passed(moderation_result)

        extract_audio(str(video_path), str(audio_path))
        transcription_result = transcribe_audio(
            str(audio_path),
            task=normalized_task,
            language=whisper_language,
        )
        text = transcription_result.get("text", "")
        segments = transcription_result.get("segments", [])
        summary = summarize_text(
            text,
            summary_length=summary_length,
            summary_style=normalized_summary_style,
        )
        key_points = generate_time_key_points(segments) if include_key_points_enabled else []
        suggested_questions = generate_suggested_questions(
            summary=summary,
            transcript_segments=segments,
            time_key_points=key_points,
        )

        transcript_segments = [
            {
                "start": item["start"],
                "end": item["end"],
                "start_label": _format_timestamp(item["start"]),
                "end_label": _format_timestamp(item["end"]),
                "text": item["text"],
            }
            for item in segments
        ]

        for point in key_points:
            point["start_label"] = _format_timestamp(point["start"])
            point["end_label"] = _format_timestamp(point["end"])
    except ModerationRejectedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ModerationConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Moderation configuration error: {exc}") from exc
    except ModerationRuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Moderation service error: {exc}") from exc
    except RuntimeError as exc:
        raw_error = str(exc)
        lowered_error = raw_error.lower()

        if "moov atom not found" in lowered_error or "invalid data found when processing input" in lowered_error:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is invalid or incomplete. Please re-export/download and upload again.",
            ) from exc

        if "no such file or directory" in lowered_error and "ffmpeg" in lowered_error:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg is not installed or not available in PATH on the backend server.",
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="Audio extraction failed. Please try another video file.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to process video.") from exc
    finally:
        await file.close()
        if video_path.exists():
            video_path.unlink()
        if audio_path.exists():
            audio_path.unlink()

    return {
        "status": "success",
        "filename": safe_name,
        "summary": summary,
        "transcription_task": normalized_task,
        "summary_style": normalized_summary_style,
        "source_language": normalized_source_language,
        "include_key_points": include_key_points_enabled,
        "detected_language": transcription_result.get("language"),
        "transcript_text": text,
        "transcript_segments": transcript_segments,
        "time_key_points": key_points,
        "suggested_questions": suggested_questions,
        "moderation": moderation_result,
    }


@app.post("/ask-video")
def ask_video_question(payload: VideoQuestionRequest):
    try:
        result = answer_question_from_transcript(
            payload.question,
            payload.transcript_text,
            payload.transcript_segments,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to answer the video question.") from exc

    for item in result.get("sources", []):
        item["start_label"] = _format_timestamp(item["start"])
        item["end_label"] = _format_timestamp(item["end"])

    return result
