import os

import whisper

DEFAULT_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny").strip() or "tiny"
ENGLISH_MODEL_NAME = os.getenv("WHISPER_ENGLISH_MODEL", "tiny.en").strip() or "tiny.en"
WHISPER_BEAM_SIZE = max(1, int(os.getenv("WHISPER_BEAM_SIZE", "2")))
WHISPER_BEST_OF = max(1, int(os.getenv("WHISPER_BEST_OF", "2")))

_models = {}


def _resolve_model_name(task="translate", language=None):
    normalized_task = (task or "translate").strip().lower()
    normalized_language = (language or "").strip().lower()

    # Use the smaller English-only model when possible for noticeably faster decoding.
    if normalized_task == "transcribe" and normalized_language == "en":
        return ENGLISH_MODEL_NAME

    return DEFAULT_MODEL_NAME


def get_model(task="translate", language=None):
    model_name = _resolve_model_name(task=task, language=language)
    model = _models.get(model_name)
    if model is None:
        model = whisper.load_model(model_name)
        _models[model_name] = model
    return model


def transcribe_audio(audio_path, task="translate", language=None):
    model = get_model(task=task, language=language)
    options = {
        "task": task,
        "fp16": False,
        "verbose": False,
        "condition_on_previous_text": False,
        "beam_size": WHISPER_BEAM_SIZE,
        "best_of": WHISPER_BEST_OF,
        "temperature": 0,
    }
    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)
    segments = []
    for segment in result.get("segments", []):
        segments.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": (segment.get("text", "") or "").strip(),
            }
        )

    return {
        "text": (result.get("text", "") or "").strip(),
        "language": result.get("language"),
        "segments": segments,
    }
