import base64
import json
import subprocess
import tempfile
from pathlib import Path
from urllib import error, request


class ModerationConfigError(RuntimeError):
    pass


class ModerationRuntimeError(RuntimeError):
    pass


class ModerationRejectedError(RuntimeError):
    pass


UNSAFE_LABEL_HINTS = (
    "nsfw",
    "porn",
    "porno",
    "pornography",
    "explicit",
    "sexual",
    "nudity",
    "nude",
    "adult",
    "erotic",
    "sex",
)


def _parse_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extract_frames(video_path, frame_interval_seconds, max_frames):
    with tempfile.TemporaryDirectory(prefix="moderation_frames_") as tmp_dir:
        output_pattern = str(Path(tmp_dir) / "frame_%03d.jpg")
        fps_filter = f"fps=1/{max(1, frame_interval_seconds)}"
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            fps_filter,
            "-frames:v",
            str(max_frames),
            output_pattern,
            "-y",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise ModerationRuntimeError(f"Failed to extract frames for moderation: {result.stderr.strip()}")

        frame_paths = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        return [frame.read_bytes() for frame in frame_paths]


def _score_from_response(payload):
    if not isinstance(payload, dict):
        return None

    direct_keys = [
        "max_risk_score",
        "risk_score",
        "unsafe_score",
        "nsfw_score",
        "violence_score",
    ]
    for key in direct_keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    categories = payload.get("categories")
    if isinstance(categories, dict):
        risk_values = []
        for key in ("violence", "graphic_violence", "sexual", "explicit", "self_harm", "hate"):
            value = categories.get(key)
            if isinstance(value, (int, float)):
                risk_values.append(float(value))
        if risk_values:
            return max(risk_values)

    return None


def _value_is_unsafe_label(value):
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return any(hint in normalized for hint in UNSAFE_LABEL_HINTS)


def _score_from_candidate(candidate):
    if not isinstance(candidate, dict):
        return None

    score = _score_from_response(candidate)
    if score is not None:
        return score

    for key in ("score", "confidence", "probability", "value"):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    return None


def _iter_response_candidates(payload):
    stack = [payload]
    seen = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, dict):
            yield current
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    stack.append(item)


def _blocked_from_response(payload, threshold):
    if not isinstance(payload, dict):
        return False, None

    highest_unsafe_score = None

    for candidate in _iter_response_candidates(payload):
        explicit_flag = candidate.get("blocked")
        if isinstance(explicit_flag, bool):
            score = _score_from_candidate(candidate)
            return explicit_flag, score

        flagged = candidate.get("flagged")
        if isinstance(flagged, bool):
            score = _score_from_candidate(candidate)
            return flagged, score

        is_safe = candidate.get("is_safe")
        if isinstance(is_safe, bool):
            score = _score_from_candidate(candidate)
            return (not is_safe), score

        label_keys = ("label", "class", "category", "name")
        if any(_value_is_unsafe_label(candidate.get(key)) for key in label_keys):
            score = _score_from_candidate(candidate)
            if score is None or score >= threshold:
                return True, score
            if highest_unsafe_score is None or score > highest_unsafe_score:
                highest_unsafe_score = score

        categories = candidate.get("categories")
        if isinstance(categories, dict):
            for key, value in categories.items():
                if _value_is_unsafe_label(key) and isinstance(value, (int, float)):
                    numeric_value = float(value)
                    if numeric_value >= threshold:
                        return True, numeric_value
                    if highest_unsafe_score is None or numeric_value > highest_unsafe_score:
                        highest_unsafe_score = numeric_value

        predictions = candidate.get("predictions")
        if isinstance(predictions, list):
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                if any(_value_is_unsafe_label(item.get(key)) for key in label_keys):
                    score = _score_from_candidate(item)
                    if score is None or score >= threshold:
                        return True, score
                    if highest_unsafe_score is None or score > highest_unsafe_score:
                        highest_unsafe_score = score

        score = _score_from_response(candidate)
        if score is not None and score >= threshold:
            return True, score

    if highest_unsafe_score is not None:
        return False, highest_unsafe_score

    return False, None


def run_video_moderation(
    *,
    video_path,
    enabled,
    api_url,
    api_key=None,
    threshold=0.65,
    frame_interval_seconds=2,
    max_frames=12,
    timeout_seconds=20,
):
    if not _parse_bool(enabled, default=False):
        return {"moderation_enabled": False, "moderation_passed": True, "checked_frames": 0}

    if not api_url:
        raise ModerationConfigError("CONTENT_MODERATION_API_URL is required when moderation is enabled.")

    threshold = _parse_float(threshold, 0.65)
    frame_interval_seconds = _parse_int(frame_interval_seconds, 2)
    max_frames = _parse_int(max_frames, 12)
    timeout_seconds = _parse_int(timeout_seconds, 20)

    frames = _extract_frames(
        video_path=video_path,
        frame_interval_seconds=frame_interval_seconds,
        max_frames=max_frames,
    )
    if not frames:
        raise ModerationRuntimeError("No frames could be extracted for moderation.")

    encoded_frames = [base64.b64encode(frame).decode("utf-8") for frame in frames]
    payload = {
        "media_type": "video_frames",
        "content_encoding": "base64_jpeg",
        "frames": encoded_frames,
        "threshold": threshold,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        url=api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw_body = response.read().decode("utf-8")
            response_payload = json.loads(raw_body) if raw_body else {}
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise ModerationRuntimeError(
            f"Moderation API returned HTTP {exc.code}. Details: {details[:300]}"
        ) from exc
    except error.URLError as exc:
        raise ModerationRuntimeError(f"Moderation API request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ModerationRuntimeError("Moderation API returned invalid JSON.") from exc

    blocked, score = _blocked_from_response(response_payload, threshold)
    if blocked:
        reason_score = f" (risk score: {score:.2f})" if score is not None else ""
        raise ModerationRejectedError(
            f"Uploaded video failed safety moderation{reason_score}. Please upload a different video."
        )

    return {
        "moderation_enabled": True,
        "moderation_passed": True,
        "checked_frames": len(encoded_frames),
        "risk_score": score,
    }
