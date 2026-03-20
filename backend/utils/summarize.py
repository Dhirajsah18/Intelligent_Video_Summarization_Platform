import os
import re
from collections import Counter

from transformers import pipeline

SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6").strip() or "sshleifer/distilbart-cnn-12-6"
_summarizer = None

SUPPORTED_SUMMARY_STYLES = {"general", "business", "student", "casual"}
FAST_SUMMARY_TRIGGER_WORDS = int(os.getenv("FAST_SUMMARY_TRIGGER_WORDS", "1800"))
FAST_KEY_POINT_TRIGGER_SEGMENTS = int(os.getenv("FAST_KEY_POINT_TRIGGER_SEGMENTS", "40"))


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        # Lazy-loading avoids paying model startup cost on every server boot.
        _summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
    return _summarizer


def _split_sentences(text):
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [item.strip() for item in parts if item and item.strip()]


def _to_bullets(sentences, limit=5):
    items = sentences[:limit]
    return "\n".join(f"- {item}" for item in items)


def _normalize_sentence(sentence):
    return re.sub(r"\s+", " ", (sentence or "").strip())


def _tokenize(text):
    return re.findall(r"[a-zA-Z0-9']+", (text or "").lower())


def _extractive_summary_ranked(text, summary_length="medium"):
    sentences = [_normalize_sentence(item) for item in _split_sentences(text)]
    sentences = [item for item in sentences if item]
    if not sentences:
        return text.strip()

    token_counts = Counter(_tokenize(text))
    sentence_limit_map = {
        "short": 3,
        "medium": 5,
        "long": 8,
    }
    word_limit_map = {
        "short": 70,
        "medium": 130,
        "long": 220,
    }

    def score(sentence, idx):
        tokens = _tokenize(sentence)
        if not tokens:
            return 0
        keyword_score = sum(token_counts.get(token, 0) for token in set(tokens))
        # Prefer concise informative lines and slightly reward earlier context.
        brevity_score = max(0, 24 - abs(len(tokens) - 24))
        position_score = max(0, 8 - idx)
        return keyword_score + brevity_score + position_score

    ranked = sorted(
        [(idx, sentence, score(sentence, idx)) for idx, sentence in enumerate(sentences)],
        key=lambda item: item[2],
        reverse=True,
    )

    picks = sorted(ranked[: sentence_limit_map.get(summary_length, 5)], key=lambda item: item[0])
    selected = []
    word_total = 0
    seen = set()
    for _, sentence, _ in picks:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        word_total += len(sentence.split())
        if word_total >= word_limit_map.get(summary_length, 130):
            break

    return " ".join(selected).strip()


def _extractive_summary(text, summary_length="medium"):
    sentences = [_normalize_sentence(item) for item in _split_sentences(text)]
    sentences = [item for item in sentences if item]
    if not sentences:
        return text.strip()

    sentence_limit_map = {
        "short": 3,
        "medium": 5,
        "long": 7,
    }
    word_limit_map = {
        "short": 70,
        "medium": 120,
        "long": 180,
    }

    selected = []
    seen = set()
    word_limit = word_limit_map.get(summary_length, 120)
    for sentence in sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= sentence_limit_map.get(summary_length, 5):
            break
        if sum(len(item.split()) for item in selected) >= word_limit:
            break

    return " ".join(selected).strip()


def _apply_summary_style(summary_text, summary_style):
    style = (summary_style or "general").strip().lower()
    sentences = _split_sentences(summary_text)

    if style == "general" or not sentences:
        return summary_text.strip()

    if style == "business":
        return "\n".join(
            [
                "Business Snapshot",
                f"Executive Summary: {sentences[0]}",
                "Key Points:",
                _to_bullets(sentences[1:] or sentences, limit=4),
            ]
        ).strip()

    if style == "student":
        return "\n".join(
            [
                "Student Notes",
                "Core Concepts:",
                _to_bullets(sentences, limit=4),
                "Quick Revision:",
                f"- {sentences[0]}",
            ]
        ).strip()

    if style == "casual":
        concise = " ".join(sentences[:3])
        return f"In short: {concise}".strip()

    return summary_text.strip()


def _resolve_summary_lengths(word_count, preferred_max, preferred_min):
    if word_count <= 0:
        return preferred_max, preferred_min

    # Keep generation lengths proportional for short inputs to avoid transformer warnings.
    safe_max = min(preferred_max, max(20, int(word_count * 0.7)))
    safe_min = min(preferred_min, max(12, int(safe_max * 0.55)))

    if safe_min >= safe_max:
        safe_min = max(10, safe_max - 8)

    return safe_max, safe_min


def summarize_text(text, summary_length="medium", summary_style="general"):
    words = text.split()
    if not words:
        return "No speech detected in the uploaded video."

    if len(words) >= FAST_SUMMARY_TRIGGER_WORDS:
        base_summary = _extractive_summary_ranked(text, summary_length=summary_length)
        return _apply_summary_style(base_summary, summary_style)

    length_map = {
        "short": {"chunk_size": 250, "max_length": 90, "min_length": 35},
        "medium": {"chunk_size": 350, "max_length": 150, "min_length": 60},
        "long": {"chunk_size": 450, "max_length": 220, "min_length": 90},
    }
    config = length_map.get(summary_length, length_map["medium"])
    chunk_size = config["chunk_size"]

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    summary_chunks = []
    summarizer = _get_summarizer()
    for chunk in chunks:
        chunk_word_count = len(chunk.split())
        max_length, min_length = _resolve_summary_lengths(
            chunk_word_count,
            config["max_length"],
            config["min_length"],
        )
        result = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        summary_chunks.append(result[0]["summary_text"])

    base_summary = " ".join(summary_chunks).strip()
    return _apply_summary_style(base_summary, summary_style)


def generate_time_key_points(segments, max_points=8, window_seconds=120):
    if not segments:
        return []

    fast_mode = len(segments) >= FAST_KEY_POINT_TRIGGER_SEGMENTS

    grouped = []
    current_group = []
    group_start = None

    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = (segment.get("text", "") or "").strip()
        if not text:
            continue

        if group_start is None:
            group_start = start

        if (start - group_start) >= window_seconds and current_group:
            grouped.append(current_group)
            current_group = []
            group_start = start

        current_group.append({"start": start, "end": end, "text": text})

    if current_group:
        grouped.append(current_group)

    points = []
    for group in grouped[:max_points]:
        start = group[0]["start"]
        end = group[-1]["end"]
        chunk_text = " ".join(item["text"] for item in group).strip()
        if not chunk_text:
            continue

        words = chunk_text.split()
        if len(words) > 380:
            chunk_text = " ".join(words[:380])

        word_count = len(chunk_text.split())
        if word_count < 25:
            point_text = chunk_text
        elif fast_mode:
            point_text = _extractive_summary_ranked(chunk_text, summary_length="short")
        else:
            summarizer = _get_summarizer()
            max_length, min_length = _resolve_summary_lengths(word_count, 70, 22)
            result = summarizer(
                chunk_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            point_text = result[0]["summary_text"].strip()

        points.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "point": point_text,
            }
        )

    return points
