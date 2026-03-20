import re
from collections import Counter

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

LOW_SIGNAL_WORDS = {
    "okay",
    "yeah",
    "yes",
    "no",
    "uh",
    "um",
    "like",
    "just",
    "really",
    "basically",
    "actually",
    "video",
    "thing",
    "things",
    "first",
    "also",
    "talk",
    "talking",
    "said",
    "says",
}


def _tokenize(text):
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def _meaningful_tokens(text):
    return [
        token
        for token in _tokenize(text)
        if token not in STOPWORDS and token not in LOW_SIGNAL_WORDS and len(token) > 2
    ]


def _split_sentences(text):
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _trim_to_word_range(text, *, min_words=100, max_words=150, fallback_text=""):
    words = text.split()
    if not words:
        return ""

    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:- ")
        if text and text[-1] not in ".!?":
            text += "."
        words = text.split()

    if len(words) >= min_words:
        return text

    sentences = _split_sentences(text)
    fallback_sentences = _split_sentences(fallback_text)
    if fallback_sentences:
        for sentence in fallback_sentences:
            if sentence not in sentences:
                sentences.append(sentence)

    if not sentences:
        return text

    expanded_words = []
    for sentence in sentences:
        expanded_words.extend(sentence.split())
        if len(expanded_words) >= min_words:
            break

    result = " ".join(expanded_words).strip()
    if result and result[-1] not in ".!?":
        result += "."
    return result or text


def _dedupe_sentences(sentences):
    seen = set()
    unique = []
    for sentence in sentences:
        normalized = re.sub(r"\s+", " ", sentence.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(sentence.strip())
    return unique


def _build_transcript_chunks(segments, chunk_word_limit=160, overlap_segments=1):
    prepared = [segment for segment in (segments or []) if (segment.get("text", "") or "").strip()]
    if not prepared:
        return []

    chunks = []
    index = 0
    while index < len(prepared):
        current_segments = []
        words = []
        start = float(prepared[index].get("start", 0.0))
        cursor = index

        while cursor < len(prepared):
            text = (prepared[cursor].get("text", "") or "").strip()
            segment_words = text.split()
            if current_segments and len(words) + len(segment_words) > chunk_word_limit:
                break
            current_segments.append(prepared[cursor])
            words.extend(segment_words)
            cursor += 1

        end = float(current_segments[-1].get("end", start))
        chunk_text = " ".join(item.get("text", "").strip() for item in current_segments).strip()
        chunks.append(
            {
                "start": start,
                "end": end,
                "text": chunk_text,
                "tokens": Counter(_meaningful_tokens(chunk_text)),
            }
        )

        step = max(1, len(current_segments) - overlap_segments)
        index += step

    return chunks


def _score_chunk(chunk, query_tokens):
    if not query_tokens:
        return 0

    overlap = sum(min(chunk["tokens"].get(token, 0), count) for token, count in query_tokens.items())
    density_bonus = len([token for token in query_tokens if token in chunk["tokens"]]) * 0.35
    return overlap + density_bonus


def _score_sentence(sentence, query_tokens):
    tokens = Counter(_meaningful_tokens(sentence))
    if not tokens:
        return 0
    overlap = sum(min(tokens.get(token, 0), count) for token, count in query_tokens.items())
    coverage = len([token for token in query_tokens if token in tokens]) * 0.3
    return overlap + coverage + min(len(sentence.split()) / 45, 1.0)


def _query_expansion_tokens(question):
    question_lower = (question or "").lower()
    expanded = []
    if any(word in question_lower for word in {"difference", "different", "vs", "compare", "comparison"}):
        expanded.extend(["difference", "different", "compare", "comparison", "instead", "whereas"])
    if any(word in question_lower for word in {"what", "define", "meaning", "explain"}):
        expanded.extend(["means", "defined", "definition", "explains", "concept"])
    if any(word in question_lower for word in {"why", "reason"}):
        expanded.extend(["because", "reason", "therefore"])
    return expanded


def retrieve_relevant_chunks(question, transcript_segments, max_chunks=4):
    chunks = _build_transcript_chunks(transcript_segments)
    if not chunks:
        return []

    query_token_list = _meaningful_tokens(question) + _query_expansion_tokens(question)
    query_tokens = Counter(query_token_list)
    if not query_tokens:
        return chunks[:max_chunks]

    ranked = sorted(
        chunks,
        key=lambda chunk: (_score_chunk(chunk, query_tokens), -chunk["start"]),
        reverse=True,
    )
    selected = [chunk for chunk in ranked if _score_chunk(chunk, query_tokens) > 0][:max_chunks]
    if not selected:
        selected = ranked[:max_chunks]

    return sorted(selected, key=lambda item: item["start"])


def _topic_from_text(text, max_words=6):
    tokens = _meaningful_tokens(text)
    if not tokens:
        return None
    topic = " ".join(tokens[:max_words]).strip()
    return topic if len(topic) >= 6 else None


def _extract_keyword_topics(summary, transcript_segments, max_topics=4):
    bucket = []
    bucket.extend(_meaningful_tokens(summary))
    for segment in transcript_segments[:60]:
        bucket.extend(_meaningful_tokens(segment.get("text", "")))

    counts = Counter(bucket)
    topics = []
    for token, _ in counts.most_common(20):
        if len(token) < 4:
            continue
        if token in LOW_SIGNAL_WORDS or token in STOPWORDS:
            continue
        topics.append(token)
        if len(topics) >= max_topics:
            break
    return topics


def generate_suggested_questions(summary, transcript_segments, time_key_points=None, max_questions=4):
    candidates = []

    for item in time_key_points or []:
        point = (item.get("point", "") or "").strip()
        if point:
            candidates.append(point)

    candidates.extend(_split_sentences(summary)[:4])

    if not candidates:
        for segment in transcript_segments or []:
            text = (segment.get("text", "") or "").strip()
            if text:
                candidates.append(text)
            if len(candidates) >= 4:
                break

    prompts = [
        "Can you explain {topic} in simple terms?",
        "What is the main takeaway about {topic}?",
        "How is {topic} used in practice?",
        "Why does {topic} matter here?",
    ]

    questions = []
    seen = set()
    for candidate, template in zip(candidates, prompts):
        topic = _topic_from_text(candidate)
        if not topic:
            continue
        question = template.format(topic=topic)
        normalized = question.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        questions.append(question)
        if len(questions) >= max_questions:
            break

    if len(questions) < max_questions:
        keyword_topics = _extract_keyword_topics(summary, transcript_segments, max_topics=max_questions)
        for idx, topic in enumerate(keyword_topics):
            template = prompts[idx % len(prompts)]
            question = template.format(topic=topic)
            normalized = question.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            questions.append(question)
            if len(questions) >= max_questions:
                break

    if not questions:
        questions = [
            "Would you like to know more about the main topic of this video?",
            "What is the biggest takeaway from this video?",
        ]

    return questions


def answer_question_from_transcript(question, transcript_text, transcript_segments):
    normalized_question = (question or "").strip()
    if not normalized_question:
        raise ValueError("Question is required.")

    chunks = retrieve_relevant_chunks(normalized_question, transcript_segments, max_chunks=4)
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"Timestamp {int(chunk['start'])} to {int(chunk['end'])}: {chunk['text']}"
        )

    if not context_parts and transcript_text:
        context_parts.append(transcript_text[:2500])

    if not context_parts:
        raise ValueError("Transcript is required to answer questions about the video.")

    combined_context = " ".join(context_parts)
    query_token_list = _meaningful_tokens(normalized_question) + _query_expansion_tokens(normalized_question)
    query_tokens = Counter(query_token_list)
    candidate_sentences = _dedupe_sentences(_split_sentences(combined_context))

    ranked_sentences = sorted(
        candidate_sentences,
        key=lambda sentence: _score_sentence(sentence, query_tokens),
        reverse=True,
    )

    selected_sentences = []
    word_total = 0
    for sentence in ranked_sentences:
        if _score_sentence(sentence, query_tokens) <= 0:
            continue
        selected_sentences.append(sentence)
        word_total += len(sentence.split())
        if word_total >= 95:
            break

    if not selected_sentences:
        selected_sentences = candidate_sentences[:4]

    answer = " ".join(selected_sentences).strip()
    answer = _trim_to_word_range(answer, min_words=70, max_words=130, fallback_text=combined_context)

    source_chunks = [
        {
            "start": round(chunk["start"], 2),
            "end": round(chunk["end"], 2),
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    return {
        "question": normalized_question,
        "answer": answer,
        "sources": source_chunks,
        "word_count": len(answer.split()),
    }
