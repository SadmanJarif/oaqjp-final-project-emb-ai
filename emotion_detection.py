"""Emotion Detection — IBM Course 8 final project (Python Project for AI
& Application Development).

Primary path: IBM Watson NLP Emotion Predict API hosted on Skills Network:
    POST https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict
    Header: grpc-metadata-mm-model-id: emotion_aggregated-workflow_lang_en_stock
    Body:   {"raw_document": {"text": <text>}}

Fallback path (USED IN THIS VM — see README.md / SUMMARY.md):
    The Watson endpoint is unreachable from this VM. Verified 2026-11-06:
    `requests.post(...)` raises
    ConnectionError: RemoteDisconnected('Remote end closed connection
    without response'), and curl returns only the proxy CONNECT line with
    no JSON body. No Watson API key is involved — the Skills Network
    endpoint is keyless — it is the endpoint itself that does not respond
    here. So when the Watson call fails (any exception / non-200 / bad
    payload), emotion_detector() falls back to a deterministic local
    keyword scorer that mimics the Watson response shape exactly:
    the same five keys (anger, disgust, fear, joy, sadness) plus
    dominant_emotion, scores in [0, 1]. The fallback is labelled as such
    wherever its output is shown (server footer + README + SUMMARY).
    It is NOT Watson — do not present it as Watson output on Coursera.
"""

import re

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}

# Keyword lexicon for the local fallback. Word-boundary matched, lowercase.
_KEYWORDS = {
    "anger": [
        "mad", "angry", "anger", "furious", "rage", "annoyed", "irritated",
        "outraged", "frustrated", "hate", "infuriated",
    ],
    "disgust": [
        "disgust", "disgusted", "disgusting", "gross", "revolting",
        "nauseated", "sickened", "repulsed",
    ],
    "fear": [
        "scared", "fear", "afraid", "terrified", "anxious", "worried",
        "nervous", "panic", "dread", "fail", "frightened",
    ],
    "joy": [
        "glad", "happy", "joy", "joyful", "delighted", "excited",
        "pleased", "wonderful", "great", "love", "cheerful", "thrilled",
    ],
    "sadness": [
        "sad", "sadness", "sorrow", "unhappy", "depressed", "grief",
        "heartbroken", "down", "miserable", "cry", "crying",
    ],
}


def _local_fallback(text):
    """Deterministic keyword scorer mimicking the Watson response shape."""
    lowered = text.lower()
    raw = {}
    for emotion, words in _KEYWORDS.items():
        count = 0
        for word in words:
            if re.search(r"\b" + re.escape(word) + r"\b", lowered):
                count += 1
        raw[emotion] = count
    total_hits = sum(raw.values())
    if total_hits == 0:
        # No emotion keywords: neutral-ish spread, joy marginally highest
        # so dominant_emotion is still a real value, never None for
        # non-blank text.
        scores = {
            "anger": 0.05, "disgust": 0.05, "fear": 0.05,
            "joy": 0.10, "sadness": 0.05,
        }
    else:
        scores = {}
        for emotion in _KEYWORDS:
            # Base 0.05 + 0.80 share for matched keywords.
            scores[emotion] = round(
                0.05 + 0.80 * (raw[emotion] / total_hits), 4
            ) if raw[emotion] else 0.05
        # Normalise so the five scores sum to ~1.0.
        total = sum(scores.values())
        scores = {k: round(v / total, 4) for k, v in scores.items()}
    dominant = max(scores, key=scores.get)
    result = dict(scores)
    result["dominant_emotion"] = dominant
    result["source"] = "local-fallback"
    return result


def _watson_call(text):
    """Call the Watson NLP Emotion Predict API. Raises on any failure."""
    response = requests.post(
        WATSON_URL,
        json={"raw_document": {"text": text}},
        headers=WATSON_HEADERS,
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    emotions = data["emotion_predictions"][0]["emotion"]
    result = {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
    }
    result["dominant_emotion"] = max(result, key=result.get)
    result["source"] = "watson"
    return result


def emotion_detector(text_to_analyse):
    """Detect emotions in text.

    Returns a dict with anger, disgust, fear, joy, sadness (floats
    0-1, or None) and dominant_emotion (str or None).
    Blank / missing text returns all None, per the IBM lab spec.
    On Watson failure, returns the local-fallback scores (see module
    docstring) — never raises for ordinary input.
    """
    if text_to_analyse is None or not str(text_to_analyse).strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
            "source": None,
        }
    try:
        return _watson_call(str(text_to_analyse))
    except Exception:
        return _local_fallback(str(text_to_analyse))
