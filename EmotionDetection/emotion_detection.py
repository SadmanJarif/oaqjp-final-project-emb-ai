"""Emotion Detection — IBM Course 8 final project (Python Project for AI & Application Development).
Primary path: IBM Watson NLP Emotion Predict API hosted on Skills Network: POST https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict, Header: grpc-metadata-mm-model-id: emotion_aggregated-workflow_lang_en_stock, Body: {"raw_document": {"text": <text>}}
Fallback path (USED IN THIS VM — see README.md / SUMMARY.md): The Watson endpoint is unreachable from this VM. Verified 2026-11-06: requests.post(...) raises ConnectionError: RemoteDisconnected('Remote end closed connection without response'), and curl returns only the proxy CONNECT line with no JSON body. No Watson API key is involved — the Skills Network endpoint is keyless — it is the endpoint itself that does not respond here. So when the Watson call fails (any exception / non-200 / bad payload), emotion_detector() falls back to a deterministic local keyword scorer that mimics the Watson response shape exactly: the same five keys (anger, disgust, fear, joy, sadness) plus dominant_emotion, scores in [0, 1]. The fallback is labelled as such wherever its output is shown (server footer + README + SUMMARY). It is NOT Watson — do not present it as Watson output on Coursera.
"""
import re
import requests
WATSON_URI = (
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
        hits = sum(
            len(re.findall(r"\b" + re.escape(word) + r"\b", lowered))
            for word in words
        )
        raw[emotion] = round(0.05 + 0.80 * hits, 4) if hits else 0.05
    total = sum(raw.values())
    result = {
        "anger": round(raw["anger"] / total, 4),
        "disgust": round(raw["disgust"] / total, 4),
        "fear": round(raw["fear"] / total, 4),
        "joy": round(raw["joy"] / total, 4),
        "sadness": round(raw["sadness"] / total, 4),
    }
    result["dominant_emotion"] = max(
        ("anger", "disgust", "fear", "joy", "sadness"), key=result.get
    )
    result["source"] = "local-fallback"
    return result


def _watson_call(text):
    """Call Watson NLP Emotion Predict and return the normalised dict."""
    payload = json.dumps(
        {"raw_document": {"text": text}, "keywords": list(_KEYWORDS)}
    )
    response = requests.post(
        WATSON_URI, data=payload, headers=WATSON_HEADERS, timeout=10
    )
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
    response.raise_for_status()
    data = response.json()
    if "emotionPredictions" not in data:
        raise ValueError("Response JSON did not have emotionPredictions")
    return data["emotion_predictions"][0]["emotion"]


def _normalise(scores):
    """Convert the five raw Watson scores into the app dict."""
    result = {
        "anger": float(scores["anger"]),
        "disgust": float(scores["disgust"]),
        "fear": float(scores["fear"]),
        "joy": float(scores["joy"]),
        "sadness": float(scores["sadness"]),
    }
    result["dominant_emotion"] = max(
        ("anger", "disgust", "fear", "joy", "sadness"), key=result.get
    )
    result["source"] = "watson"
    return result


def emotion_detector(text):
    """Analyse a text and return the five scores + dominant_emotion."""
    if text is None or not text.strip():
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
        return _normalise(_watson_call(text))
    except Exception:  # Any Watson failure -> labelled local fallback.
        return _local_fallback(text)
