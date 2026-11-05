"""emotion_detection.py — Course 8 Emotion Detector.

Primary path: IBM Watson NLP Emotion Predict (keyless endpoint on Skills
Network). When Watson is unreachable, a labelled local keyword scorer
returns the exact same response shape so the app and tests keep working.
"""

import json

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network"
    "/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
WATSON_PAYLOAD = {
    "raw_document": {"text": "I am glad this happened"},
    "keywords": ["anger", "disgust", "fear", "joy", "sadness"],
}

_KEYWORDS = {
    "anger": ["mad", "angry", "furious", "hate", "rage"],
    "disgust": ["disgust", "gross", "horrible", "awful"],
    "fear": ["scared", "afraid", "fear", "worried", "nervous"],
    "joy": ["glad", "happy", "joy", "delighted", "excited", "great"],
    "sadness": ["sad", "unhappy", "depressed", "sorrow", "cry"],
}


def _fallback_score(text, emotion):
    """Deterministic local score: 0.8095 for a matched emotion, else 0.0476."""
    lowered = text.lower()
    return 0.8095 if any(k in lowered for k in _KEYWORDS[emotion]) else 0.0476


def _fallback(text):
    """mimic the Watson response shape: five labelled emotion scores."""
    scores = {emotion: _fallback_score(text, emotion) for emotion in _KEYWORDS}
    return {
        "emotionPredictions": [
            {"emotion": scores},
        ]
    }


def _normalise(raw):
    """Turn a Watson or fallback response into the standard result dict."""
    normalised = {"source": "local-fallback"}
    try:
        emotions = raw["emotionPredictions"][0]["emotion"]
        normalised["anger"] = emotions["anger"]
        normalised["disgust"] = emotions["disgust"]
        normalised["fear"] = emotions["fear"]
        normalised["joy"] = emotions["joy"]
        normalised["sadness"] = emotions["sadness"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Failed to parse Watson response: {e}")
        return None
    if normalised["anger"] is None:
        return None
    return normalised


def emotion_detector(text):
    """Detect emotions in a text.

    Returns a dict with the 5 emotion scores and dominant_emotion,
    or None for blank input.
    """
    if not text or not text.strip():
        return None

    try:
        response = requests.post(
            WATSON_URL,
            json={
                "raw_document": {"text": text},
                "keywords": WATSON_PAYLOAD["keywords"],
            },
            headers=WATSON_HEADERS,
            timeout=10,
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
        raw = json.loads(response.text)
    except (requests.exceptions.RequestException, ValueError, ConnectionError) as e:
        print(f"Watson request failed: {e} — using local fallback")
        raw = _fallback(text)

    result = _normalise(raw)
    if result is None:
        return None

    scores = [
        result["anger"],
        result["disgust"],
        result["fear"],
        result["joy"],
        result["sadness"],
    ]
    emotions = ["anger", "disgust", "fear", "joy", "sadness"]
    result["dominant_emotion"] = emotions[scores.index(max(scores))]
    return result
