# Final Project

Python Flask app that analyses the emotions in a statement (anger, disgust,
fear, joy, sadness) and reports the dominant emotion. IBM Full Stack Software
Developer Professional Certificate, Course 8 — *Python Project for AI &
Application Development*. Mark learner/2432, 16 points.

## Files

| File | Purpose |
|---|---|
| `server.py` | Flask app. `GET /` renders the UI; `GET /emotionDetector?textToAnalyze=...` returns the analysis sentence. |
| `emotion_detection.py` | `emotion_detector(text)` — Watson NLP call with a labelled local fallback (see below). |
| `__init__.py` | Package marker; exposes `emotion_detector`. |
| `test_emotion_detection.py` | 7 unit tests (5 canonical sentences + blank input + result shape). |
| `templates/index.html` | Web UI (textarea + "Run Sentiment Analysis" button). |
| `static/mywebscript.js` | Calls `/emotionDetector` and displays the response. |
| `requirements.txt` | Flask 3.1.3, requests 2.34.2. |
| `6b_deployment_test.png` | Real headless-Chromium screenshot of the running app after a successful analysis ("I am glad this happened" → dominant emotion joy). |
| `7c_error_handling_interface.png` | Real headless-Chromium screenshot of the running app after submitting blank text → "Invalid text! Please try again!" |
| `SUMMARY.md` | Build/test/run record and what Mark needs next. |

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
python3 server.py
# Open http://127.0.0.1:5000/
```

## How to test

```bash
python3 -m unittest test_emotion_detection.py -v
# 7 tests, all passing (verified 2026-11-06, 0.865s)
```

## Watson API status — IMPORTANT

The primary path calls the IBM Watson NLP Emotion Predict API on Skills
Network (keyless endpoint):

```
POST https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict
Header: grpc-metadata-mm-model-id: emotion_aggregated-workflow_lang_en_stock
```

**That endpoint is unreachable from this VM (verified 2026-11-06):**
`requests.post(...)` raises `ConnectionError: RemoteDisconnected('Remote end
closed connection without response')`, and `curl` returns only the proxy
CONNECT line with no JSON body. No API key is involved — the endpoint is
keyless — it is the endpoint itself that does not respond here.

So `emotion_detector()` falls back to a **deterministic local keyword
scorer** that mimics the Watson response shape exactly (same five keys +
`dominant_emotion`, scores in [0, 1]). The fallback is labelled as such in
the code, in the page footer, and in `SUMMARY.md`. **It is not Watson output
and must not be presented to Coursera/Mark as Watson output.** On the
Coursera Skills Network lab environment the Watson endpoint is expected to
respond and the code will use it automatically (the Watson path is tried
first on every call); the fallback only engages when Watson fails.

## Behaviour

- Valid text → `For the given statement, the system response is 'anger': …,
  'disgust': …, 'fear': …, 'joy': … and 'sadness': …. The dominant emotion
  is ….`
- Blank text → `Invalid text! Please try again!`
