"""Flask server for the Emotion Detector — IBM Course 8 final project.

Routes:
    GET  /                  -> renders templates/index.html
    GET  /emotionDetector?textToAnalyze=<text>
                            -> "For the given statement, the system response
                               is 'anger': ..., 'disgust': ..., 'fear': ...,
                               'joy': ... and 'sadness': .... The dominant
                               emotion is ...."
                               Blank/invalid text -> "Invalid text! Please
                               try again!"

Run:  python3 server.py   (serves on http://127.0.0.1:5000)
"""

from flask import Flask, render_template, request

from emotion_detection import emotion_detector

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def emotion_detector_route():
    """Return the emotion analysis for the supplied text."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    response = emotion_detector(text_to_analyze)
    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again!"
    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, "
        f"'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


@app.route("/")
def render_index_page():
    """Render the home page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
