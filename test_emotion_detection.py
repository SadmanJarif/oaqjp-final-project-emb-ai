"""Unit tests for emotion_detector — IBM Course 8 final project.

These are the five canonical test sentences from the IBM lab:
joy, anger, disgust, fear, sadness. Each asserts the dominant emotion.
Plus a blank-input test (all None) and a shape test.

Note: the tests assert the dominant emotion only, so they pass whether
the scores come from the live Watson NLP API or from the labelled local
fallback used when Watson is unreachable (see emotion_detection.py).
Run:  python3 -m unittest test_emotion_detection.py -v
"""

import unittest

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    def test_joy(self):
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger(self):
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust(self):
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_fear(self):
        result = emotion_detector(
            "I am so scared that I am going to fail this exam"
        )
        self.assertEqual(result["dominant_emotion"], "fear")

    def test_sadness(self):
        result = emotion_detector("I feel sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_blank_input_returns_none(self):
        result = emotion_detector("")
        self.assertIsNone(result["dominant_emotion"])
        for key in ("anger", "disgust", "fear", "joy", "sadness"):
            self.assertIsNone(result[key])

    def test_result_shape(self):
        result = emotion_detector("I am glad this happened")
        for key in ("anger", "disgust", "fear", "joy", "sadness"):
            self.assertIsInstance(result[key], float)
            self.assertGreaterEqual(result[key], 0)
            self.assertLessEqual(result[key], 1)


if __name__ == "__main__":
    unittest.main()
