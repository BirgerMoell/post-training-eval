import unittest

from post_training_eval.generation_canary import _unique_ratio


class GenerationCanaryTests(unittest.TestCase):
    def test_unique_ratio_detects_repetition(self):
        self.assertEqual(_unique_ratio([]), 1.0)
        self.assertEqual(_unique_ratio([1, 2, 3, 4]), 1.0)
        self.assertEqual(_unique_ratio([1, 1, 1, 1]), 0.25)


if __name__ == "__main__":
    unittest.main()
