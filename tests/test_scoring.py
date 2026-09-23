import unittest

from conflict_projection.scoring import (
    parse_answers,
    parse_option,
    present_option,
    present_qacc,
    score_predictions,
)


class ScoringTests(unittest.TestCase):
    def test_extra_prediction_fails_exact_match(self):
        score = score_predictions(["Paris", "London"], ["Paris"], matcher=present_qacc)
        self.assertFalse(score.exact)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.precision, 0.5)

    def test_empty_gold_is_not_automatically_correct(self):
        score = score_predictions([], [])
        self.assertFalse(score.exact)

    def test_single_qacc_answer_is_exact(self):
        score = score_predictions(["The Eiffel Tower"], ["Eiffel Tower"], matcher=present_qacc)
        self.assertTrue(score.exact)

    def test_answer_parser(self):
        parsed = parse_answers('All Correct Answers: ["1963", "1956"]. Explanation: evidence')
        self.assertEqual(parsed, ["1963", "1956"])

    def test_conflictbank_option_parser_and_score(self):
        predictions = parse_option("Option: C.")
        self.assertEqual(predictions, ["C"])
        self.assertTrue(score_predictions(predictions, ["C"], matcher=present_option).exact)


if __name__ == "__main__":
    unittest.main()
