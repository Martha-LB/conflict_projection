import unittest

from conflict_projection.scoring import (
    parse_answers,
    parse_qacc_answer,
    parse_option,
    present_option,
    present_qacc,
    score_predictions,
    score_qacc_predictions,
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

    def test_qacc_matches_any_reference_alias(self):
        score = score_qacc_predictions(
            ["L. K. Advani"], ["7th Deputy Prime Minister of India", "L. K. Advani"]
        )
        self.assertTrue(score.exact)

    def test_qacc_parser_preserves_commas_in_single_answer(self):
        parsed = parse_qacc_answer(
            "All Correct Answers: [July 4, 1776]. Explanation: evidence"
        )
        self.assertEqual(parsed, ["July 4, 1776"])

    def test_answer_parser(self):
        parsed = parse_answers('All Correct Answers: ["1963", "1956"]. Explanation: evidence')
        self.assertEqual(parsed, ["1963", "1956"])

    def test_conflictbank_option_parser_and_score(self):
        predictions = parse_option("Option: C.")
        self.assertEqual(predictions, ["C"])
        self.assertTrue(score_predictions(predictions, ["C"], matcher=present_option).exact)


if __name__ == "__main__":
    unittest.main()
