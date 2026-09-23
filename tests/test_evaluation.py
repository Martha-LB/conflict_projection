import unittest

from conflict_projection.evaluation import run_evaluation, subset_result
from conflict_projection.schemas import Document, Instance, MethodOutput
from conflict_projection.scoring import present_qacc


class EvaluationTests(unittest.TestCase):
    def test_denominator_uses_actual_number_of_items(self):
        instances = [
            Instance(
                question=f"Question {index}",
                documents=[Document("Context")],
                gold_answers=["yes"],
                instance_id=str(index),
            )
            for index in range(2)
        ]

        def method(instance):
            return MethodOutput("All Correct Answers: [yes]. Explanation: evidence.")

        result = run_evaluation(instances, method, "test", n=10, matcher=present_qacc)
        self.assertEqual(result.summary["n"], 2)
        self.assertEqual(result.summary["exact"], 1.0)
        subset = subset_result(result, ["1"])
        self.assertEqual(subset.summary["n"], 1)
        self.assertEqual(subset.items[0].instance_id, "1")


if __name__ == "__main__":
    unittest.main()
