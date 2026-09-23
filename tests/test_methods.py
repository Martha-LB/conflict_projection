import unittest

from conflict_projection.methods import make_concat_method, make_madam_method
from conflict_projection.schemas import Document, Instance


class QueueClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete(self, prompt, *, max_tokens=None):
        self.prompts.append(prompt)
        return next(self.responses)


class MethodTests(unittest.TestCase):
    def test_conflictbank_prompt_contains_all_options(self):
        client = QueueClient(["Option: B."])
        method = make_concat_method(client, "conflictbank")
        instance = Instance(
            question="Which answer?",
            documents=[Document("Evidence")],
            gold_answers=["B"],
            metadata={"options": ["one", "two", "three", "four"]},
        )
        result = method(instance)
        self.assertEqual(result.text, "Option: B.")
        self.assertIn("A. one", client.prompts[0])
        self.assertIn("D. four", client.prompts[0])

    def test_madam_returns_final_round_aggregation(self):
        client = QueueClient(
            [
                "Answer: old-a. Explanation: first.",
                "Answer: old-b. Explanation: first.",
                "All Correct Answers: [old]. Explanation: initial aggregation.",
                "Answer: new. Explanation: revised.",
                "Answer: new. Explanation: revised.",
                "All Correct Answers: [new]. Explanation: final aggregation.",
            ]
        )
        method = make_madam_method(client, "qacc", rounds=2)
        instance = Instance(
            question="Question?",
            documents=[Document("Document A"), Document("Document B")],
            gold_answers=["new"],
        )
        result = method(instance)
        self.assertIn("[new]", result.text)
        self.assertEqual(result.metadata["rounds_run"], 2)


if __name__ == "__main__":
    unittest.main()
