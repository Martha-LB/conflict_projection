import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from conflict_projection.datasets import (
    download_qacc,
    extract_domain,
    extract_publication_date,
    load_qacc,
)


class DatasetUtilityTests(unittest.TestCase):
    def test_qacc_downloader_writes_and_reuses_valid_file(self):
        payload = json.dumps(
            [{"question": "Q?", "contexts": ["Context"], "correctAnswer": "A"}]
        ).encode("utf-8")
        response = io.BytesIO(payload)
        with tempfile.TemporaryDirectory() as directory:
            with patch("conflict_projection.datasets.urllib.request.urlopen", return_value=response) as mocked:
                first = download_qacc(Path(directory), revision="test-commit")
                second = download_qacc(Path(directory), revision="test-commit")
        self.assertEqual(first, second)
        self.assertEqual(mocked.call_count, 1)

    def test_domain_removes_only_exact_www_prefix(self):
        self.assertEqual(extract_domain("https://www.example.com/a"), "example.com")
        self.assertEqual(extract_domain("https://weather.com/a"), "weather.com")
        self.assertEqual(extract_domain("wow.com"), "wow.com")

    def test_date_must_be_at_start_of_snippet(self):
        self.assertEqual(extract_publication_date("Mar 14, 2023 — story"), "2023-03-14")
        self.assertIsNone(extract_publication_date("The event occurred on Mar 14, 2023"))

    def test_qacc_filters_requested_split(self):
        rows = [
            {
                "question": "dev question",
                "contexts": ["A context"],
                "sources": ["https://example.com"],
                "correctAnswer": "A",
                "split": "dev",
            },
            {
                "question": "test question",
                "contexts": ["B context"],
                "sources": ["https://example.org"],
                "correctAnswer": "B",
                "split": "test",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qacc.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            instances = load_qacc(path, split="test")
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].question, "test question")

    def test_qacc_a_means_conflict_and_b_means_no_conflict(self):
        rows = [
            {
                "question": "conflict",
                "contexts": ["Context"],
                "sources": ["https://example.com"],
                "correctAnswer": "A",
                "secondAnswerExist": "A",
                "secondAnswer": "B",
                "split": "test",
            },
            {
                "question": "no conflict",
                "contexts": ["Context"],
                "sources": ["https://example.com"],
                "correctAnswer": "A",
                "secondAnswerExist": "B",
                "secondAnswer": "",
                "split": "test",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qacc.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            instances = load_qacc(path, split="test")
        self.assertTrue(instances[0].has_conflict)
        self.assertFalse(instances[1].has_conflict)

    def test_qacc_uses_official_ambigqa_references_not_annotation_label(self):
        rows = [{
            "question": "Who created it?",
            "contexts": ["The United Nations created it."],
            "sources": ["https://example.com"],
            "ambigqa_answer": ["United Nations", "the United Nations"],
            "firstAnswer": "the United Nations",
            "correctAnswer": "most common",
            "secondAnswerExist": "B",
            "split": "test",
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qacc.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            instance = load_qacc(path, split="test")[0]
        self.assertEqual(instance.gold_answers, ["United Nations", "the United Nations"])
        self.assertEqual(instance.metadata["annotator_correct_answer"], "most common")


if __name__ == "__main__":
    unittest.main()
