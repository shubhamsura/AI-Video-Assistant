import unittest
from engine.config import config
from engine.models import MeetingAnalysisResult
from helpers.exporter import export_markdown_report, export_json_report


class TestVideoAssistantSuite(unittest.TestCase):
    def setUp(self):
        self.mock_data = {
            "title": "Quarterly Technical Architecture Review",
            "summary": "- Scalability review\n- Vector search adoption",
            "action_items": "1. Task to optimize audio processing.",
            "key_decisions": "1. Migrate vector database to Chroma.",
            "open_questions": "1. Evaluate STT accuracy on low-bitrate streams.",
            "transcript": "Welcome to our technical architecture sync. Today we evaluate AI processing pipelines...",
        }

    def test_app_config(self):
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.COLLECTION_NAME, "meeting_transcript_collection")

    def test_meeting_analysis_model(self):
        result_obj = MeetingAnalysisResult(
            title=self.mock_data["title"],
            transcript=self.mock_data["transcript"],
            summary=self.mock_data["summary"],
            action_items=self.mock_data["action_items"],
            key_decisions=self.mock_data["key_decisions"],
            open_questions=self.mock_data["open_questions"],
        )
        self.assertGreater(result_obj.word_count, 0)
        self.assertEqual(result_obj.title, "Quarterly Technical Architecture Review")

    def test_markdown_exporter(self):
        md_output = export_markdown_report(self.mock_data)
        self.assertIn("# 🎬 Quarterly Technical Architecture Review", md_output)
        self.assertIn("## 📋 Executive Summary", md_output)

    def test_json_exporter(self):
        json_output = export_json_report(self.mock_data)
        self.assertIn('"engine": "AI Video Assistant"', json_output)
        self.assertIn('"Quarterly Technical Architecture Review"', json_output)


if __name__ == "__main__":
    unittest.main()
