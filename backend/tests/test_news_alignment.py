import unittest
from datetime import datetime, timezone

from backend.news_alignment import (
    align_articles_to_sessions,
    canonical_article_reference,
    filter_articles_as_of,
    parse_publication_time,
    session_close_cutoff,
)
from tools.run_news_alignment_report import build_report


class NewsAlignmentTests(unittest.TestCase):
    def test_publication_time_parser_requires_unambiguous_timezone(self):
        self.assertEqual(parse_publication_time(1793462400).tzinfo, timezone.utc)
        self.assertIsNone(parse_publication_time("2026-10-31T12:00:00"))
        self.assertIsNone(parse_publication_time("2026-10-31"))
        self.assertIsNone(parse_publication_time(None))

    def test_as_of_filter_drops_future_missing_irrelevant_and_duplicate_news(self):
        cutoff = "2026-10-30T20:00:00Z"
        articles = [
            {"title": "AAPL results", "published_at": "2026-10-30T19:00:00Z",
             "source": "Wire", "url": "https://example.com/story?utm_source=x", "matched_query": "AAPL"},
            {"title": "AAPL results", "published_at": "2026-10-30T19:00:00Z",
             "source": "Wire", "url": "https://example.com/story", "matched_query": "AAPL"},
            {"title": "AAPL after close", "published_at": "2026-10-30T20:01:00Z",
             "source": "Wire", "url": "https://example.com/future", "matched_query": "AAPL"},
            {"title": "AAPL unknown timestamp", "source": "Wire", "matched_query": "AAPL"},
            {"title": "Other company", "published_at": "2026-10-30T19:00:00Z",
             "source": "Wire", "url": "https://example.com/other", "matched_query": "MSFT"},
        ]
        accepted, audit = filter_articles_as_of(
            articles, cutoff, symbol="AAPL", relevance_terms=["AAPL", "Apple Inc."],
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["published_at"], "2026-10-30T19:00:00Z")
        self.assertEqual(audit["after_cutoff"], 1)
        self.assertEqual(audit["missing_or_ambiguous_timestamp"], 1)
        self.assertEqual(audit["irrelevant"], 1)
        self.assertEqual(audit["duplicate"], 1)
        self.assertEqual(audit["accepted"], 1)

    def test_exact_cutoff_is_eligible_and_naive_cutoff_is_rejected(self):
        article = {"title": "AAPL close", "published_at": "2026-10-30T20:00:00Z",
                   "source": "Wire", "matched_query": "AAPL"}
        accepted, _ = filter_articles_as_of([article], datetime(2026, 10, 30, 20, tzinfo=timezone.utc),
                                           symbol="AAPL", relevance_terms=["AAPL"])
        self.assertEqual(len(accepted), 1)
        with self.assertRaises(ValueError):
            filter_articles_as_of([article], datetime(2026, 10, 30, 20), symbol="AAPL")

    def test_session_alignment_uses_new_york_close_and_reports_missing_dates(self):
        sessions = ["2026-03-09", "2026-03-10"]
        self.assertEqual(session_close_cutoff(sessions[0]).isoformat(), "2026-03-09T20:00:00+00:00")
        articles = [
            {"title": "AAPL pre-close", "published_at": "2026-03-09T19:59:00Z",
             "source": "Wire", "url": "https://example.com/one", "matched_query": "AAPL"},
            {"title": "AAPL after first close", "published_at": "2026-03-09T20:01:00Z",
             "source": "Wire", "url": "https://example.com/two", "matched_query": "AAPL"},
            {"title": "AAPL after final close", "published_at": "2026-03-10T20:01:00Z",
             "source": "Wire", "url": "https://example.com/three", "matched_query": "AAPL"},
        ]
        report = align_articles_to_sessions(sessions, articles, symbol="AAPL", relevance_terms=["AAPL"])
        self.assertEqual([row["article_count"] for row in report["sessions"]], [1, 1])
        self.assertEqual(report["sessions"][0]["articles"][0]["article_id"], "https://example.com/one")
        self.assertEqual(report["sessions"][1]["articles"][0]["article_id"], "https://example.com/two")
        self.assertEqual(report["coverage"]["missing_dates"], [])
        self.assertEqual(report["coverage"]["after_last_cutoff"], 1)

    def test_reference_removes_tracking_parameters_and_hashes_missing_urls(self):
        left = canonical_article_reference("https://EXAMPLE.com/story?utm_campaign=a&id=4#top",
                                          title="AAPL", source="Wire", published_at="2026-01-01T12:00:00Z")
        right = canonical_article_reference("https://example.com/story?id=4",
                                           title="Different title", source="Other", published_at="2026-01-02T12:00:00Z")
        opaque1 = canonical_article_reference(None, title="AAPL update", source="Wire", published_at="2026-01-01T12:00:00Z")
        opaque2 = canonical_article_reference(None, title="AAPL update", source="Wire", published_at="2026-01-01T12:00:00Z")
        self.assertEqual(left, right)
        self.assertEqual(opaque1, opaque2)
        self.assertTrue(opaque1.startswith("sha256:"))

    def test_unavailable_archive_is_reported_as_unknown_coverage_without_ablation(self):
        report = build_report()
        coverage = report["historical_coverage"]
        self.assertEqual(report["status"], "historical_corpus_unavailable_no_ablation")
        self.assertFalse(report["ablation"]["run"])
        self.assertIsNone(coverage["coverage_fraction"])
        self.assertIsNone(coverage["missing_news_dates"])
        self.assertFalse(report["ablation"]["today_headlines_substituted_for_history"])


if __name__ == "__main__":
    unittest.main()
