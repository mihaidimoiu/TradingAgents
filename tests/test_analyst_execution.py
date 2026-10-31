import unittest

from tradingagents.graph.analyst_execution import (
    build_analyst_execution_plan,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "market"])

        self.assertEqual([spec.key for spec in plan.specs], ["news", "market"])
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].report_key, "news_report")
        self.assertFalse(hasattr(plan.specs[0], "clear_node"))

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market", "options"])

    def test_macro_files_its_own_report_with_the_macro_tools(self):
        spec = build_analyst_execution_plan(["macro"]).specs[0]
        self.assertEqual(spec.agent_node, "Macro Analyst")
        self.assertEqual(spec.report_key, "macro_report")
        self.assertEqual([t.name for t in spec.tools],
                         ["get_macro_indicators", "get_prediction_markets", "calculate"])
        news = build_analyst_execution_plan(["news"]).specs[0]
        self.assertEqual([t.name for t in news.tools], ["get_news", "get_global_news", "calculate"])

    def test_social_key_displays_as_sentiment_analyst(self):
        # The wire key stays "social" for saved-config back-compat, but the
        # user-visible agent_node label must match the v0.2.5 rename so the
        # wall-time summary and any future consumer of agent_node says
        # "Sentiment Analyst" rather than the legacy "Social Analyst".
        plan = build_analyst_execution_plan(["social"])
        spec = plan.specs[0]
        self.assertEqual(spec.key, "social")
        self.assertEqual(spec.agent_node, "Sentiment Analyst")
        self.assertEqual(spec.report_key, "sentiment_report")
