"""Tests for claims_evidence.py — claims-vs-evidence table extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.agents.nodes.claims_evidence import (
    ClaimEvidence,
    ClaimsEvidenceExtraction,
    extract_claims_evidence_node,
)


def _mock_llm_instance(extraction: ClaimsEvidenceExtraction) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.return_value = extraction
    mock_llm.invoke.return_value = extraction
    return mock_llm


class TestExtractClaimsEvidenceNode:
    def test_skips_when_no_breakthroughs_or_results(self):
        result = extract_claims_evidence_node({"deep_analysis": {}})
        assert result["claims_evidence"] == []
        assert result["current_step"] == "claims_evidence_skipped"

    def test_extracts_claims_with_evidence_status(self):
        extraction = ClaimsEvidenceExtraction(claims=[
            ClaimEvidence(claim="Beats GPT-4 on MMLU", evidence_status="quantified"),
            ClaimEvidence(claim="Substantially more robust", evidence_status="qualitative"),
            ClaimEvidence(claim="Generalizes to any domain", evidence_status="absent"),
        ])
        with patch("app.agents.nodes.claims_evidence.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(extraction)
            MockLLM.return_value = mock_instance

            result = extract_claims_evidence_node({
                "deep_analysis": {"breakthroughs": "Beats GPT-4 on MMLU by 5%.", "quantitative_results": []},
            })

        statuses = {c["evidence_status"] for c in result["claims_evidence"]}
        assert statuses == {"quantified", "qualitative", "absent"}
        assert result["current_step"] == "claims_evidence_extracted"

    def test_caps_at_six_claims(self):
        extraction = ClaimsEvidenceExtraction(
            claims=[ClaimEvidence(claim=f"Claim {i}", evidence_status="quantified") for i in range(10)]
        )
        with patch("app.agents.nodes.claims_evidence.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(extraction)
            MockLLM.return_value = mock_instance

            result = extract_claims_evidence_node({
                "deep_analysis": {"breakthroughs": "Many claims.", "quantitative_results": []},
            })

        assert len(result["claims_evidence"]) == 6

    def test_llm_failure_returns_empty_gracefully(self):
        with patch("app.agents.nodes.claims_evidence.ChatGoogleGenerativeAI") as MockLLM:
            MockLLM.side_effect = RuntimeError("API down")
            result = extract_claims_evidence_node({
                "deep_analysis": {"breakthroughs": "Some results.", "quantitative_results": []},
            })

        assert result["claims_evidence"] == []
        assert result["current_step"] == "claims_evidence_skipped"

    def test_retries_once_after_transient_failure_then_succeeds(self, monkeypatch):
        from app.agents.nodes import claims_evidence as claims_evidence_module

        monkeypatch.setattr(claims_evidence_module.settings, "llm_schema_max_retries", 2)
        extraction = ClaimsEvidenceExtraction(claims=[
            ClaimEvidence(claim="Recovered claim", evidence_status="quantified"),
        ])
        mock_llm = MagicMock()
        mock_llm.side_effect = [ValueError("schema validation failed"), extraction]
        with patch("app.agents.nodes.claims_evidence.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = mock_llm
            MockLLM.return_value = mock_instance

            result = extract_claims_evidence_node({
                "deep_analysis": {"breakthroughs": "Some results.", "quantitative_results": []},
            })

        assert result["current_step"] == "claims_evidence_extracted"
        assert len(result["claims_evidence"]) == 1
        assert mock_llm.call_count == 2

    def test_claim_text_is_latex_normalized(self):
        extraction = ClaimsEvidenceExtraction(claims=[
            ClaimEvidence(claim="Achieves $O(n^2)$ scaling", evidence_status="quantified"),
        ])
        with patch("app.agents.nodes.claims_evidence.ChatGoogleGenerativeAI") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.with_structured_output.return_value = _mock_llm_instance(extraction)
            MockLLM.return_value = mock_instance

            result = extract_claims_evidence_node({
                "deep_analysis": {"breakthroughs": "text", "quantitative_results": []},
            })

        assert "$" not in result["claims_evidence"][0]["claim"]
