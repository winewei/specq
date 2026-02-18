"""Tests for vote aggregation strategies."""

import pytest
from specq.aggregator import aggregate_votes
from specq.models import VoteResult


# --- skip strategy ---

def test_skip_always_approved():
    """skip: unconditionally approved."""
    decision, findings = aggregate_votes([], "skip", "low")
    assert decision == "approved"


def test_skip_ignores_fail_votes():
    """skip: approved even with fail votes."""
    votes = [VoteResult(voter="a", verdict="fail", confidence=1.0,
                        findings=[{"severity": "critical", "description": "bad"}], summary="")]
    decision, _ = aggregate_votes(votes, "skip", "high")
    assert decision == "approved"


# --- majority strategy ---

def test_majority_2_of_3_pass():
    """majority: 2/3 → approved."""
    votes = [
        VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary=""),
        VoteResult(voter="b", verdict="pass", confidence=0.8, findings=[], summary=""),
        VoteResult(voter="c", verdict="fail", confidence=0.7, findings=[], summary=""),
    ]
    decision, _ = aggregate_votes(votes, "majority", "medium")
    assert decision == "approved"


def test_majority_1_of_3_rejected():
    """majority: 1/3 → rejected."""
    votes = [
        VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary=""),
        VoteResult(voter="b", verdict="fail", confidence=0.8, findings=[], summary=""),
        VoteResult(voter="c", verdict="fail", confidence=0.7, findings=[], summary=""),
    ]
    decision, _ = aggregate_votes(votes, "majority", "medium")
    assert decision == "rejected"


def test_majority_50_percent_rejected():
    """majority: 2/4 = 50% → rejected (need more than half)."""
    votes = [
        VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary=""),
        VoteResult(voter="b", verdict="pass", confidence=0.8, findings=[], summary=""),
        VoteResult(voter="c", verdict="fail", confidence=0.7, findings=[], summary=""),
        VoteResult(voter="d", verdict="fail", confidence=0.6, findings=[], summary=""),
    ]
    decision, _ = aggregate_votes(votes, "majority", "medium")
    assert decision == "rejected"


def test_majority_single_voter_pass():
    """majority: 1/1 pass → approved."""
    votes = [VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary="")]
    decision, _ = aggregate_votes(votes, "majority", "medium")
    assert decision == "approved"


# --- unanimous strategy ---

def test_unanimous_all_pass():
    """unanimous: all pass."""
    votes = [
        VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary=""),
        VoteResult(voter="b", verdict="pass", confidence=0.8, findings=[], summary=""),
    ]
    decision, _ = aggregate_votes(votes, "unanimous", "medium")
    assert decision == "approved"


def test_unanimous_one_fail_rejected():
    """unanimous: one fail → rejected."""
    votes = [
        VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary=""),
        VoteResult(voter="b", verdict="fail", confidence=0.3,
                   findings=[{"severity": "critical", "description": "违反规范"}], summary=""),
    ]
    decision, findings = aggregate_votes(votes, "unanimous", "medium")
    assert decision == "rejected"
    assert len(findings) >= 1


# --- Risk escalation ---

def test_critical_finding_upgrades_to_needs_review():
    """majority pass + critical finding → needs_review."""
    votes = [
        VoteResult(voter="a", verdict="pass", confidence=0.9,
                   findings=[{"severity": "critical", "category": "spec",
                              "description": "严重问题"}], summary=""),
        VoteResult(voter="b", verdict="pass", confidence=0.8, findings=[], summary=""),
    ]
    decision, _ = aggregate_votes(votes, "majority", "medium")
    assert decision == "needs_review"


def test_high_risk_pass_upgrades_to_needs_review():
    """high risk + majority pass → needs_review."""
    votes = [
        VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary=""),
        VoteResult(voter="b", verdict="pass", confidence=0.8, findings=[], summary=""),
    ]
    decision, _ = aggregate_votes(votes, "majority", "high")
    assert decision == "needs_review"


def test_low_risk_stays_approved():
    """low risk + pass → approved."""
    votes = [VoteResult(voter="a", verdict="pass", confidence=0.9, findings=[], summary="")]
    decision, _ = aggregate_votes(votes, "majority", "low")
    assert decision == "approved"


# --- Edge cases ---

def test_empty_votes_non_skip_rejected():
    """Non-skip with no votes → rejected."""
    decision, _ = aggregate_votes([], "majority", "medium")
    assert decision == "rejected"


def test_all_findings_collected():
    """All findings from all voters are merged."""
    votes = [
        VoteResult(voter="a", verdict="fail", confidence=0.5,
                   findings=[{"severity": "critical", "description": "f1"},
                             {"severity": "warning", "description": "f2"}], summary=""),
        VoteResult(voter="b", verdict="fail", confidence=0.4,
                   findings=[{"severity": "info", "description": "f3"}], summary=""),
    ]
    _, findings = aggregate_votes(votes, "majority", "medium")
    assert len(findings) == 3
