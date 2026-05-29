import pytest
from ph_economic_ai.engine.quality_scorer import QualityScorer
from ph_economic_ai.engine.debate import AgentResponse


def _resp(name, statement, estimate=1.0, round_num=1):
    return AgentResponse(
        agent_name=name, round_num=round_num,
        thinking='', statement=statement, price_estimate=estimate,
    )


def test_citation_count_high():
    r = _resp('A', 'Brent at $72.40 and USD/PHP at ₱57.80 suggests ₱1.20 rise.')
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['citation_score'] >= 0.6


def test_citation_count_zero():
    r = _resp('A', 'Prices are expected to rise significantly.')
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['citation_score'] == 0.0


def test_causal_chain_full():
    stmt = ('Analysis.\nCAUSAL CHAIN: oil shock → import cost → pump price → household budget\n'
            'ESTIMATE: +₱1.00/L')
    r = _resp('A', stmt)
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['chain_score'] == 1.0


def test_causal_chain_full_ascii_arrows():
    stmt = ('Analysis.\nCAUSAL CHAIN: oil shock -> import cost -> pump price -> household budget\n'
            'ESTIMATE: +P1.00/L')
    r = _resp('A', stmt)
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['chain_score'] == 1.0


def test_causal_chain_missing():
    r = _resp('A', 'Prices go up. ESTIMATE: +₱1.00/L')
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['chain_score'] == 0.0


def test_convergence_on_median():
    responses = [
        _resp('A', 'text', estimate=1.0),
        _resp('B', 'text', estimate=1.0),
        _resp('C', 'text', estimate=1.0),
    ]
    result = QualityScorer.score_responses(responses, group_estimates=[1.0, 1.0, 1.0])
    assert result['A']['convergence_score'] == 1.0


def test_convergence_outlier():
    responses = [_resp('A', 'text', estimate=5.0)]
    result = QualityScorer.score_responses(responses, group_estimates=[1.0, 1.0, 5.0])
    assert result['A']['convergence_score'] < 0.5


def test_overall_score_in_range():
    stmt = ('Brent $72.40, USD/PHP ₱57.80. CAUSAL CHAIN: oil → cost → price → consumer. '
            'ESTIMATE: +₱1.20/L')
    r = _resp('A', stmt)
    result = QualityScorer.score_responses([r], group_estimates=[1.2])
    assert 0.0 <= result['A']['overall'] <= 1.0


def test_causal_chain_partial():
    stmt = 'Analysis. CAUSAL CHAIN: oil → price. ESTIMATE: +₱1.00/L'
    r = _resp('A', stmt)
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['chain_score'] == 0.5


def test_citation_single_digit_pct():
    r = _resp('A', 'Oil demand dropped 5% while USD weakened 3%.')
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['citation_count'] == 2


def test_run_quality_average():
    responses = [
        _resp('A', 'Brent $72.40. CAUSAL CHAIN: a → b → c → d. ESTIMATE: +₱1.00/L', estimate=1.0),
        _resp('B', 'Prices rise. ESTIMATE: +₱1.00/L', estimate=1.0),
    ]
    quality = QualityScorer.run_quality(responses, group_estimates=[1.0, 1.0])
    assert 0.0 <= quality <= 1.0
