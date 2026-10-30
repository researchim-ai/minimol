import pytest
import torch

from minimol.decide import RegDecider, RegDecideRequest
from minimol.models.reg_cnn import MinimolReg
from minimol.models.reg_decide import RegDecisionHead


@pytest.fixture
def decider():
    torch.manual_seed(1)
    encoder = MinimolReg('1m', channels=8, n_blocks=1, dropout=0)
    head = RegDecisionHead(8, 3, 16)
    return RegDecider({
        'format_version': 1,
        'encoder_metadata': {'preset': '1m', 'n_assays': 3, 'head_kind': 'nonlinear',
                             'padding_policy': 'mask_aware', 'overrides': {'channels': 8, 'n_blocks': 1, 'dropout': 0},
                             'length_min': 4, 'length_max': 32, 'scaler': {'mean': [0, 0, 0], 'std': [1, 1, 1]}},
        'encoder_state': encoder.state_dict(), 'head_state': head.state_dict(),
        'head_metadata': {'feature_dim': 8, 'n_assays': 3, 'hidden': 16},
        'assays': ['K562', 'HepG2', 'SK-N-SH'], 'temperature': 1.,
        'baseline_temperature': 1., 'baseline_scale': [.4, .4, .4],
        'ood_rms_z_threshold': 1e9, 'calibration_version': 'test', 'calibration_scope': 'test', 'gate': 'GO',
    }, backend='trained')


def test_sdk_reuses_encoder_and_returns_coherent_typed_answers(decider):
    count = {'calls': 0}
    hook = decider.encoder.stem.register_forward_hook(lambda *args: count.__setitem__('calls', count['calls'] + 1))
    result = decider.predict({'candidates': [{'id': 'x', 'sequence': 'ACGTACGT'}],
                             'questions': [
                                 {'id': 'low', 'kind': 'noul', 'candidate_id': 'x', 'assay': 'K562', 'threshold': 0},
                                 {'id': 'high', 'kind': 'noul', 'candidate_id': 'x', 'assay': 'K562', 'threshold': 1},
                                 {'id': 'score', 'kind': 'score', 'candidate_id': 'x', 'assay': 'K562', 'bins': [0, 1]},
                             ]})
    hook.remove()
    assert count['calls'] == 1
    a, b, score = result['answers']
    assert a['probability'] >= b['probability']
    assert abs(sum(score['distribution']) - 1) < 1e-6
    assert score['distribution'][1] == pytest.approx(a['probability'] - b['probability'], abs=1e-6)


def test_choice_permutation_and_unsupported_domain(decider):
    state = {'candidates': [{'id': 'a', 'sequence': 'ACGT'}, {'id': 'b', 'sequence': 'TTTTACGT'}],
             'questions': [
                 {'id': 'ab', 'kind': 'candidate_choice', 'candidate_ids': ['a', 'b'], 'assay': 'K562'},
                 {'id': 'ba', 'kind': 'candidate_choice', 'candidate_ids': ['b', 'a'], 'assay': 'K562'},
                 {'id': 'unknown', 'kind': 'noul', 'candidate_id': 'a', 'assay': 'HeLa', 'threshold': 0},
             ]}
    a, b, rejected = decider.predict(state)['answers']
    assert a['distribution'] == pytest.approx(b['distribution'][::-1])
    assert rejected['abstain'] and rejected['reason'] == 'unsupported_assay'


def test_schemas_reject_invalid_questions_and_duplicate_ids():
    with pytest.raises(ValueError, match='threshold'):
        RegDecideRequest.model_validate({'candidates': [{'id': 'x', 'sequence': 'ACGT'}],
                                         'questions': [{'id': 'q', 'kind': 'noul', 'candidate_id': 'x', 'assay': 'K562'}]})
