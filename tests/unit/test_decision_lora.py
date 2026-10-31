import json

import pytest

from minimol.decision_lora import (
    SPLITS,
    CompletionCollator,
    encode_example,
    generate_policy_benchmark,
    load_records,
    messages,
    score_predictions,
)


def test_group_contract_and_labels_do_not_enter_prompt(tmp_path):
    path = tmp_path / 'benchmark.jsonl'
    generate_policy_benchmark(path, n_groups=40)
    records = load_records(path, required_splits=SPLITS)
    assert len(records) == 120
    assert len({r.group_id for r in records}) == 40
    prompt = messages(records[0])
    assert 'group_id' not in prompt[1]['content']
    changed = records[1].model_copy(update={'split': 'test'})
    path.write_text(path.read_text() + json.dumps(changed.model_copy(update={'id': 'other'}).model_dump()) + '\n')
    with pytest.raises(ValueError, match='crosses splits'):
        load_records(path)


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]

    def encode(self, text, **kwargs):
        return [4, 5]  # An answer need not be a single tokenizer token.


def test_completion_mask_and_no_silent_truncation(tmp_path):
    path = tmp_path / 'data.jsonl'
    generate_policy_benchmark(path, 40)
    record = load_records(path)[0]
    item = encode_example(Tokenizer(), record, 6)
    assert item['labels'] == [-100, -100, -100, 4, 5, 9]
    with pytest.raises(ValueError, match='truncation'):
        encode_example(Tokenizer(), record, 5)
    shorter = {k: v[:-1] for k, v in item.items()}
    batch = CompletionCollator(0)([item, shorter])
    assert batch['labels'][1, -1] == -100
    assert batch['attention_mask'][1, -1] == 0


def test_metrics_reject_missing_ids_or_invalid_distributions(tmp_path):
    path = tmp_path / 'data.jsonl'
    generate_policy_benchmark(path, 40)
    records = load_records(path)
    predictions = [{'id': r.id, 'codes': [o.code for o in r.options],
                    'probabilities': [float(o.code == r.answer) for o in r.options]} for r in records]
    result = score_predictions(records, predictions)
    assert all(v['accuracy'] == 1 and v['multiclass_brier'] == 0 for v in result.values())
    with pytest.raises(ValueError, match='every evaluation id'):
        score_predictions(records, predictions[:-1])
    predictions[0]['probabilities'][0] = float('nan')
    with pytest.raises(ValueError, match='probabilities'):
        score_predictions(records, predictions)


def test_policy_and_temperature_use_only_calibration(tmp_path):
    from minimol.decision_lora import baseline_predictions, fit_temperature

    path = tmp_path / 'data.jsonl'
    generate_policy_benchmark(path, 40)
    records = load_records(path)
    predictions = baseline_predictions(records, 'policy')
    assert all(v['accuracy'] == 1 for v in score_predictions(records, predictions).values())
    with pytest.raises(ValueError, match='calibration split'):
        fit_temperature(records, predictions)
    calibration = [r for r in records if r.split == 'calibration']
    temperature = fit_temperature(calibration, baseline_predictions(calibration, 'uniform'))
    assert temperature > 0


def test_multitoken_option_scoring_normalizes_without_gold_leak(tmp_path):
    import torch

    from minimol.decision_lora import option_probabilities

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 4)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask):
            from types import SimpleNamespace
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 10))

    path = tmp_path / 'data.jsonl'
    generate_policy_benchmark(path, 40)
    record = load_records(path)[0]
    p = option_probabilities(Model(), Tokenizer(), record, 6)
    assert p == pytest.approx([1 / len(record.options)] * len(record.options))
