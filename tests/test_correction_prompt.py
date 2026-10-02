import copy
import hashlib
import json
from pathlib import Path

import pytest

from eesd.correction_prompt import prepare_correction_prompt, user_prompt


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['tokenize'] is False
        return json.dumps(messages)

    def __call__(self, text, **kwargs):
        assert kwargs['return_token_type_ids'] is False
        assert kwargs['add_special_tokens'] is False
        return {'input_ids': list(text.encode())}


def row():
    return {'schema': 'eesd-public-correction-row-v1', 'task_id': 'task',
            'task_text': 'Fix this', 'candidate': 'print(input())',
            'tests': [{'id': str(i), 'input': 'x', 'expected': 'x', 'actual': 'y',
                       'stderr': '', 'outcome': 'WRONG_OUTPUT', 'expected_error': None}
                      for i in range(4)]}


def test_normal_prompt_and_hash_are_exact():
    item = row()
    result = prepare_correction_prompt(Tokenizer(), item, 'rbr')
    assert result['prompt'] == user_prompt(item, 'rbr')
    assert result['messages'][-1]['content'] == result['prompt']
    assert result['prompt_metadata']['truncated_fields'] == []
    assert result['prompt_token_ids_sha256'] == hashlib.sha256(
        json.dumps(result['input_ids'], separators=(',', ':')).encode()).hexdigest()


def test_large_fields_clip_without_changing_full_program_slots_or_source():
    item = row()
    item['task_text'] = 'a' * 1_200_000
    for test in item['tests']:
        test.update(input={'value': 'b' * 10000}, expected='c' * 10000,
                    actual='d' * 10000, stderr='e' * 10000)
    original = copy.deepcopy(item)
    result = prepare_correction_prompt(Tokenizer(), item, 'codearc')
    assert result['prompt_metadata']['input_tokens'] <= 4096
    payload = json.loads(result['prompt'].split('\n', 1)[1])
    assert payload['program'] == original['candidate']
    assert len(payload['public_executions']) == 4
    for before, after in zip(original['tests'], payload['public_executions']):
        for key in ('id', 'outcome', 'expected_error'):
            assert after[key] == before[key]
        for key in ('input', 'expected', 'actual', 'stderr'):
            assert 'PUBLIC TEXT TRUNCATED' in after[key]
    assert item == original


def test_full_program_cannot_be_truncated():
    item = row()
    item['candidate'] = 'x' * 10000
    with pytest.raises(ValueError, match='task.*cannot fit'):
        prepare_correction_prompt(Tokenizer(), item, 'rbr')


def test_four_slots_required():
    item = row()
    item['tests'].pop()
    with pytest.raises(ValueError, match='four'):
        prepare_correction_prompt(Tokenizer(), item, 'rbr')

