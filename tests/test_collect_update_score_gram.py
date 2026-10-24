import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT=Path(__file__).parents[1]/'scripts'/'collect_update_score_gram.py'
spec=importlib.util.spec_from_file_location('eesd_collect_update_score_gram',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Args:
    dry_run=True
    input=None


def test_dry_run_rejects_hidden_labels(tmp_path):
    path=tmp_path/'bad.jsonl'
    row={'source_id':'a','prompt_ids':[1,2],'candidate_ids':[[3],[4]],
         'before':[0,0],'after':[[1,0],[0,1]],'hidden_label':1}
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='exact fields'):
        module.read_bank(path)


def test_saved_token_bank_is_validated_without_transformers(tmp_path):
    path=tmp_path/'input.jsonl'
    row={'source_id':'a','prompt_ids':[1,2],'candidate_ids':[[3],[4,5]],
         'before':[0,0],'after':[[1,0],[0,1]]}
    path.write_text(json.dumps(row)+'\n')
    args=Args()
    args.input=str(path)
    result=module.score_bank(args)
    assert result['status']=='validated_without_model_or_gpu'
    assert result['sources']==1 and result['max_sequence_tokens']==4


def test_input_bank_rejects_duplicated_ids_or_wrong_response_order(tmp_path):
    path=tmp_path/'input.jsonl'
    row={'source_id':'a','prompt_ids':[1,2],'candidate_ids':[[3],[4]],
         'before':[0,0],'after':[[1,0],[0,1]]}
    path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='duplicate'):
        module.read_bank(path)
    row['after']=[[1,0]]
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='disagree'):
        module.read_bank(path)
