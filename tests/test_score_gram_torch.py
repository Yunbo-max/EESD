import numpy as np
import pytest

torch = pytest.importorskip('torch')
from types import SimpleNamespace
from eesd.score_gram_torch import score_gram_for_rollouts, teacher_forced_sequence_logprob


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(17,5)
        self.head=torch.nn.Linear(5,17,bias=False)
    def get_input_embeddings(self):
        return self.embedding
    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def test_sequence_score_uses_all_response_tokens_and_correct_shift():
    torch.manual_seed(111)
    model=TinyCausalLM().eval()
    ids=torch.tensor([[2,3,4,5]],dtype=torch.long)
    logits=model(ids).logits[0].log_softmax(-1)
    expected=logits[1,4]+logits[2,5]
    obtained=teacher_forced_sequence_logprob(model,[2,3],[4,5])
    torch.testing.assert_close(obtained,expected)


def test_gram_matches_explicit_autograd_score_and_is_psd():
    torch.manual_seed(112)
    model=TinyCausalLM().eval()
    model.embedding.weight.requires_grad_(False)
    responses=[[4,5],[5,6,7],[4,7]]
    result=score_gram_for_rollouts(model,[2,3],responses,max_trainable_parameters=1000)
    grad_vectors=[]
    for r in responses:
        lp=teacher_forced_sequence_logprob(model,[2,3],r)
        grad=torch.autograd.grad(lp,[model.head.weight])[0]
        grad_vectors.append(grad.reshape(-1).double().detach().numpy())
    explicit=np.stack(grad_vectors)
    np.testing.assert_allclose(result['score_kernel'],explicit@explicit.T,atol=1e-6)
    assert min(np.linalg.eigvalsh(result['score_kernel']))>-1e-9
    assert result['trainable_parameter_names']==['head.weight']
    assert result['score_definition']=='full_sequence_logprob_sum'


def test_no_trainable_adapter_fails_closed():
    model=TinyCausalLM()
    for p in model.parameters(): p.requires_grad_(False)
    with pytest.raises(ValueError,match='no trainable parameters'):
        score_gram_for_rollouts(model,[2,3],[[4],[5]])


def test_score_rejects_retokenized_or_empty_ids():
    model=TinyCausalLM()
    for bad in [[],[1,'2'],[-1]]:
        with pytest.raises(ValueError,match='token'):
            teacher_forced_sequence_logprob(model,[2,3],bad)
