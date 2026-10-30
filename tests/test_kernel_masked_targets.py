"""Per-structure target costs honor empty candidate domains."""
import pytest

from kernel import learn as K
from kernel.structure import Structure

pytestmark = pytest.mark.invariant


@pytest.mark.parametrize("slots",[(0,1,0),(1,0,1),(0,0,1,0,0,1,0),(0,0),(1,1)])
def test_empty_domains_cost_zero_without_borrowing_a_neighboring_row(slots):
    positive=Structure({'slot':{('e',)},'answer':{('e',)}})
    empty=Structure({'slot':set(),'answer':set()})
    target=K.Target('answer',1,('slot',),body=())
    m,_=K.learn([positive],[target],K.Search(max_rules=0,invent=False,cells=False))
    structures=[positive if present else empty for present in slots]
    assert K.bits_each(m,structures,target).tolist() == pytest.approx(
        [K.bits_each(m,[positive],target)[0] if present else 0. for present in slots])
    assert len(K.outcomes(structures,target)) == sum(slots)


def test_empty_corpus_has_no_per_structure_costs():
    target=K.Target('answer',1,('slot',),body=())
    m,_=K.learn([], [target],K.Search(max_rules=0,invent=False,cells=False))
    assert K.bits_each(m,[],target).tolist() == []
