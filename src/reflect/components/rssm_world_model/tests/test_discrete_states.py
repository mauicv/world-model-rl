from reflect.components.rssm_world_model.state import InternalStateDiscrete, InternalStateDiscreteSequence
import torch

def test_discrete_state():
    state = InternalStateDiscrete(
        deter_state=torch.zeros(32, 200),
        stoch_state=torch.zeros(32, 30),
        logits=torch.zeros(32, 30, 30),
    )
    assert state.deter_state.shape == (32, 200)
    assert state.stoch_state.shape == (32, 30)
    assert state.logits.shape == (32, 30, 30)
    assert state.get_features().shape == (32, 230)
    assert state.shapes == ((32, 200), (32, 30), (32, 30, 30))


def test_discrete_state_sequence():
    state_sequence = InternalStateDiscreteSequence(
        deter_states=torch.zeros(32, 10, 200),
        stoch_states=torch.zeros(32, 10, 30),
        logits=torch.zeros(32, 10, 30, 30),
    )
    assert state_sequence.deter_states.shape == (32, 10, 200)
    assert state_sequence.stoch_states.shape == (32, 10, 30)
    assert state_sequence.get_features().shape == (32, 9, 230)
    assert state_sequence.shapes == ((32, 10, 200), (32, 10, 30), (32, 10, 30, 30))
    

def test_discrete_state_sequence_append():
    state_sequence = InternalStateDiscreteSequence(
        deter_states=torch.zeros(32, 10, 200),
        stoch_states=torch.zeros(32, 10, 30),
        logits=torch.zeros(32, 10, 30, 30),
    )
    state = InternalStateDiscrete(
        deter_state=torch.zeros(32, 200),
        stoch_state=torch.zeros(32, 30),
        logits=torch.zeros(32, 30, 30),
    )
    state_sequence.append_(state)
    assert state_sequence.deter_states.shape == (32, 11, 200)
    assert state_sequence.stoch_states.shape == (32, 11, 30)
    assert state_sequence.get_features().shape == (32, 10, 230)
    assert state_sequence.shapes == ((32, 11, 200), (32, 11, 30), (32, 11, 30, 30))


def test_discrete_state_sequence_getitem():
    state_sequence = InternalStateDiscreteSequence(
        deter_states=torch.zeros(32, 10, 200),
        stoch_states=torch.zeros(32, 10, 30),
        logits=torch.zeros(32, 10, 30, 30),
    )
    item = state_sequence[3]
    assert isinstance(item, InternalStateDiscrete)
    assert item.deter_state.shape == (32, 200)
    assert item.stoch_state.shape == (32, 30)
    assert item.get_features().shape == (32, 230)
    assert item.shapes == ((32, 200), (32, 30), (32, 30, 30))


def test_discrete_state_sequence_from_init():
    state = InternalStateDiscrete(
        deter_state=torch.zeros(32, 200),
        stoch_state=torch.zeros(32, 30),
        logits=torch.zeros(32, 30, 30),
    )
    state_sequence = InternalStateDiscreteSequence.from_init(state)
    assert state_sequence.deter_states.shape == (32, 1, 200)
    assert state_sequence.stoch_states.shape == (32, 1, 30)
    assert state_sequence.get_features().shape == (32, 0, 230)
    assert state_sequence.shapes == ((32, 1, 200), (32, 1, 30), (32, 1, 30, 30))
