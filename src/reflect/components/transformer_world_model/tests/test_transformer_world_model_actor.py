from reflect.components.transformer_world_model.memory_actor import TransformerWorldModelActor
import torch


def test_transformer_world_model_actor(
        transformer_world_model_actor: TransformerWorldModelActor
    ):
    obs = torch.Tensor(1, 1, 3, 64, 64)
    action = transformer_world_model_actor(obs)
    assert action.shape == (1, 1)