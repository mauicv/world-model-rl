import torch


class Base(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(
            self,
            path,
            name="world-model-checkpoint.pth",
            targets=None
        ):
        device = next(self.parameters()).device
        checkpoint = torch.load(
            f'{path}/{name}',
            map_location=torch.device(device)
        )
        if targets is None: targets = self.model_list
        
        for target in targets:
            print(f'Loading {target}...')
            getattr(self, target).load_state_dict(
                checkpoint[target]
            )

    def save(
            self,
            path,
            name="world-model-checkpoint.pth",
            targets=None
        ):
        if targets is None: targets = self.model_list
        
        checkpoint = {
            target: getattr(self, target).state_dict()
            for target in targets
        }
        torch.save(checkpoint, f'{path}/{name}')