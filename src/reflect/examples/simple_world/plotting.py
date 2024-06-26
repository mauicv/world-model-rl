from reflect.utils import CSVLogger
import click


@click.command()
def plot():
    logger = CSVLogger(path=f"./experiments/wm-td3/results.csv",
        fieldnames=[
            # 'recon_loss',
            # 'reg_loss',
            # 'consistency_loss',
            # 'dynamic_loss',
            # 'reward_loss',
            # 'done_loss',
            'actor_loss',
            'rewards'
        ])

    logger.plot(
        [
            # ('recon_loss',),
            # # ('reg_loss',),
            # ('consistency_loss',),
            # ('dynamic_loss',),
            # ('reward_loss',),
            # ('done_loss', ),
            ('rewards',), 
            ('actor_loss', )
        ]
    )