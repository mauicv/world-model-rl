import click
from reflect.examples.simple_world import plotting, testing, training


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


cli.add_command(plotting.plot)


cli.add_command(testing.play_real)
cli.add_command(testing.play_model)
cli.add_command(testing.play_agent)
cli.add_command(testing.test_obs)


cli.add_command(training.train_obs)
cli.add_command(training.train_wm)
cli.add_command(training.train_stgrad)



if __name__ == '__main__':
    cli()