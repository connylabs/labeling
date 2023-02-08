# -*- coding: utf-8 -*-
import click
# import sys
# import subprocess

# from streamlit.web import cli as stcli
import streamlit.web.bootstrap

from labeling import app


@click.command()
@click.option("--img-dir", type=click.Path(), required=True)
@click.option("--metadata-path", type=click.Path(), required=True)
@click.option("--labels", multiple=True, type=click.Path(), required=True)
@click.option("--sampler-name", "--sampler", type=str, default=app._DEFAULT_SAMPLER_NAME)
@click.option("--model-name", "--model", type=str, default=app._DEFAULT_MODEL_NAME)
@click.option("--retrain-steps", "--retrain", type=int, default=app._DEFAULT_RETRAIN_STEPS)
@click.option("--limit", type=int, default=app._DEFAULT_LIMIT)
@click.option("--resize", type=int, default=app._DEFAULT_RESIZE)
def cli(
    img_dir,
    metadata_path,
    labels,
    **kwargs):
    """
    labeling
    """
    args = [
        "--img-dir", img_dir,
        "--metadata-path", metadata_path
    ]

    for label in labels:
        args.append("--labels")
        args.append(label)

    for key, value in kwargs.items():
        if value is not None:
            args.extend([f"--{key.replace('_', '-')}", value])

    streamlit.web.bootstrap.run(app.__file__, "", args, flag_options={})

if __name__ == "__main__":
    cli()
