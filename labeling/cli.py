# -*- coding: utf-8 -*-
import click
import streamlit.web.bootstrap

from labeling import preprocess
from labeling import defaults


@click.group()
def cli():
    """
    Labeling with Active Learning
    """
    pass


cli.command("preprocess")(preprocess.run)


@cli.command()
@click.option("--img-dir", type=click.Path(), required=True)
@click.option("--metadata-path", type=click.Path(), required=True)
@click.option("--labels", multiple=True, type=click.Path(), required=True)
@click.option("--sampler-name", "--sampler", type=str, default=defaults.SAMPLER_NAME)
@click.option("--model-name", "--model", type=str, default=defaults.MODEL_NAME)
@click.option("--retrain-steps", "--retrain", type=int, default=defaults.RETRAIN_STEPS)
@click.option("--limit", type=int, default=defaults.LIMIT)
@click.option("--resize", type=int, default=defaults.RESIZE)
def label(img_dir, metadata_path, labels, **kwargs):
    """
    label your data with active learning
    """
    args = ["--img-dir", img_dir, "--metadata-path", metadata_path]

    for label in labels:
        args.append("--labels")
        args.append(label)

    for key, value in kwargs.items():
        if value is not None:
            args.extend([f"--{key.replace('_', '-')}", value])

    from labeling import label as app

    streamlit.web.bootstrap.run(app.__file__, "", args, flag_options={})


if __name__ == "__main__":
    cli()
