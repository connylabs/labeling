# -*- coding: utf-8 -*-
import os
import click
import logging

from cookiecutter.package_name import models
from cookiecutter.package_name import data


@click.group()
def cli():
    """
    labeling
    """
    pass


cli.command("train")(models.train.run)
cli.command("data")(data.build_dataset.run)


if __name__ == "__main__":
    cli()
