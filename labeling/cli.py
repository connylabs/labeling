# -*- coding: utf-8 -*-
import os
import click
import logging

from labeling import app

import sys
from streamlit.web import cli as stcli


@click.command()
def cli():
    """
    labeling
    """
    sys.argv = ["streamlit", "run", f"{app.__file__}"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    cli()
