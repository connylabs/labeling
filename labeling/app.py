import os
from pathlib import Path
import base64

import click
import streamlit as st
from st_click_detector import click_detector
from datasets import load_dataset, Image, Features, ClassLabel

from labeling.annotator import Annotator
from labeling.utils import SKIP_LABEL, load_dataset_from_disk
from labeling.samplers import SAMPLERS
from labeling.html import make_history_divs


# @click.command()
@click.option("--sampler-name", "--sampler", type=str, default="active-learning")
@click.option("--batch-size", type=int, default=10)
@click.option("--limit", type=int, default=None)
@click.argument("--img-dir", type=click.Path())
@click.argument("--metadata-path", type=click.Path())
@click.argument("--labels", type=click.Path())
def run(sampler_name, batch_size, limit, img_dir, metadata_path, labels):
    img_dir = Path(img_dir)

    def refresh():

        dataset = load_dataset_from_disk(img_dir, metadata_path, labels=labels)

        Sampler = SAMPLERS[sampler_name]
        if sampler_name == "active-learning":
            sampler = Sampler(labels=labels, batch_size=batch_size)
        else:
            sampler = Sampler()

        st.session_state["annotator"] = Annotator(
            dataset,
            sampler,
            metadata_path,
            limit
        )

    if "annotator" not in st.session_state:
        refresh()

    # Sidebar: show status
    n_total = len(st.session_state["annotator"])
    n_labeled = len(st.session_state["annotator"].labeled_data)
    n_unlabeled = len(st.session_state["annotator"].unlabeled_data)

    st.sidebar.write("Total samples:", n_total)
    st.sidebar.write("Total done:", n_labeled)
    st.sidebar.write("To-Do:", n_unlabeled)
    st.sidebar.progress(int(100 * (n_labeled / n_total)))

    def handle_history_click(index):
        idx = n_labeled - int(index) - 1
        st.session_state["annotator"].redo(idx)
        st.experimental_rerun()

    # render history
    if st.session_state["annotator"].labeled_data:
        history = make_history_divs(st.session_state["annotator"].labeled_data[::-1])
        with st.sidebar:
            index = click_detector(history)
            if index != "":
                handle_history_click(index)

    try:
        label_tab, info_tab = st.tabs(["Label", "Info"])

        with label_tab:
            sample = st.session_state["annotator"].current_sample
            st.image(Image(decode=True).decode_example(sample["thumbnail"]))

            buttons = []
            types = ["secondary", "primary"]
            columns = st.columns(len(custom_labels))

            for i, (label, col) in enumerate(zip(custom_labels, columns)):

                try:
                    index = custom_labels.index(sample["label"])
                except ValueError:
                    index = None

                is_primary = i == index
                type = types[is_primary]

                buttons.append(col.button(
                    str(label),
                    on_click=st.session_state["annotator"].set_label,
                    args=(label,),
                    type=type
                ))

        with info_tab:
            st.info(sample.keys())

    except IndexError:
        st.balloons()
        st.success("Done!")

if __name__ == "__main__":
    sampler_name = "active-learning"
    labels = ["dog", "cat", SKIP_LABEL]
    img_dir = "/home/dani/dev/conny-dev/labeling/data"
    run(sampler_name, img_dir, labels)
