import os
from pathlib import Path
import base64

import click
import streamlit as st
from st_click_detector import click_detector
from datasets import load_dataset, Image, Features, ClassLabel

from labeling.annotator import Annotator
from labeling.utils import SKIP_LABEL, load_dataset_from_disk, make_tiny, load_image
from labeling.samplers import SAMPLERS
from labeling.html import make_history_divs


_DEFAULT_SAMPLER_NAME = "active-learning"
_DEFAULT_MODEL_NAME = None
_DEFAULT_LIMIT = None
_DEFAULT_RESIZE = None
_DEFAULT_RETRAIN_STEPS = 50

# @click.command()
@click.argument("--img-dir", type=click.Path())
@click.argument("--metadata-path", type=click.Path())
@click.argument("--labels", type=click.Path())
@click.option("--sampler-name", "--sampler", type=str, default=_DEFAULT_SAMPLER_NAME)
@click.option("--model-name", "--model", type=str, default=_DEFAULT_MODEL_NAME)
@click.option("--retrain-steps", "--retrain", type=int, default=_DEFAULT_RETRAIN_STEPS)
@click.option("--limit", type=int, default=_DEFAULT_LIMIT)
@click.option("--resize", type=int, default=_DEFAULT_RESIZE)
def run(
    img_dir,
    metadata_path,
    labels,
    sampler_name=_DEFAULT_SAMPLER_NAME,
    model_name=_DEFAULT_MODEL_NAME,
    retrain_steps=_DEFAULT_RETRAIN_STEPS,
    limit=_DEFAULT_LIMIT,
    resize=_DEFAULT_RESIZE
    ):

    img_dir = Path(img_dir)

    def refresh():

        with st.spinner(f"Loading dataset from: `{img_dir}`..."):
            dataset = load_dataset_from_disk(img_dir, metadata_path, labels=labels)

        Sampler = SAMPLERS[sampler_name]
        if sampler_name == "active-learning":
            sampler = Sampler(labels=labels, retrain_steps=retrain_steps, model_name=model_name)
        else:
            sampler = Sampler()

        with st.spinner(f"Sorting unlabeled data..."):
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
        history = make_history_divs(st.session_state["annotator"].labeled_data[::-1], load_image_fn=make_tiny)
        with st.sidebar:
            index = click_detector(history)
            if index != "":
                handle_history_click(index)

    try:
        label_tab, info_tab = st.tabs(["Label", "Info"])

        with label_tab:
            sample = st.session_state["annotator"].current_sample

            st.image(load_image(sample["image"]["path"], size=resize))

            buttons = []
            types = ["secondary", "primary"]
            columns = st.columns(len(labels))

            for i, (label, col) in enumerate(zip(labels, columns)):

                try:
                    index = labels.index(sample["label"])
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
            st.info(sample)

    except IndexError:
        import pdb; pdb.set_trace()
        st.balloons()
        st.success("Done!")

if __name__ == "__main__":
    sampler_name = "active-learning"
    labels = ["world", "document", SKIP_LABEL]
    img_dir = "/home/dani/dev/conny-dev/image-classification/data/output/thumbnails"
    metadata_path = "/home/dani/dev/conny-dev/image-classification/data/output/metadata.jsonl"
    model_name = None
    limit = 1000
    retrain_steps = 5
    run(img_dir, metadata_path, labels, sampler_name=sampler_name, model_name=model_name, retrain_steps=retrain_steps, limit=limit)
