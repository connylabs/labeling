import os
from pathlib import Path
import base64

import click
import streamlit as st
from st_click_detector import click_detector
from datasets import load_dataset, Image, Features, ClassLabel

from labeling.annotator import Annotator
from labeling.utils import SKIP_LABEL, load_dataset_from_disk, load_image
from labeling.samplers import SAMPLERS
from labeling.html import make_history_divs
from labeling.logger import get_logger


_DEFAULT_SAMPLER_NAME = "active-learning"
_DEFAULT_MODEL_NAME = None
_DEFAULT_LIMIT = None
_DEFAULT_RESIZE = None
_DEFAULT_RETRAIN_STEPS = 50

@click.command()
@click.option("--img-dir", type=click.Path(), required=True)
@click.option("--metadata-path", type=click.Path(), required=True)
@click.option("--labels", multiple=True, type=click.Path(), required=True)
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

    logger = get_logger(__name__)

    logger.info(f"Using image directory: `{img_dir}`")
    logger.info(f"Using metadata path: `{metadata_path}`")
    logger.info(f"Using labels: `{labels}`")
    logger.info(f"Using sampler-name: `{sampler_name}`")
    logger.info(f"Using model-name: `{model_name}`")
    logger.info(f"Using retrain-steps: `{retrain_steps}`")
    logger.info(f"Using limit: `{limit}`")
    logger.info(f"Using resize: `{resize}`")

    labels = list(labels)
    labels.append(SKIP_LABEL)
    img_dir = Path(img_dir)

    def refresh():

        logger.info(f"Loading dataset from: `{img_dir}`...")
        with st.spinner(f"Loading dataset from: `{img_dir}`..."):
            dataset = load_dataset_from_disk(img_dir, metadata_path, labels=labels)

        Sampler = SAMPLERS[sampler_name]
        if sampler_name == "active-learning":
            sampler = Sampler(labels=labels, retrain_steps=retrain_steps, model_name=model_name)
        else:
            sampler = Sampler()

        logger.info("Sorting unlabeled data...")
        with st.spinner("Sorting unlabeled data..."):
            st.session_state["annotator"] = Annotator(
                dataset,
                sampler,
                metadata_path,
                limit
            )

    if "annotator" not in st.session_state:
        logger.info("Refreshing app...")
        refresh()

    # Sidebar: show status
    n_total = len(st.session_state["annotator"])
    n_labeled = len(st.session_state["annotator"].labeled_data)
    n_unlabeled = len(st.session_state["annotator"].unlabeled_data)
    logger.info(f"n-total: `{n_total}`")
    logger.info(f"n-labeled: `{n_labeled}`")
    logger.info(f"n-unlabeled: `{n_unlabeled}`")

    st.sidebar.write("Total samples:", n_total)
    st.sidebar.write("Total done:", n_labeled)
    st.sidebar.write("To-Do:", n_unlabeled)
    st.sidebar.progress(int(100 * (n_labeled / n_total)))

    def handle_history_click(index):
        idx = n_labeled - int(index) - 1
        st.session_state["annotator"].redo(idx)
        logger.info(f"Clicked history item with index: `{idx}`")
        st.experimental_rerun()

    # render history
    if st.session_state["annotator"].labeled_data:
        history = make_history_divs(st.session_state["annotator"].labeled_data[::-1])
        with st.sidebar:
            index = click_detector(history)
            if index != "":
                handle_history_click(index)

    label_tab, info_tab = st.tabs(["Label", "Info"])

    try:
        sample = st.session_state["annotator"].current_sample
        logger.info(f"Current Sample: `{sample}`")
    except IndexError:
        st.balloons()
        st.success("Done!")
    else:
        with label_tab:
            st.image(load_image(sample["image"]["path"], size=resize))

            # labeling UI with one button per column
            buttons = []
            types = ["secondary", "primary"]
            columns = st.columns(len(labels))

            for i, (label, col) in enumerate(zip(labels, columns)):

                # if sample is already labeled,
                # get button index for sample's current label
                try:
                    index = labels.index(sample["label"])
                except ValueError:
                    index = None

                is_primary = i == index
                type = types[is_primary]

                def handle_label(label):
                    logger.info(f"setting current sample with label : `{label}`")
                    st.session_state["annotator"].set_label(label)

                buttons.append(col.button(
                    str(label),
                    on_click=handle_label,
                    args=(label,),
                    type=type
                ))

        with info_tab:
            st.info(sample)



if __name__ == "__main__":
    run(standalone_mode=False)
