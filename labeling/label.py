from pathlib import Path

import click
import streamlit as st
from st_click_detector import click_detector

from labeling.annotator import Annotator
from labeling.utils import load_dataset_from_disk, load_image
from labeling.html import make_history_divs
from labeling.logger import get_logger
from labeling import defaults
from labeling.samplers import SAMPLERS


@click.command()
@click.option("--img-dir", type=click.Path(), required=True)
@click.option("--metadata-path", type=click.Path(), required=True)
@click.option("--labels", multiple=True, type=click.Path(), required=True)
@click.option("--sampler-name", "--sampler", type=str, default=defaults.SAMPLER_NAME)
@click.option("--model-name", "--model", type=str, default=defaults.MODEL_NAME)
@click.option("--retrain-steps", "--retrain", type=int, default=defaults.RETRAIN_STEPS)
@click.option("--limit", type=int, default=defaults.LIMIT)
@click.option("--resize", type=int, default=defaults.RESIZE)
def run(
    img_dir,
    metadata_path,
    labels,
    sampler_name=defaults.SAMPLER_NAME,
    model_name=defaults.MODEL_NAME,
    retrain_steps=defaults.RETRAIN_STEPS,
    limit=defaults.LIMIT,
    resize=defaults.RESIZE,
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
    labels.append(defaults.SKIP_LABEL)
    img_dir = Path(img_dir)

    def refresh():

        logger.info(f"Loading dataset from: `{img_dir}`...")
        with st.spinner(f"Loading dataset from: `{img_dir}`..."):
            dataset = load_dataset_from_disk(img_dir, metadata_path, labels=labels)

        Sampler = SAMPLERS[sampler_name]
        if sampler_name == "active-learning":
            sampler = Sampler(
                labels=labels, retrain_steps=retrain_steps, model_name=model_name
            )
        else:
            sampler = Sampler()

        logger.info("Sorting unlabeled data...")
        with st.spinner("Sorting unlabeled data..."):
            st.session_state["annotator"] = Annotator(
                dataset, sampler, metadata_path, limit
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
        st.session_state["annotator"].redo(int(index))
        logger.info(f"Redoing labeled sample with index: `{index}`")
        st.experimental_rerun()

    # render history
    if st.session_state["annotator"].labeled_data:
        with st.sidebar:
            history_filters = st.multiselect('Filter History', labels, labels)
            history_len = st.slider('History Size', 1, 1000, defaults.HISTORY_LEN)
            history_samples = [
                (i, s)
                for i, s in enumerate(st.session_state["annotator"].labeled_data)
                if s["label"] in history_filters
            ]

            if len(history_samples) > 0:
                idxs, history_samples = zip(*history_samples)

                history = make_history_divs(history_samples[-history_len:][::-1])
                idxs = idxs[-history_len:][::-1]

                clicked_index = click_detector(history)
                if clicked_index != "":
                    logger.info(f"Clicked history item with ID: `{clicked_index}`")
                    logger.info(f"Filtered idxs`{idxs}`")
                    index = idxs[int(clicked_index)]
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
                button_type = types[is_primary]

                def handle_label(label):
                    logger.info(f"setting current sample with label : `{label}`")
                    st.session_state["annotator"].set_label(label)

                buttons.append(
                    col.button(
                        str(label),
                        on_click=handle_label,
                        args=(label,),
                        type=button_type,
                    )
                )

        with info_tab:
            st.write(sample)


if __name__ == "__main__":
    run(standalone_mode=False)
