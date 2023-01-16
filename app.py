import os
from pathlib import Path
import base64

import click
import streamlit as st
from st_click_detector import click_detector
from datasets import load_dataset, Image

from labeling.annotator import Annotator
from labeling.samplers import RandomSampler, ActiveLearningSampler

@click.argument("img-dir", type=click.Path())
@click.argument("labels", type=click.Path())
def run(img_dir, labels):
    img_dir = Path(img_dir)

    def refresh():
        dataset = load_dataset(
            "imagefolder",
            data_dir=img_dir,
            split="train"
        )
        st.session_state["annotator"] = Annotator(dataset, RandomSampler(), img_dir/"metadata.jsonl")

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

    if st.session_state["annotator"].labeled_data:

        def image_bytes_to_md(bytes):
            return f"![](data:image/png;base64,{str(base64.b64encode(bytes))[2:-1]}"

        def image_bytes_to_html(bytes, alt=""):
            return f'<img src="data:image/png;base64, {str(base64.b64encode(bytes))[2:-1]}" alt="{alt}"/>'

        def sample_div(text, data, id=None):
            id = id if id is not None else text
            return f"<p><a href='#' id='{id}'>{data}{text}</a></p>"

        history = "\n".join([
            sample_div(
                text=sample["label"],
                data=image_bytes_to_html(sample["tiny"]["bytes"]),
                id=i,
            ) for i, sample in enumerate(st.session_state["annotator"].labeled_data[::-1])])

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
                    on_click=st.session_state["annotator"].update,
                    args=(label,),
                    type=type
                ))

        with info_tab:
            st.info(sample.keys())

    except IndexError:
        st.balloons()
        st.success("Done!")

if __name__ == "__main__":
    custom_labels = ["dog", "cat", "SKIP"]
    img_dir = Path(__file__).parent.joinpath("data")
    # img_dir = "/"
    run(img_dir, custom_labels)
