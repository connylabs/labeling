import base64


def image_bytes_to_html(bytes, alt=""):
    return f'<img src="data:image/png;base64, {str(base64.b64encode(bytes))[2:-1]}" alt="{alt}"/>'

def sample_div(text, data, id=None):
    id = id if id is not None else text
    return f"<p><a href='#' id='{id}'>{data}{text}</a></p>"

def make_history_divs(samples):
    history = "\n".join([
        sample_div(
            text=sample["label"],
            data=image_bytes_to_html(sample["tiny"]["bytes"]),
            id=i,
        ) for i, sample in enumerate(samples)])

    return history
