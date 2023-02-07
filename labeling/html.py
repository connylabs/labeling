import base64
from io import BytesIO


def sample_div(text, data, id=None):
    id = id if id is not None else text
    return f"<p><a href='#' id='{id}'>{data}{text}</a></p>"

def image_to_html(image, alt=""):
    b = BytesIO()
    image.save(b, format="png")
    b.seek(0)
    b64_string = base64.b64encode(b.read()).decode("ascii")
    return f'<img src="data:image/png;base64, {b64_string}" alt="{alt}"/>'

def make_history_divs(samples, load_image_fn=None):
    history = "\n".join([
        sample_div(
            text=sample["label"],
            data=image_to_html(load_image_fn(sample["image"]["path"])),
            id=i,
        ) for i, sample in enumerate(samples)])

    return history
