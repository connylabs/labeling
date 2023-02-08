import hashlib
import pathlib
from functools import partial

import PIL
import PIL.Image
from joblib import Parallel, delayed
import click
import pandas as pd
import numpy as np
from tqdm import tqdm

from labeling.logger import get_logger
from labeling.utils import tqdm_joblib


IMG_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".gif",
    ".bmp"
]

def image_path_filter(path):
    return (path.suffix.lower() in IMG_EXTS) and (path.is_file())

def _crop_white(image):
    bbox = PIL.ImageOps.invert(image).getbbox()
    if bbox:
        return image.crop(bbox)
    return image

def _crop_black(image):
    bbox = image.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

def crop(image):
    return _crop_black(_crop_white(image))

def load_image(path, convert="RGB"):
    try:
        return PIL.Image.open(path).convert(convert)
    except PIL.UnidentifiedImageError:
        return None

def make_thumbnail(image, size=500):
    if isinstance(size, int):
        size = (size, size)
    image.thumbnail(size)
    return image


def hash_image(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def _preprocess(path, thumbnail_dir=None, size=500):
    image = load_image(path, convert="RGB")
    if image is None:
        return
    image = crop(image)
    image = make_thumbnail(image, size=size)
    hash = hash_image(image)

    thumbnail_path = thumbnail_dir.joinpath(f"{hash}.png")
    image.save(thumbnail_path, format="png")
    return (path, hash, thumbnail_path)


@click.option("--input-dir", "--input", default="/home/dani/dev/conny-dev/image-classification/data/input/raw", type=click.Path())
@click.option("--thumbnail-dir", "--output", default="/home/dani/dev/conny-dev/image-classification/data/input/thumbnails", type=click.Path())
@click.option("--size", default=500, type=int)
@click.option("--limit", default=None, type=int)
@click.option("--n-workers", default=8, type=int)
def run(input_dir, thumbnail_dir, size, limit, n_workers):
    """
    speed up labeling by reducing image size
    """
    logger = get_logger(__name__)

    thumbnail_dir = pathlib.Path(thumbnail_dir)
    thumbnail_dir.mkdir(exist_ok=True, parents=True)

    paths = pathlib.Path(input_dir).rglob("*")
    paths = filter(image_path_filter, paths)

    if limit:
        paths = (path for i, path in zip(range(limit), paths))

    paths = list(paths)
    logger.info(f"Found {len(paths)} images...")

    preprocess = partial(_preprocess, thumbnail_dir=thumbnail_dir, size=size)
    with tqdm_joblib(tqdm(total=len(paths))) as progress_bar:
        data = Parallel(n_jobs=n_workers)(delayed(preprocess)(path) for path in paths)

    data = filter(None, data)

    file_names, hashes, thumbnail_paths = zip(*data)
    df = pd.DataFrame.from_dict(
        {
            "file_name": file_names,
            "hash": hashes,
            "thumbnail_path": thumbnail_paths
        }
    )

    df.to_csv(thumbnail_dir/"hashes.csv", index=False)

if __name__ == '__main__':
    click.command(run)()
