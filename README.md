# labeling
![Alt Text](https://github.com/connylabs/labeling/blob/main/labeling.gif)



## Installation
`pip install .`
  or
`pip install -e .` for an editable installation

You will need an globally accessible `DVC` installation.  
You can install DVC as a binary with `pipx` by doing:
```bash
pip install pipx
pipx install dvc[s3]
```

## Development: labeling
Your main CLI program should be written `labeling.cli:cli`
and will then be registered as the CLI command `labeling` .

You can then flesh-out your data processing and model training scripts at
- `labeling/data/build_dataset.py`
- `labeling/models/train.py`

which you can execute in the CLI via
- `labeling data`
- `labeling train`
