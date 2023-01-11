# Getting Started
You will need to have `cookiecutter` installed.

```
pip install cookiecutter
```  
Then you can start a new repository for your Model-Service. Run:
```
cookiecutter ml-model-template
```
and follow the prompts.
You can then create a new empty repository on mietright's github and set up and push your new local repository to the github remote.

# Setting up Git
```
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:mietright/mietright/labeling.git
git push -u origin main
```


# labeling

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


## DVC
This whole project can be versioned as a DVC pipeline: from gathering external data, cleaning it, and training a model.  

Versioning with DVC is nice because it means that you can process your data once and re-use it many times (for example in CI/CD) and even authorize others to download your data.

Once you have DVC installed, you should begin making stages in your pipeline.

For instance your dataset-building stage might have a code dependency (`build_dataset.py`) as well as a couple of data dependencies, it might produce an output file at `data/clean.jsonl` and runs with the CLI
`labeling data` command.

Then versioning the stage with DVC would require just::
```bash
dvc run -n build-dataset -d labeling/data/build_dataset.py -d data/input/train.jsonl -d data/input/test.jsonl -o data/output/clean.jsonl labeling data
```

You can check the pipeline stages by doing
`dvc dag`

or by looking inside `dvc.yaml`.

You can reproduce the entire pipeline by doing  
`dvc repro`

or a specific stage (eg `build-dataset`) with
`dvc repro build-dataset`

# Deployment
The directory `deployments` contains files that should be committed to the corresponding
locations in the [mieright/deployments](https://github.com/mietright/deployments) repository.

Two modifications need to be made:
1. `deployments/base/labeling/manifests/labeling.deploy.yaml`
      Make sure that this file captures the entrypoint and arguments that you want to run in your service. The `port` value should match the value in `service.yaml`
2. `deployments/envs/staging/conny/labeling/kustomization.yaml`
    The `newTag` value should be the first 8 characters of the commit SHA that produced the successfully built docker image of your model API-Service that you wish to deploy.
