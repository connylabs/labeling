.PHONY: black black-test check clean clean-build clean-pyc clean-test coverage docs flake8 generate help install lint servedocs test test-all
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"
VERSION := `cat VERSION`
package := "labeling"
MEOWLFLOW_SCHEMA_LOCAL_PATH := "labeling/models/model/meowlflow_schema.py"

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "black - check formatting with black"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate API docs"
	@echo "install - install the package to the active Python's site-packages"
	@echo "generate - build all generated artifacts"

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.mypy_cache' -exec rm -fr {} +
	find . -name '.pyre' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

test:
	poetry run pytest --cov=$(package) --cov-report=html --cov-report=term-missing  --verbose tests

coverage:
	coverage run --source $(package) setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

GENERATE_IMAGE ?= openapitools/openapi-generator-cli:v5.3.1
CONTAINERIZE_GENERATE ?= true
GENERATE_PREFIX ?= /usr/local/bin/docker-entrypoint.sh
GENERATE_SUFIX ?=
ifeq ($(CONTAINERIZE_GENERATE), true)
	GENERATE_PREFIX := docker run --rm \
	    -u $$(id -u):$$(id -g) \
	    -v $$(pwd):/src \
	    -w /src \
	    $(GENERATE_IMAGE)
	GENERATE_SUFIX :=
endif

docs: Documentation/README.md
Documentation/README.md: ./api/api.json
	mkdir -p $(@D)
	$(GENERATE_PREFIX) generate -i $< -g markdown -o $(@D) $(GENERATE_SUFIX)

servedocs: docs
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

install: clean
	pip install poetry
	poetry install

install-dev: clean
		pip install poetry
		poetry install --only dev

lint:
	poetry run flake8 $(package) tests

black:
	poetry run black -t py39 tests $(package)

black-test:
	poetry run black -t py39 tests $(package) --check

check: flake8 black-test

openapi: ./api/api.json
./api/api.json: $(MEOWLFLOW_SCHEMA_LOCAL_PATH)
	poetry run meowlflow openapi --endpoint infer --schema-path $< > $@

generate: openapi docs
