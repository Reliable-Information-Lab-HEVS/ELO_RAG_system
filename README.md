# RAG system

This is a RAG system prototype designed to provide support to students in simple mathematics.

## Hardware requirements

Note that you will need a machine with at least 2 GPUs (with at least 16GiB of RAM each) to run everything smoothly. With minimal change, a single larger GPU (~40 GiB of RAM) is also compatible.

## Environment setup

If you do not already have `conda` on your machine, please run the following command, which will install
`miniforge3` (only for Linux) and create the correct `conda` environment:

```sh
cd path/to/this/repo
source config.sh
```

If you already have a `conda` installation, you can simply run:

```sh
cd path/to/this/repo
conda env create -f requirements.yaml
```

## Running the webapp

To launch the demo webapp, run:

```
python webapp.py
```

Once started, you can either visit the given URL to use it in your browser, or query it via the API (example usage in `webapp_api.py`).