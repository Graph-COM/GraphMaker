# GraphMaker

![model](model.png)

## Table of Contents

- [Installation](#installation)
- [Frequently Asked Questions](#frequently-asked-questions)
  * [Q1: libcusparse.so](#q1-libcusparseso)
  * [Q2: Other requests](#q2-other-requests)

## Installation

```bash
conda create -n GraphMaker python=3.8 -y
conda activate GraphMaker
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
conda install -c conda-forge cudatoolkit=11.6
pip install dgl==1.1.0+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html
pip install pandas scikit-learn pydantic wandb
```

You also need to compile `orca.cpp`.

```bash
cd orca
g++ -O2 -std=c++11 -o orca orca.cpp
```

## Frequently Asked Questions

### Q1: libcusparse.so

**An error occurs that the program cannot find `libcusparse.so`.**

To search for the location of it on linux,

```bash
find /path/to/directory -name libcusparse.so.11 -exec realpath {} \;
```

where `/path/to/directory` is the directory you want to search. Assume that the search returns `home/miniconda3/envs/GraphMaker/lib/libcusparse.so.11`. Then you need to manually specify the environment variable as follows.

```bash
export LD_LIBRARY_PATH=home/miniconda3/envs/GraphMaker/lib:$LD_LIBRARY_PATH
```

### Q2: Other requests

**I have a question not listed here.**

- It's generally recommended to open a GitHub issue. This allows us to track the progress, and the discussion might help others who have the same question.
- Otherwise, you can also send an email to `mufeili1996@gmail.com`.
