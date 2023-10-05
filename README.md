# GraphMaker

![model](model.png)

## Table of Contents

- [Installation](#installation)
- [Frequently Asked Questions](#frequently-asked-questions)

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
