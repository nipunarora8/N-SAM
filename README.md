<div align="center">

# BIMAP - Foundation models for Dendrite Segmentation

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![wandb](https://img.shields.io/badge/Weights_&_Biases-FFCC33?logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/site)

This project demonstrates the use of Foundation Models for Dendrite Segmentation.
The model of choice is SAM with LoRA and the framework used it pytorch.

![A stack of neural dendrites](./images/Dendrite_U_maskdendrite_max.png "Neural Dendrites")

</div>

### 📝  Table of Contents

- [BIMAP - Foundation models for Dendrite Segmentation](#bimap---foundation-models-for-dendrite-segmentation)
    - [📝  Table of Contents](#--table-of-contents)
    - [📦  Built With](#--built-with)
    - [📂  Project Structure](#--project-structure)
    - [🚀  Setup](#--setup)
    - [🏋️  SAM Weights](#️--sam-weights)
    - [🤖  Training](#training)
    - [🧪  Evaluation](#evaluation)
    - [✨  Bonus](#--bonus)
      - [⏱️ Schedule a Job on HPC](#️-schedule-a-job-on-hpc)
      - [Useful VSCode Extensions](#useful-vscode-extensions)
    - [📬  Contact](#--contact)
    - [📚  References](#--references)

<br>

### 📦  Built With

[PyTorch](https://pytorch.org) - an open-source machine learning library for Python, widely used for deep learning applications.

[Segment Anything Model](https://segment-anything.com) - a foundation model used for segmentation built by Meta AI.

[Weights and Biases](https://wandb.ai/site) - a tool for tracking and visualizing machine learning experiments.

[Visual Studio Code](https://code.visualstudio.com/) - a code editor redefined and optimized for building applications.

[FAU High Performance Computing](https://doc.nhr.fau.de/) - a high-performance computing cluster at Friedrich-Alexander-Universität Erlangen-Nürnberg.

<br>

### 📂  Project Structure

The directory structure of the project looks like this:

```
├── datasets                            <- Project data
│
├── logs                                <- HPC Logs
│
├── results                             <- Training Results
│   └── date_time                           <- Data_Time of training start
│       └── checkpoints                         <- Checkpoints
│           └── checkpoint.pth                      <- Checkpoint
│
├── sam_weights                         <- Keep the SAM weights here
│
├── scheduler                           <- Automatic Training Scheduler
│   ├── slurm                           
│   │   └── BIMAP.slurm                     <- Default SLURM script
│   └──  schedule.py                    <- Scheduler Script
│
├── segment_anything                    <- SAM Official Code
│   ├── modeling                            <- SAM Modeling
│   ├── utils                               <- SAM Utils
│   ├── build_sam.py                        <- Build SAM
│   └──  predictor.py                       <- SAM Predictor
│  
├── test                                <- Test Results are stored here
│   └── date_time_testname             
│       ├── outputs                         <- Outputs Image Folder
│       │   └── images.png                      <- Output Images
│       ├── preds.tif                       <- Predicted Tif
│       └── metrics.txt                     <- Output Metrics
│
├── .gitignore                          <- Gitignore
├── configs.py                          <- Train/Test Configs
├── README.md                           <- ReadMe File
├── requirements.txt                    <- Installation Dependencies
├── sam_lora_image_encoder.py           <- LoRA wrapper
├── schedule.sh                         <- Training Scheduler
├── test.py                             <- Test Script
└── train.py                            <- Training Script
```

<br>

### 🚀  Setup

Before getting started here are the few things that are required to do.

1. **Setting up working repo**

   First of all clone this repository in your system.

   ```bash
   git clone https://gitlab.rrze.fau.de/nipun.arora/bimap-dendrite-segmentation.git
   ```

2. **Installing Dependencies**

   Follow the commands below to create a conda environment and install dependencies.

   ```bash
    cd bimap-dendrite-segmentation
    conda create -n bimap python==3.10
    conda activate bimap
    pip install -r requirements.txt
    ```

3. **Weight and Biases Logging**

    The logger of choice is Weights and Biases, so it is required to perform a setup for it. 
   
   - Create an account on [Weights and Biases](https://wandb.ai/site).
   - Login to your account using `wandb login`.
   - Update the `wandb` section in the `configs/logger/wandb.yaml` file with your project name and entity.
   - Run the training script with the `logger=wandb` argument.
  
<br>

### 🗃️  Data

The data should be stored in the datasets directory or a symbolic soft link can be created to target the dataset in this directory. Using symbolic soft links is recommended to avoid duplication of large datasets and to separate the storage locations for code and data.

For example, the deepd3 dataset can be linked in the `datasets` directory as follows:

```bash
cd datasets
ln -s /absolute/path/to/DeepD3_Training datasets/DeepD3_Training
ln -s /absolute/path/to/DeepD3_Validation datasets/DeepD3_Validation
```

This will create a symbolic link from the original source to the one in the repo.

<br>

### 🏋️  SAM Weights

The SAM weights should be copied into the folder `sam_weights`. You can chose from the different SAM model type mentioned below

```bash
vit_h (recommended): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

<br>

### 🤖  Training

This is where the fun begins, configure params in ```configs.py```

To start training follow the code below

```bash
# Uses the params from configs.py and starts a training job
python trainer.py
```

<br>

### 🧪  Evaluation

To start testing follow the code below

```bash
# Uses the params from configs.py and starts a testing job
python test.py
```

❗️If you don't have a comparison SWC file, please change `line 16` in `configs.py` 

```bash
# Keep it as '' if you don't have masks to compare
parser.add_argument("-swc_filename", type=str, default = "") 
```

<br>

### ✨  Bonus

#### ⏱️ Schedule a Job on HPC

For interactive scheduling a job on the HPC, you can use the `schedule.sh` script. The script is designed to interactively schedule the training script on the HPC with the specified configuration.

```bash
# Schedule a job on the HPC interactively
source schedule.sh
```

#### Useful VSCode Extensions

- [Remote Explorer](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer) - Open projects on remote servers.
- [Log Viewer](https://marketplace.visualstudio.com/items?itemName=berublan.vscode-log-viewer) - A log monitoring extension.
- [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) - Python auto code formatter.
- [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) - Markdown preview and editing.

<br>

### 📬  Contact

Nipun Arora - nipun.arora@fau.de

<br>

### 📚  References

Basic code and project is inspired by [SAM-OCTA](https://github.com/ShellRedia/SAM-OCTA.git).

<br>
