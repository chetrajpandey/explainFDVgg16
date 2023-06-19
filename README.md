## Explaining Full-disk Deep Learning Model for Solar Flare Prediction using Attribution Methods

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) 
[![python](https://img.shields.io/badge/Python-3.7.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.10.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=captum&style=flat&color=pink&label=lib&message=captum-0.5.0)](https://captum.ai/)

This repo presents a post hoc analysis of a deep learning-based full-disk solar flare prediction model using three post hoc attention methods: (i) Guided Gradient-weighted Class Activation Mapping, (ii) Deep Shapley Additive Explanations, and (iii) Integrated Gradients.

### Source Code Documentation

#### 1. download_mag:
This folder/package contains one python module, "download_jp2.py". There are two functions inside this module. First Function: "download_from_helioviewer()" downloads jp2 magnetograms from helioveiwer api : Helioviewer Second Function: "jp2_to_jpg_conversion()" converts jp2s to jpgs for faster computation. If resize=True, pass height and width to resize the magnetograms

#### 2. data_labeling:
Run python labeling.py : Contains functions to generate labels, binarize, filtering files, and creating 4-fold CV dataset.
Reads goes_integrated_flares.csv files from data_source.
Generated labels are stored inside data_labels. 
labeling.py generates labels with multiple columns that we can use for post result analysis. Information about flares locations, and any other flares that occured with in the period of 24 hours.
For simplification:  folder inside data_labels, named simplified_data_labels that contains two columns: the name of the file and actual target that is sufficient to train the model.

#### 3. modeling:
 This code requires two GPUS, for the current batchsize for VGG16 Model, and the code is configured to use two GPUS for while training.
 Modify the line that uses nn.DataParallel in train.py to use a single GPU. 
 <br/>

(a) model.py: This module contains the architecture of our model. Passing train=True utilizes the logsoftmax on the final activation. To get the probabilities during model predictions, pass train=False, and apply softmax to obtain the probabilities.<br /> 
(b) dataloader.py: This contains custom-defined data loaders for loading FL and NF class for selected augmentations.<br /> 
(c) evaluation.py: This includes functions to convert tensors to sklearn compatible array to compute confusion matrix. Furthermore TSS and HSS skill scores definition.<br /> 
(d) train.py: This module is the main module to train the model. Uses argument parsers for parameters change. This has seven paramters to change the model configuration:<br /> 
(i) --fold: choose from 1 to 4, to run the corresponding fold in 4CV-cross validation; default=1<br /> 
(ii) --epochs: number of epochs; default=50<br /> 
(iii) --batch_size: default=64<br /> 
(iv) --lr: initial learning rate selection; default=0.0001<br /> 
##### The three arguments below are used for learning rate scheduler and are optional for the code. The code uses manual lr scheduling which halves the learning rate every 5 epochs. To use the lr_scheduler uncomment the line:210 and 300, while commenting line: 226 to 228 in train.py 
(v) --weight_decay: regularization parameter used by the loss function; default=0.01<br /> 
(vi) --patience: lr scheduler parameter used to reduce learning rate at specified value which indicates the fold tolerance; default=4<br /> 
(vii) --factor: lr scheduler parameter determining the quantitiy by which the learning rate is to be reduced; default=0.03<br /> 

We used HPCE (SLURM) to run our models as jobs. For this code to work outside of SLURM, remove the line 45: job_id = os.getenv('SLURM_JOB_ID') in train.py  and specify a directory instead. However, for the SLURM env, an example bash script to run a job is given below: <br /> 

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=1-23:59:59
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --mail-user=user@email.institution.edu
#SBATCH --account=account_name
#SBATCH --partition=qGPU48
#SBATCH --gres=gpu:V100:2
#SBATCH --output=../outputs/output_%j
#SBATCH --error=../errors/error_%j
cd /scratch
mkdir $SLURM_JOB_ID
cd $SLURM_JOB_ID

#If using IRODS environment, then get your data from the IRODS to sever
iget -r code_or_data_directory
source /userapp/virtualenv/name_of_virtual_env/venv/bin/activate

python directory_to/modeling/train.py --fold=1 --batch_size=64 --lr=0.00001 --weight_decay=0.0001 --max_lr=0.0001 --models alex

iput -rf $SLURM_JOB_ID
```

