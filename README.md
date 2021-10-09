# VGAE 
variational graph autoencoder 

## Requirements
- python=3.8
- tensorflow=2.3.0
- tensorflow-estimator=2.3.0
- tensorboard=2.3.0
- numpy=1.18.5
- matplotlib=3.3.2
- spektral==1.0.6

## How To run example
- unzip graphs.zip 
- open Testing_learning.ipynb and change PATH_FRAMS so it matches local dir of frams 
- run all cells

## How to set up training using miniconda:
- ```conda env create --file=vgae.yaml ```
- ```source ~/miniconda3/bin/activate vgae ```
- ``` unzip graphs.zip ```
- ``` vim gen_config.py ``` -> change line 2 so it match with path to [Framsticks](http://www.framsticks.com/apps-devel) local path. Optionaly change params to train with.
- run using one of the commands:
    -  ```./run_locally.sh $(cat 'configs/<name of th conf file to run>')``` - to run one instance locally
    -  ```find configs/ -type f -iname "*" -exec bash -c 'sbatch -p idss-student run.sh $(cat "$1")' _ {} \;``` - to run all configs using [slurm](https://slurm.schedmd.com/overview.html)
- trained models and plots will save in models/ folder

