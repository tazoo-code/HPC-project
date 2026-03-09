# HPC-project

### Installation

#### Conda environment
```
conda create --name hpc python=3.10.19 pip=26.0.1 setuptools=80.10.2 -y
conda activate hpc
pip install -r pip_reqs.txt
```

#### Singularity container
```
sudo singularity build singularity.sif singularity.def
```

### Preprocessing
```
python preprocess.py
```

### Training
```
./train.sh
```

### Inference
```
python inference.py
```

### Cluster
```
sbatch slurm_preprocess.slurm
sbatch slurm_train.slurm
```
for slurm_train.slurm, can insert optional keywords: {`--gres=gpu:rtx:1`, `--gres=gpu:rtx:2`, `--gres=gpu:rtx:4`} after `sbatch`