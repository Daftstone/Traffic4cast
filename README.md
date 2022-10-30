## The Code for Traffic4cast 2022
### Team: ustc-gobbler

### Download Links

- [LONDON_2022.zip](https://developer.here.com/sample-data) from HERE (2.8 GB)
- [MADRID_2022.zip](https://developer.here.com/sample-data) from HERE (4.0 GB)
- [MELBOURNE_2022.zip](https://developer.here.com/sample-data) from HERE (0.9 GB)
- [T4C_INPUTS_2022.zip](https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_2022.zip) (1.0 GB)
- [T4C_INPUTS_ETA_2022.zip](https://iarai-public.s3-eu-west-1.amazonaws.com/competitions/t4c/t4c22/T4C_INPUTS_ETA_2022.zip) (available September 2, 2022, 1.5MB)

**After downloading and unzipping the data, please revise the data path in “t4c22_config.json”.**

### ****Prepare environment****

```python
conda env update -f environment.yml
conda activate t4c22

# Installing the torch geometric extras is optional, required only if using `torch_geometric`
# install-extras is not passed to pip from environment yaml, therefore add as post-step (https://github.com/conda/conda/issues/6805)
# replace with your CUDA version (cpu, ...), see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA="cu113"
python -m pip install -r install-extras-torch-geometric.txt -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html

python t4c22/misc/check_torch_geometric_setup.py
```

### Generate inputs and labels
**Enter t4c22 folder and run the following commands.**
```python
python prepare_training_data_cc.py --data_folder [DATA_FOLDER]
python prepare_training_data_eta.py --data_folder [DATA_FOLDER]
```

### Run models
**You can choose to train the model from scratch, or use [our trained ones](https://drive.google.com/drive/folders/1IPS8awH8Htmt9hGMa-cE6lwoJg60ttaC?usp=share_link) for testing (put the save folder in the root).**

**train model for core challenge**
```python
python rec_cc.py --city [city] --device [gpu_id] --batch_size 2 --hidden_channels 32 --epochs 20 --fill -1
```
**test model for core challenge**
```python
python rec_cc.py --city [city] --device [gpu_id] --batch_size 2 --hidden_channels 32 --epochs 20 --fill -1 --model_state test
```

**train model for extended challenge**
```python
python rec_eta.py --city [city] --device [gpu_id] --batch_size 2 --hidden_channels 64 --epochs 50
```
**test model for extended challenge**
```python
python rec_eta.py --city [city] --device [gpu_id] --batch_size 2 --hidden_channels 64 --epochs 50 --model_state test
```

### Acknowledgements

This repository is based on [NeurIPS 2022 Traffic4cast](https://github.com/iarai/NeurIPS2022-traffic4cast).
