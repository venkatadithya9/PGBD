# PGBD
Official code repository for the ICCV 2025 paper "Prototype Guided Backdoor Defense via Activations Space Manipulation".

## Prerequisite
- PyTorch 2.4.1

## Usage
The main file is located at `src/proto_based.py`. Running this without any arguments will run PGBD on the default checkpoint located at `src/weights/`. 

```bash
cd src/
python3 proto_based.py
```
The file uses wandb by default for logging, so make sure to initialize wandb before running the file for the plots.

The `src/proto_based.py` script provides different options:
- `--update_pav`: bool, whether to do periodic update of prototypes or not. If toggled, updates are done at every `update_gap` number of epochs.
- `--weight_pav`: bool, if updating, should a slow update be performed on the PAVs.
- `--weight_proto`: bool, if updating, should a slow update be performed on the prototypes.
- `--lambda`: float [0,1], weightage given to sanitization loss in the overall criteria.
- `--model`: str, name of the architecture of the model to perofrm defense on.
- `--dataset`: str, name of the dataset that is to be used in this experiment.
- `--attack_method`: str, name of the attack against which defene is to be performed. 
- `--trigger_type`: str, type/subtype of triggers if everyone comes in with similar identities.

To run semantic attack based experiments, first download the dataset from [here](https://drive.google.com/drive/folders/1e_FiGx0ShUhnE5X_GhVfxxgsh46DgCeb) and place the dataset at the path `src/data/ROF/`.


## TODOs
1. Remove attack specific evaluation code from `proto_based.py`.

## Acknowledgements
* We obtain all baseline weights from [BackdoorBench](https://github.com/SCLBD/BackdoorBench).
* Most of the main code is adapted from the paper [Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks, ICLR'21](https://github.com/bboylyg/NAD).
* CD_utils is adopted from the paper [Concept Distillation: Leveraging Human-Centered Explanations for Model Improvement, NeurIPS'23](https://github.com/avani17101/CD).
* We use DINO feature extractor from [Dino](https://github.com/dichotomies/N3F) and download DINO checkpoints from [Official DINO repository](https://github.com/facebookresearch/dino).
