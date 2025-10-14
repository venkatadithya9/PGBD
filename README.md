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
