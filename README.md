# Reinforcement Learning for Inventory Dispatch

This repository contains an Inventory Dispatch Environment and a training/loading RL agent. 

## Installation
```
pip install -r requirements.txt
```

## Environments
- `DoDistEnv-v0`: multi-echelon supply chain re-order problem with backlogs (DemandForecast.csv).

## Run
Just simply run the DoDistMain.py file.
```
python examples/DoDistMain.py
```

## References
```
@misc{HubbsOR-Gym,
    author={Christian D. Hubbs and Hector D. Perez and Owais Sarwar and Nikolaos V. Sahinidis and Ignacio E. Grossmann and John M. Wassick},
    title={OR-Gym: A Reinforcement Learning Library for Operations Research Problems},
    year={2020},
    Eprint={arXiv:2008.06319}
}

@misc{author={Kevin Geevers},
    title={Deep Reinforcement Learning in InventoryManagement},
    year={2020},
    month={December},
    Eprint={http://essay.utwente.nl/85432/1/Geevers_MA_BMS.pdf}
}
```