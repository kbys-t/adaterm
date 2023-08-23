# AdaTerm

AdaTerm implemented with PyTorch

## how to use

1. install by pip
```bash
git clone https://github.com/kbys-t/adaterm.git
cd adaterm
pip install -e .
```
2. example
```python
# import
import torch
from adaterm import AdaTerm
# define network models (as nets) + learning rate (as lr)
optimizer = AdaTerm(nets.parameters(), lr=lr)
# compute loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## reference

(Peer-reviewed Journal version)
```bibtex
@article{ILBOUDO2023126692,
title = {AdaTerm: Adaptive T-distribution estimated robust moments for Noise-Robust stochastic gradient optimization},
journal = {Neurocomputing},
pages = {126692},
year = {2023},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.126692},
url = {https://www.sciencedirect.com/science/article/pii/S0925231223008159},
author = {Wendyam Eric Lionel Ilboudo and Taisuke Kobayashi and Takamitsu Matsubara},
keywords = {Robust optimization, Stochastic gradient descent, Deep neural networks, Student’s t-distribution}
}
```

(Preprint version)
```bibtex
@article{ilboudo2022adaterm,
  title={Adaterm: Adaptive t-distribution estimated robust moments towards noise-robust stochastic gradient optimizer},
  author={Ilboudo, Wendyam Eric Lionel and Kobayashi, Taisuke and Matsubara, Takamitsu},
  journal={arXiv preprint arXiv:2201.06714},
  year={2022}
}
```
