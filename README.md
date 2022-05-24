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

```bibtex
@article{ilboudo2022adaterm,
  title={AdaTerm: Adaptive T-Distribution Estimated Robust Moments towards Noise-Robust Stochastic Gradient Optimizer},
  author={Ilboudo, Wendyam Eric Lionel and Kobayashi, Taisuke and Sugimoto, Kenji},
  journal={arXiv preprint arXiv:2201.06714},
  year={2022}
}
```
