# HyperMask: Adaptive Hypernetwork-based Masks for Continual Learning

Generate a semi-binary mask for a target network using a hypernetwork.

![Scheme of HyperMask method](HyperMask.png)

Use <code>environment.yml</code> file to create a conda environment with necessary libraries. One of the most essential packages is [hypnettorch](https://github.com/chrhenning/hypnettorch) which should easy create hypernetworks in [PyTorch](https://pytorch.org/).

## DATASETS

The implemented experiments uses three publicly available datasets for continual learning tasks: Permuted MNIST, Split MNIST and Split CIFAR-100. The datasets may be downloaded when the algorithm runs.

## USAGE

The description of HyperMask is included in the [paper](https://arxiv.org/abs/2310.00113). To perform experiments with the use of the best hyperparameters found and reproduce the results from the publication for five different seed values, one should run <code>main.py</code> file with the variable <code>create_grid_search</code> set to <code>False</code> and the variable <code>dataset</code> set to <code>PermutedMNIST</code>, <code>SplitMNIST</code> or <code>CIFAR100</code>. In the third case, as a target network <code>ResNet-20</code> or <code>ZenkeNet</code> can be selected. To train ResNets, it is necessary to set <code>part = 0</code>, while to prepare ZenkeNets, one has to set <code>part = 1</code>. In the remaining cases, the variable <code>part</code> is insignificant.

One can also easily perform hyperparameter optimization using a grid search technique. For this purpose, one should set the variable <code>create_grid_search</code> to <code>True</code> in <code>main.py</code> file and modify lists with hyperparameters for the selected dataset in <code>datasets.py</code> file.

## CITATION

If you use this library in your research project, please cite the following paper:

```
@misc{książek2023hypermask,  
     title={HyperMask: Adaptive Hypernetwork-based Masks for Continual Learning},  
     author={Kamil Książek and Przemysław Spurek},  
     year={2023},  
     eprint={2310.00113},  
     archivePrefix={arXiv},  
     primaryClass={cs.LG}  
}
```

## LICENSE

Copyright 2023 Institute of Theoretical and Applied Informatics, Polish Academy of Sciences (ITAI PAS) <https://www.iitis.pl> and Group of Machine Learning Research (GMUM), Faculty of Mathematics and Computer Science of Jagiellonian University <https://gmum.net/>.

Authors:<ul>
    <li> Kamil Książek (ITAI PAS, ORCID ID: 0000−0002−0201−6220),
    <li> Przemysław Spurek (Jagiellonian University, ORCID ID: 0000-0003-0097-5521).
</ul>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.