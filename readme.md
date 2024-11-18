# Stochastic Q-learning for Large Discrete Action Spaces

In complex environments with large discrete action spaces, effective decision-making is critical in reinforcement learning (RL). Despite the widespread use of value-based RL approaches like Q-learning, they come with a computational burden, necessitating the maximization of a value function over all actions in each iteration. This burden becomes particularly challenging when addressing large-scale problems and using deep neural networks as function approximators. We present stochastic value-based RL approaches which, in each iteration, as opposed to optimizing over the entire set of n
 actions, only consider a variable stochastic set of a sublinear number of actions, possibly as small as O(log(n)). StochDQN integrate this stochastic approach for both value-function updates and action selection.

### Stochastic Q-learnig Paper, [ICML 2024 Paper Link](https://proceedings.mlr.press/v235/fourati24a.html).

### Citing the Project

To cite this repository in publications:

```bibtex

@InProceedings{pmlr-v235-fourati24a,
  title = 	 {Stochastic Q-learning for Large Discrete Action Spaces},
  author =       {Fourati, Fares and Aggarwal, Vaneet and Alouini, Mohamed-Slim},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {13734--13759},
  year = 	 {2024},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/fourati24a/fourati24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/fourati24a.html},
}
```

