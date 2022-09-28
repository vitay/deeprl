# Model-based RL

**work in progress**

Model-free: The future is cached into values.

Two problems of model-free:

1. Needs a lot of samples
2. Cannot adapt to novel tasks in the same environment.

Model-based uses an internal model to reason about the future (imagination).

Works only when the model is fixed (AlphaGo) or easy to learn (symbolic, low-dimensional). Not robust yet against model imperfection.

## Dyna-Q

[@Sutton1990a]

<https://medium.com/@ranko.mosic/online-planning-agent-dyna-q-algorithm-and-dyna-maze-example-sutton-and-barto-2016-7ad84a6dc52b>


## Unsorted references

Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images [@Watter2015]

Efficient Model-Based Deep Reinforcement Learning with Variational State Tabulation [@Corneil2018]

Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning [@Feinberg2018]

Imagination-Augmented Agents for Deep Reinforcement Learning [@Weber2017].

Temporal Difference Model TDM [@Pong2018]: <http://bair.berkeley.edu/blog/2018/04/26/tdm/>

Learning to Adapt: Meta-Learning for Model-Based Control, [@Clavera2018]

The Predictron: End-To-End Learning and Planning [@Silver2016a]

Model-Based Planning with Discrete and Continuous Actions [@Henaff2017]

Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics [@Kansky2017]

Universal Planning Networks [@Srinivas2018]

World models <https://worldmodels.github.io/> [@Ha2018]

Recall Traces: Backtracking Models for Efficient Reinforcement Learning [@Goyal2018]

Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning [@Peng2018]

Q-map: a Convolutional Approach for Goal-Oriented Reinforcement Learning [@Pardo2018]
