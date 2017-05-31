---
title: Deep Reinforcement Learning
author: Julien Vitay - <julien.vitay@informatik.tu-chemnitz.de>
abstract: The goal of this document is to summarize the state-of-the-art in deep reinforcement learning. It starts with basics in reinforcement learning and deep learning to introduce the notations. It is then composed of three main parts 1) value-based algorithms (DQN...) 2) policy-gradient based algorithms (DDPG...) 3) Recurrent attention models (RAM...). Finally it provides code snippets to the "official" implementation of these algorithms. This document is work in progress and will be updated when new algorithms are published.
autoSectionLabels: True 
---

# Introduction

Deep reinforcement learning (deep RL) is the successful interation of deep learning methods, classically used in supervised or unsupervised learning contexts, with reinforcement learning (RL), a well-studied adaptive control method used in problems with delayed and partial feedback [@Sutton1998]. This section starts with the basics of RL, mostly to set the notations, and provides a quick overview of deep neural networks.

## Basic reinforcement learning 

### MDP: Markov Decision Process

Reinforcement learning problems are modeled as \emph{Markov Decision Processes} (MDP), with a state space $\mathcal{S}$, an action space $\mathcal{A}$, a transition dynamics model with density $p(s_{t+1}|s_t, a_t)$ and a reward function $r(s_t, a_t) : \mathcal{S}\times\mathcal{A} \rightarrow \Re$. The policy is defined as a mapping of the state space into the action space: a stochastic policy $\pi_\theta : \mathcal{S} \rightarrow P(\mathcal{A})$ defines the probability distribution $P(\mathcal{A})$ of performing an action, while a deterministic policy $\mu_\theta(s_t)$ is a discrete mapping of $\mathcal{S} \rightarrow \mathcal{A}$.

### POMDP: Partially Observable Markov Decision Process

See @sec:recurrent-neural-networks

### Bellman equations


The policy can be used to explore the environment and generate trajectories of states, rewards and actions. The performance of a policy is determined by calculating the \emph{expected discounted return}, i.e. the sum of all rewards received from time step t onwards: $R_t = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k+1}$, where $0 < \gamma < 1$ is the discount rate and $r_{t+1}$ represents the reward obtained during the transition from $s_t$ to $s_{t+1}$. The Q-value of an action $a$ is defined as the expected discounted reward if the agent takes $a$ from a state $s$ and follows the policy distribution $\pi_\theta$ thereafter:

$$
    Q^{\pi_\theta}(s, a) = {E}_{\pi_\theta}(R_t | s_t = s, a_t=a)
$$

### Temporal Difference

### Actor-critic architectures

### Function approximation 

$\theta \in \Re^n$ is a vector of parameters defining the policy, typically the weights of a neural network when using function approximators.


The goal of the agent is to find the optimal policy maximizing the expected return from every state. \emph{Value-based} methods (such as DQN) achieve that goal by estimating the Q-value of each state-action pair. Discrete algorithms transform these Q-values into a stochastic policy by sampling from a Gibbs distribution (softmax) to obtain the probability of choosing an action. The Q-values can be approximated by a deep neural network, by minimizing the quadratic error between the predicted Q-value $Q^{\pi_\theta}(s, a)$ and an estimation of the real expected return $R_t$ after that action:

$$
    \mathcal{L}(\theta) = {E}_{\pi_\theta} [r_t + \gamma Q^{\pi_\theta}(s_{t+1}, a_{t+1}) - Q^{\pi_\theta}(s_t, a_t)]^2
$$


## Deep learning

### Deep neural networks

### Convolutional networks

### Recurrent neural networks

# Value-based methods

## Limitations of deep neural networks for function approximation

## DQN: Deep Q Network (Mnih et al. 2013)

## Double DQN

## Prioritised replay

## Duelling network

## GORILA

# Policy-gradient methods

*Policy gradient* methods directly learn to produce the policy (stochastic or not). The goal of the neural network is to maximize an objective function $J(\theta) = {E}_{\pi_\theta}(R_t)$. The \emph{stochastic policy gradient theorem} \cite{Sutton1999} provides a useful estimate of the gradient that should be given to the neural network:

$$
\nabla_\theta J(\theta) = {E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)]
$$

## A3C

## Continuous action spaces

## Policy gradient theorems

## DDPG

## Fictitious Self-Play (FSP)

# Recurrent attention models

# Code samples

# References