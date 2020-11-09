---
Multi-Armed Bandit Notes
---

# What is the Multi-Armed Bandit
## Introduction

Definitions

* $K$ machines with reward probabilities, $\{\theta_1, ..., \theta_K\}$.
* At each time step, $t$, take an action, $a$, on one slot machine and receive
  reward, $r$.
* $\mathcal{A}$ is the set of available actions. The value of the action, $a$,
  is the expected reward, $Q(a) = \mathbb{E}[r|a] = \theta$. If action $a_t$ at
  the time step $t$ is on the i-th machine, then $Q(a_t) = \theta_i$.
* $\mathcal{R}$ is a reward function. In the case of Bernoulli bandits, we
  observe a reward $r$ in a \it{stochastic} fashion.

# Bandit Strategies

Strategies are based on how exploration is handled:

* No exploration: the most naive approach (and a bad one)
* Exploration at random
* Exploration smartly with preference to uncertainty

## $\epsilon-Greedy$ Algorithm

Take the best action most of the time, but does occasional random exploration.
Action value is estimated according to the past experience by averaging the
rewards associated with the target action, $a$, that we have observed so far:

$$ \hat{Q}_t(a) = \frac{1}{N_t(A)} \sum^t_{\tau=1} r_{\tau} \mathbb{I}[a_{\tau} = a] $$

where $\mathbb{I}$ is a binary indicator function and $N_t(a)$ is how many
times the action, $a$, has been selected so far.

## Upper Confidence Bounds

To avoid inefficient exploration, one approach is to decrease the parameter,
$\epsilon$, in time and the other is to be optimistic about options with
$\it{high uncertainty}$ and thus to prefer actions for which we haven't had a
confident value estimation yet. In other words, we favor exploration of actions
with a strong potential to have an optimal value.

The Upper Confidence Bounds, UCB, algorithm measures this potential by an upper
confidence bound of the reward value, $\hat{U}_ t(a)$, so that the true value is
below the bound $Q(a) \le \hat{Q}_ t(a) + \hat{U}_ t(a)$ with high probability.
The upper bound, $\hat{U}_ t(a)$, is a function of $N_ t(a)$.

In the UCS algorithm, we always select the greediest action to maximize the
upper confidence bound:

$$ a_ t^{UCB} = argmax_ {a \in \mathcal{A}} \hat{Q}_ t(a) + \hat{U}_ t(a) $$

### Hoeffding's Inequality

Pick a small probability threshold, $p$, thus:

$$ U_ t(a) = \sqrt{\frac{-\log p}{2N_ t(a)}} $$

### UCB1

One heuristic is to reduce the threshold, $p$, in time as we want to make a
more confident bound estimation with more rewards observed. Setting $p =
t^{-4}$ we get the  $\bf{UCB1}$ algorithm:

$$ U_ t(a) = \sqrt{\frac{2 \log p}{N_ t(a)}} $$

