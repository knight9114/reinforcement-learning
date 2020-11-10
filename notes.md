---
title: Reinforcement Learning Notes
header-includes: |
    \usepackage{bbm}
    \usepackage{amsmath}
    \DeclareMathOperator*{\argmax}{argmax}
---


# Chapter 2: Multi-armed Bandits

## Section 2.1 The k-Armed Bandit Problem

Definitions

* $k$ - Number of possible actions available
* $A_ t$ - Action taken at time $t$
* $R_ t$ - Reward of taking action $A_ t$
* $q_ * (a) \doteq \mathbb{E}[R_ t | A_ t = a]$ - Expected reward of action $a$
* $Q_ t(a)$ - Estimated value function at time $t$ for action $a$

## Section 2.2 - Action-value Methods

\emph{Action-value methods} are methods that approximate the true value
function and then use that estimate for making decisions. Given that the true
value of an action is the mean reward of taking said action, one natural way to
estimate the value function is:

$$ 
    Q_ t(a) = \frac{\sum_{i=1}^{t - 1} R_ i \cdot \mathbb{I}_ {A_ i = a}}
                   {\sum_{i=1}^{t - 1} \mathbb{I}_ {A_ i = a}}
$$

Given a way to estimate the value function, how can one use it to make
decisions? The simplest rule is to select the action with the highest estimated
value, known as the \emph{greedy} strategy, $$A_ t \doteq \argmax_ x Q_ t(a)$$
Greedy action selection always exploits current knowledge to maximize immediate
reward; however, this will not discover new, potenially more rewarding,
actions. A simple alternative to this is to behave greedily most of the time,
but occasionally try random actions with a small probability, $\epsilon$. These
near-greedy strategies are known as \emph{$\epsilon$-Greedy methods}.

## Section 2.4 - Incremental Implementation 

Computing the action-value estimate requires storing every observed action and reward. This requires a significant amount of memory for long running experiments. It is possible to incrementally compute the updates via

\begin{equation} 
    \begin{split}
    Q_ {n+1} & = \frac{1}{n} \sum_ {i=1}^n R_ i \\
    & = \frac{1}{n} \left( R_ n + \sum_ {i=1}^{n-1} R_ i \right) \\
    & = \frac{1}{n} \left( R_ n + (n - 1) \frac{1}{n - 1} \sum_ {i=1}^{n-1} R_ i \right)\\
    & = \frac{1}{n} \left( R_ n + (n - 1) Q_ n \right) \\
    & = \frac{1}{n} \left( R_ n + nQ_ n - Q_ n \right) \\
    & = Q_ n + \frac{1}{n} \left[ R_ n - Q_ n \right]
    \end{split}
\end{equation}

This update rule has a general form:

$$ NewEstimate \leftarrow OldEstimate + StepSize \left[ Target - OldEstimate \right] $$

The expression $\left[ Target - OldEstimate \right]$ is an \emph{error} in the estimate. By taking a step toward the $Target$, it reduces the error, assuming of course that $Target$ represents a desirable direction.


