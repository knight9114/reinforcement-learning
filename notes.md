---
title: Reinforcement Learning Notes
header-includes: |
    \usepackage{bbm}
    \usepackage{amsmath}
    \DeclareMathOperator*{\argmax}{argmax}
---

\newcommand{\Qn}{Q_ n}
\newcommand{\Qnpo}{Q_ {n + 1}}
\newcommand{\Rn}{R_ n}
\newcommand{\Ri}{R_ i}

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
    Q_ t(a) = \frac{\sum_{i=1}^{t - 1} \Ri \cdot \mathbb{I}_ {A_ i = a}}
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

Computing the action-value estimate requires storing every observed action and
reward. This requires a significant amount of memory for long running
experiments. It is possible to incrementally compute the updates via

\begin{equation} 
    \begin{split}
    \Qnpo & = \frac{1}{n} \sum_ {i=1}^n \Ri \\
    & = \frac{1}{n} \left( \Rn + \sum_ {i=1}^{n-1} \Ri \right) \\
    & = \frac{1}{n} \left( \Rn + (n - 1) \frac{1}{n - 1} \sum_ {i=1}^{n-1} \Ri \right)\\
    & = \frac{1}{n} \left( \Rn + (n - 1) \Qn \right) \\
    & = \frac{1}{n} \left( \Rn + n\Qn - \Qn \right) \\
    & = \Qn + \frac{1}{n} \left[ \Rn - \Qn \right]
    \end{split}
\end{equation}

This update rule has a general form:

$$ NewEstimate \leftarrow OldEstimate + StepSize \left[ Target - OldEstimate \right] $$

The expression $\left[ Target - OldEstimate \right]$ is an \emph{error} in the
estimate. By taking a step toward the $Target$, it reduces the error, assuming
of course that $Target$ represents a desirable direction.

\pagebreak
## Section 2.5 - Tracking a Nonstationary Problem

The averaging methods discussed previously are suitable for stationary bandit
problems; however, they fail when the reward probabilities shift over time. In
such cases, it makes sense to weight the more recent rewards more than the
previous ones. This is a simple change

$$ \Qnpo \doteq \Qn + \alpha \left[ \Rn - \Qn \right] $$

where the step-size parameter $\alpha \in (0, 1]$ is constant. This results in
$\Qnpo$ being a weighted average of the past rewards and the initial estimate
$Q_ 1$:

\begin{equation}
    \begin{split}
    \Qnpo & = \Qn + \alpha \left[ \Rn - \Qn \right] \\
         & = \alpha \Rn + (1 - \alpha) \Qn \\
         & ... \\
         & = (1 - \alpha)^n Q_ 1 + \sum_ {i=1}^n \alpha(1 - \alpha)^{n-i} \Ri
    \end{split}
\end{equation}

A few notes on this expansion:

* $(1 - \alpha)^n + \sum_ {i=1}^n \alpha (1 - \alpha)^{n-i} = 1$
* The weight given to \Ri decays exponentially according to $(1 - \alpha)$
* This method is known as \emph{exponential recency-weighted average}

Sometimes it is convenient to vary the step-size parameter, noted as $\alpha_
n(a)$, based on the action, $a$, taken at time, $n$. In the constant step-size
variants, $\alpha_ n(a) = \alpha$. For the sample-average method, $\alpha_ n(a)
= \frac{1}{n}$, which we know converges due to the Law of Large Numbers.

A well-known result in stochastic approximation theory gives the following
conditions required to assure that the sequence $\{ \alpha_ n(a) \}$
converges with absolute certainty:

$$ 
\sum_ {n=1}^{\infty} \alpha_ n(a) = \infty 
    \quad \text{and} \quad
\sum_ {n=1}^{\infty} \alpha^2_ n(a) < \infty
$$

\pagebreak
## Section 2.7 - Upper-Confidence-Bound Action Selection

Exploration is needed because there is always uncertainty about the accuracy of
the action-value estimates. $\epsilon-greedy$ action selection forces
non-greedy actions; however, there is no strategy given to the exploration.
Ideally, it would be best to select non-greedy actions based on the uncertainty
of the reward. One effective way of doing this is to select actions according to

$$ A_ t \doteq \argmax_ a \left[ Q_ t(a) + c \sqrt{\frac{\ln t}{N_ t(a)}} \right] $$

where $N_ t(a)$ is a count of how many times each action has been taking. When
$N_ t(a_ i) = 0$, the action $a_ i$ is considered the most uncertain action to
take.

The driving idea behind the \emph{upper confidence bound} is to estimate the
uncertainty in the rewards for each action. As an action is taken, $N_ t(a)$
increases, which reduces the uncertainty in the reward given; however, as an
action is not taken, $\ln t$ increases which increases the uncertainty of said
action. 

There are a few issues with UCB action selection:

* It is more difficult to extend beyond simple bandit problems
* UCB struggles with nonstationary reward distributions
* Learning large state-spaces takes a long time to converge

In these more advanced situtations, the fundamental idea of UCB action
selection is usually not practical.

\pagebreak
## Section 2.8 - Gradient Bandit Algorithms

All of the previously mentioned methods require an estimate of the action-value
function; however, there are other ways of choosing actions. The following
method learns a \emph{preference} for each action $a$, which is donoted $H_
t(a)$. The larger the preference, the more often that action is taken. These
preferences, however, have no relation to the actual reward function - if each
action's preference was increased by $1000$, then that would have no effect on
the action probabilities. The asformentioned probabilities are represented by
the Gibbs Distribution or \emph{soft-max distribution}:

$$ Pr\{A_ t = a\} \doteq \frac{e^{H_ t(a)}}{\sum_ {b=1}^k e^{H_ t(b)}} \doteq \pi_ t(a) $$

where $\pi_ t(a)$ is the probability of taking action $a$ at time step $t$. The
initial prefences for each action start at $0$ and are learned through
interaction with the environment.

To learn the preferences, stochastic gradient ascent is is used. On each step,
after selecting action $A_ t$ and receiving reward $R_ t$, the action
preferences are updated with the following rule:

\begin{equation}
    \begin{alignedat}{2}
    H_ {t+1}(A_ t) & \doteq H_ t(A_ t) + \alpha (R_ t - \bar{R_ t}(1 - \pi_ t(A_ t)), & & \text{and} \\
    H_ {t+1}(a) & \doteq H_ t(a) - \alpha (R_ t - \bar{R_ t})\pi_ t(a), &\quad & \text{for all } a \ne A_ t,
    \end{alignedat}
\end{equation}

where $\alpha > 0$ is a step-size parameter, and $\bar{R_ t} \in \mathbb{R}$ is
the average of all the rewards up through and including time $t$. The $\bar{R_
t}$ serves as a \emph{baseline}. When an action's reward is below the baseline,
the probability of taking that action again is decreased. Rewards above the
baseline increase the probability of that action being taken again.

\pagebreak
# Chapter 3 - Finite Markov Decision Processes

## Section 3.1 - Agent-Environment Interface

Markov Decision Processes, MDPs, are a classical formalization of sequential
decision making, where actions influence not just immediate rewards, but also
subsequent situations. In the bandit problem, we estimate the value function,
$q_ * (a)$, for every action $a$. In the MDPs, we estimate the value function,
$q_ * (s, a)$, for each action, $a$, or equivalently the _state-value
function_, $v_ * (s)$.

Definitions:

* _agent_: The learner or decision maker
* _environment_: The thing the agent interacts with

The agent and the environment interact with each other continually - the agent
selects actions given the environment, and the environment changes with each
action. More specifically, the agent and environment interact at each of a
sequence of discrete time steps, $t = 0, 1, 2, ...$. For each time step, the
agent receives a representation of the environment's current _state_, $S_ t \in
\mathcal{S}$, and then selections an action, $A_ t \in \mathcal{A}(S_ t)$. On
the next time step, the agent receives the next state, $S_ {t+1}$, and a
reward, $R_ {t+1}$. This process results in a _trajectory_ that begins like
this:

$$ S_ 0, A_ 0, R_ 1, S_ 1, A_ 1, R_ 2, S_ 2, A_ 2, R_ 3 $$

In a _finite_ MDP, the sets of states, actions, and rewards $(\mathcal{S},
\mathcal{A}, \mathcal{R})$ all have a finite number of elements. For particular
values of $s' \in \mathcal{S}$ and $r \in \mathcal{R}$, there is a probability
of those values occurring at time $t$, given particular values of the
preceding state and action:

$$ p(s', r | s, a) \doteq Pr\{S_ t = s', R_ t = r | S_ {t-1} = s, A_ {t-1}=a\} $$

The function $p$ defines the _dynamics_ of the MDP. The dynamics function
\linebreak $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times
\mathcal{A} \rightarrow [0, 1]$ is an ordinary deterministic function of four
arguments. In a _Markov_ process, the probabilities given by $p$ completely
characterize the environment's dynamics. Given the dynamics function, it is
possible to compute anything else one might want to know about the environment:

\pagebreak

_State-Transition Probabilities_
 
$$
p(s' | s, a) \doteq Pr\{S_ t = s' | S_ {t-1} = s, A_ {t-1}=a\} = \sum_{r \in
\mathcal{R}} p(s', r | s, a)
$$

_Expected State-Action Rewards_

$$ 
r(s, a) \doteq \mathbb{E}\left[ R_ t | S_ {t-1} = s, A_ {t-1}=a\right] =
\sum_{r \in \mathcal{R}} r \sum_ {s' \in \mathcal{S}} p(s', r | s, a)
$$

_Expected State-Action-Next-State Rewards_

$$
r(s, a, s') \doteq \mathbb{E}\left[R_ t | S_ {t-1}=s, A_ {t-1}=a, S_
t=s'\right] = \sum_ {r \in \mathcal{R}} r \frac{p(s', r | s, a)}{p(s' | s, a)}
$$

## Section 3.2 - Goals and Rewards

The purpose of reinforcement learning is maximizing the long-term reward, not
the short-term reward. This can be state formally as the _reward hypothesis_:

> That all of what we mean by goals and purposes can be well thought of as the
> maximization of the expected value of the cumulative sum of a received scalar
> signal (called reward).

The reward signal is _not_ the place to impart prior knowledge to the agent.
Instead, do that via the initial policy or the initial value function. The
reward signal is your way of communicating to the agent _what_ you want to
achieve, not _how_ you want it achieved.

## Section 3.3 - Returns and Episodes

In general, the agent seeks to maximize the _expected return_, $G_ t$, is
defined as some specific function of the reward sequence. The simplest case is
just:

$$ G_ t \doteq \sum_ {i=t+1}^T R_ i $$

where $T$ is the final time step. This method works well when the environment
can be naturally broken up into subsequences, called _episodes_ or _trials_.
Environments such as that are called _episodic tasks_. It is often necessary to
discern between terminal and non-terminal states. We denote the set of
non-terminal states as $\mathcal{S}$ and the set of all states, including
terminal ones, as $\mathcal{S}^+$. Tasks that do not have natural terminal
states are called _continuous tasks_, which makes computing the return
formulation difficult since $T = \infty$.

This naive approact to expected return does not take into consideration how far
away the reward is from the current time step. To handle this, we introduce
_discounting_. According to this approach, the agent tries to select actions so
that the sum of the discounted rewards is maximized. In particular, it chooses
$A_ t$ to maximize the expected _discounted return_:

$$ G_ t \doteq \sum_ {k=0}^\infty \gamma^k R_ {t + k + 1} $$

where $\gamma$ is the _discount rate_ such that $\gamma \in [0, 1]$. It is
possible to reformulate the expected discounted return using a familiar-looking
equation:

\begin{equation}
    \begin{split}
    G_ t & \doteq R_ {t+1} + \gamma R_ {t+2} + \gamma^2 R_ {t+3} + \cdots \\
         & = R_ {t+1} + \gamma \left( R_ {t+2} + \gamma R_ {t+3} + \cdots \right) \\
         & = R_ {t+1} + \gamma G_ {t+1}
    \end{split}
\end{equation}
