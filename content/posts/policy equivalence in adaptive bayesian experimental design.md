---
draft: false
title: Notes on policy equivalence in adaptive Bayesian experimental design
date: 2024-04-01
lastmod: 2026-02-11
---
{{< katex >}}
Continuing from [Bayesian experimental design]({{< ref "bayesian experimental design" >}}) and
[adaptive Bayesian experimental design]({{< ref "adaptive bayesian experimental design" >}}),
we can make precise why the nested adaptive criterion is equivalent to an optimal policy problem.
<!--more-->

Consider 3 sequential design choices \\(D_1,D_2,D_3\\) and outcomes \\(Y_1,Y_2,Y_3\\).
Let
$$
u(d_1,d_2,d_3,y_1,y_2,y_3)
= -S(\theta \mid y_{1:3},d_{1:3}),
$$
so maximizing \\(u\\) is the same as minimizing posterior entropy.

The adaptive nested objective can be written as
$$
V
=\max_{d_1\in D_1}
\mathbb E_{Y_1\mid d_1}\Big[
\max_{d_2\in D_2}
\mathbb E_{Y_2\mid Y_1,d_1,d_2}\Big[
\max_{d_3\in D_3}
\mathbb E_{Y_3\mid Y_1,Y_2,d_1,d_2,d_3}
\big[u(d_1,d_2,d_3,Y_1,Y_2,Y_3)\big]
\Big]
\Big].
$$

This is not a myopic rule.
The stage-1 choice is optimized for expected utility over all future observation paths,
while explicitly anticipating that stages 2 and 3 will adapt optimally to whatever data are observed.
Equivalently, we are solving for a *policy* that maps observations to decisions.

A deterministic non-anticipative policy is
$$
\pi=(\pi_1,\pi_2,\pi_3)
$$
with
$$
\pi_1\in D_1,\qquad
\pi_2:D_1\times \mathcal Y_1\to D_2,\qquad
\pi_3:D_1\times \mathcal Y_1\times D_2\times \mathcal Y_2\to D_3.
$$

Given \\(\pi\\), the policy value is
$$
J(\pi)=
\mathbb E\Big[
u\big(
\pi_1,
\pi_2(\pi_1,Y_1),
\pi_3(\pi_1,Y_1,\pi_2(\pi_1,Y_1),Y_2),
Y_1,Y_2,Y_3
\big)
\Big].
$$
Let \\(\Pi\\) be the set of admissible policies.
The key equivalence is
$$
V = \max_{\pi\in\Pi} J(\pi).
$$

The proof has two parts.

First, define the continuation values once:
$$
G_3(d_1,y_1,d_2,y_2)=
\max_{a_3\in D_3}
\mathbb E\big[u(d_1,d_2,a_3,y_1,y_2,Y_3)\mid d_1,y_1,d_2,y_2,a_3\big],
$$
$$
G_2(d_1,y_1)=
\max_{a_2\in D_2}
\mathbb E\big[G_3(d_1,y_1,a_2,Y_2)\mid d_1,y_1,a_2\big],
$$
$$
G_1=\max_{a_1\in D_1}\mathbb E\big[G_2(a_1,Y_1)\mid a_1\big].
$$
By definition of the nested objective, \\(G_1=V\\).

### Part 1: no policy can score higher than V

Fix any policy \\(\pi\\).
For each realized history \\(h_2=(d_1,y_1,d_2,y_2)\\), define the stage-3 value of choosing action \\(a_3\\) as
$$
q_3(a_3;h_2)=\mathbb E\big[u(d_1,d_2,a_3,y_1,y_2,Y_3)\mid h_2,a_3\big].
$$
The policy chooses \\(a_3=\pi_3(h_2)\\), so it gets \\(q_3(\pi_3(h_2);h_2)\\).
Since this is one feasible action in \\(D_3\\),
$$
q_3(\pi_3(h_2);h_2)\le \max_{a_3\in D_3}q_3(a_3;h_2)=G_3(d_1,y_1,d_2,y_2).
$$

Now move one step earlier. For each \\((d_1,y_1)\\), define
$$
q_2(a_2;d_1,y_1)=\mathbb E\big[G_3(d_1,y_1,a_2,Y_2)\mid y_1,d_1,a_2\big].
$$
Again, the policy uses one feasible action \\(a_2=\pi_2(d_1,y_1)\\), so
$$
q_2(\pi_2(d_1,y_1);d_1,y_1)\le \max_{a_2\in D_2}q_2(a_2;d_1,y_1)=G_2(d_1,y_1).
$$

At stage 1, define
$$
q_1(a_1)=\mathbb E\big[G_2(a_1,Y_1)\mid a_1\big].
$$
Because \\(\pi_1\\) is one feasible action in \\(D_1\\),
$$
q_1(\pi_1)\le \max_{a_1\in D_1}q_1(a_1)=G_1,
$$
and from the definitions above, \\(G_1=V\\).

Putting these stage-wise inequalities together gives \\(J(\pi)\le V\\).
Since \\(\pi\\) was arbitrary,
$$
\max_{\pi\in\Pi}J(\pi)\le V.
$$

### Part 2: there exists a policy with value exactly V

Using those same continuation-value definitions, construct \\(\pi^\star\\) by choosing argmax actions at every history.
Then \\(\pi^\star\\) makes exactly the same choices as the nested objective definition of \\(V\\).
Evaluating \\(J(\pi^\star)\\) and applying iterated expectation from stage 3 back to stage 1 yields
$$
J(\pi^\star)=G_1=V,
$$
so
$$
V\le \max_{\pi\in\Pi}J(\pi).
$$

Combining the two parts yields
$$
V=\max_{\pi\in\Pi}J(\pi).
$$

For Bayesian experimental design this matters because it clarifies what is computed offline and online.
The optimization is over decision rules, not over a single static sequence.
Once an approximate optimal policy is available, online adaptation is cheap:
new measurements are plugged into \\(\pi_2\\) and \\(\pi_3\\) without re-solving the full nested problem.

In practice, for continuous observations \\(Y\\), tabular dynamic programming is not available.
A common approximation is to parameterize the policy with neural networks,
for example \\(d_2=\pi_{\phi,2}(d_1,y_1)\\) and \\(d_3=\pi_{\phi,3}(d_1,y_1,d_2,y_2)\\),
where \\(\phi\\) are network weights.
Conceptually, the exact problem is optimization in function space (choose the best measurable mapping from histories to actions).
After parameterization, that infinite-dimensional search is replaced by finite-dimensional nonlinear programming over \\(\phi\\).
Then we simulate trajectories from the Bayesian model, estimate \\(J(\pi_\phi)\\) by Monte Carlo,
and optimize \\(\phi\\) with policy gradient methods.
