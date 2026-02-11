---
draft: false
title: Notes on adaptive Bayesian experimental design
date: 2024-02-01
lastmod: 2026-02-11
---
{{< katex >}}
Continuing from [Bayesian experimental design]({{< ref "bayesian experimental design" >}}),
we  now consider adaptivity.
<!--more-->
Typically, we choose designs \\(D_1\\), \\(D_2\\), and \\(D_3\\) sequentially to gather measurements.
Let's focus on 3 observations \\(y_1\\), \\(y_2\\) and \\(y_3\\).
We could just optimize the entropy of the posterior of all 3 measurements once at the start of the experiment:
$$
\argmin_{D_{1:3}} \iiint S\left (\theta | y_{1:3}, D_{1:3} \right) p(y_{1:3}|D_{1:3}) dy_{1:3}.
$$
This is called a static experiment.
This technique does not use the opportunity to adapt the experimental design,
after each measurement is gathered.

A popular adaptive experimental design technique minimizes the expected entropy of the posterior given one additional measurement:
$$
D_1^\ast = \argmin_{D_1} \int S\left (\theta |y_1, D_1 \right) p(y_1|D_1) dy_1,
$$
$$
D_2^\ast = \argmin_{D_2} \int S\left (\theta | y_1^\ast, D_1^\ast, y_2, D_2 \right) p(y_2|y_1^\ast, D_1^\ast, D_2) dy_2,
$$
Here \\(D_1^\ast\\) is the optimized and actually performed experiment, and \\(y_1^\ast\\) is the observed measurement value.
$$
D_3^\ast = \argmin_{D_3} \int S\left (\theta | y_{1:2}^\ast, D_{1:2}^\ast, y_3, D_3 \right) p(y_3|y_{1:2}^\ast, D_{1:2}^\ast, D_3) dy_3
$$
Similarly, in the third experiment, the optimal values \\(D_{1:2}^\ast\\) and observed \\(y_{1:2}^\ast\\) are used. 

This method thus requires us to keep track of the posterior after every new measurement is gathered.
Performing the outer integration is also not straightforward.
To perform Monte Carlo integration, we need to sample from \\(p(y_2|y_1, D_{1:2})\\):
$$
p(y_2|y_1, D_{1:2}) = \int p(y_2, \theta|y_1, D_{1:2}) d\theta = \int p(y_2|\theta, D_2)p(\theta|y_1, D_1)d\theta.
$$
This method is called myopic adaptive Bayesian experimental design, because it only looks one observation ahead.
The good aspect of this method is that it allows us to use the information of \\(y_1\\) to plan \\(D_2\\).
The bad aspect, compared to the static method, is that it is greedy,
the method does not carefully plan multiple experiments into the future, like the static method does.

We can combine the good aspects of both, by using the following 3 experiments:
$$
D_1^\ast = \left\lbrack \argmin_{D_{1:3}} \iiint S\left (\theta | y_{1:3}, D_{1:3} \right) p(y_{1:3}|D_{1:3}) dy_{1:3} \right\rbrack_1,
$$
$$
D_2^\ast = \left\lbrack \argmin_{D_{2:3}} \iint S\left (\theta | y_1^\ast, D_1^\ast, y_{2:3}, D_{2:3} \right) p(y_{2:3}|y_1^\ast, D_1^\ast, D_{2:3}) dy_{2:3} \right\rbrack_1,
$$
$$
D_3^\ast = \argmin_{D_3} \int S\left (\theta | y_{1:2}^\ast, D_{1:2}^\ast, y_3, D_3 \right) p(y_3|y_{1:2}^\ast, D_{1:2}^\ast, D_3) dy_3.
$$
Here we use the same entropy of the posterior after all 3 measurements have been gathered, except with the known optimal values \\(D_i^\ast\\) and observations \\(y_i^\ast\\) already filled in.

We will call this fully adaptive Bayesian experimental design.
The major downside of this method is that a very challenging optimization method has to be solved in between measurements.
Even if this challenge could be overcome, there is still one imperfection in this method:
the experiment is adapted after every measurement, but the optimization criteria do not "know" this.
The criteria encode the best experiments if they would be continued in a static fashion.

To remedy this we need the following criterion:
$$
D_1^\ast = \argmin_{D_1} \int \left ( \argmin_{D_2}  \int \left (  \argmin_{D_3} S (\theta | y_{1:3}, D_{1:3}) p(y_3|y_{1:2}, D_{1:3}) dy_3\right ) p(y_2|y_1, D_{1:2}) dy_2\right ) p(y_1|D_1) dy_1.
$$
The difference here is that the optimizations are pushed into the expectations, the optimization of \\(D_2\\) can thus condition on \\(y_1\\) and \\(D_1\\).
After \\(y_1^\ast\\) has been observed and \\(D_1^\ast\\) performed, we get:
$$
D_2^\ast = \argmin_{D_2}  \int \left (  \argmin_{D_3} S (\theta | y_1^\ast, D_1^\ast, y_{2:3}, D_{2:3}) p(y_3|y_1^\ast, D_1^\ast, y_2, D_{2:3}) dy_3\right ) p(y_2|y_1^\ast, D_1^\ast, D_2) dy_2.
$$
After \\(y_2^\ast\\) has been observed and \\(D_2^\ast\\) performed, we get:
$$
D_3^\ast = \argmin_{D_3} \int S (\theta | y_{1:2}^\ast, D_{1:2}^\ast, y_3, D_3) p(y_3|y_{1:2}^\ast, D_{1:2}^\ast, D_3) dy_3.
$$

Notice that the later criteria are present in the earlier criteria.
To find \\(D_1^\ast\\) we need to calculate the criterion used to find \\(D_2^\ast\\) after \\(y_1^\ast\\) has been gathered.
In fact we need to calculate this criterion for every possible \\(y_1^\ast\\) which could have been gathered.
We call this policy-based Bayesian experimental design.
The optimization problem has the structure of a dynamic programming problem.
Calculating an optimal policy allows us to get rid of the online computations present in fully adaptive Bayesian experimental design,
by using the pre-computed dynamic programming structure.

However, when observations \\(y\\) are continuous, we cannot use tabular dynamic programming:
the state space of possible observations is infinite.
Approximate dynamic programming methods are required to solve these optimization problems in practice,
which is the focus of much current experimental design literature, e.g. [Deep Adaptive Design](https://arxiv.org/abs/2103.02438).

For a direct continuation that formalizes why this nested criterion is exactly a policy-optimization problem,
see [policy equivalence in adaptive Bayesian experimental design](/posts/policy-equivalence-in-adaptive-bayesian-experimental-design/).
