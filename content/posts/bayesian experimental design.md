---
draft: false
title: Notes on Bayesian experimental design
date: 2023-08-01
lastmod: 2026-02-11
---
{{< katex >}}
For much of the experimental design literature, Bayesian experimental design is synonymous with the
[expected Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Bayesian_experimental_design#Gain_in_Shannon_information_as_utility).
It is not obvious at first glance why this criterion leads to informative experiments,
so let's dive a bit deeper into Bayesian experimental design.
<!--more-->
In Bayesian experimental design, we start with a prior, \\(p(\theta)\\).
This prior represents our state of belief in possible values of the uncertain parameters, \\(\theta\\).
We then have to pick a design \\(D\\), to learn as much as possible about \\(\theta\\).
After the design has been chosen, a new measurement \\(y\\) is gathered.
This measurement is then used to update our states of belief to the posterior distribution \\( p(\theta |y, D)\\).
We want this posterior distribution to be spread as narrowly as possible.
But what does it mean for a distribution to be narrow?
The most popular tool to quantify the spread of a distribution in Bayesian experimental design is entropy.
The entropy of a distribution \\(X\\) is defined as:
$$
S(X) = -\int \ln p(X) p(X)dX
$$

I might write a future post about an intuitive understanding of entropy, but for now, you need to know which distributions have high/low entropy.

For univariate distributions supported on the interval \\([a,b]\\),
the uniform distribution on the interval \\([a,b]\\),
is the distribution with the maximum entropy.
The uniform distribution is equivalent to the state of belief where no value is to be preferred over any other.

This is different from other measures to quantify spread, such as variance.
The distribution with maximal variance would be the 50%-50% mixture distribution of two Dirac delta distributions at \\(a\\) and \\(b\\).

The distribution with the lowest entropy, and also the lowest variance, is any Dirac delta distribution \\(\delta(\theta - c)\\) with \\( c \in [a,b]\\).
This Dirac delta distribution corresponds to a state of belief where you are absolutely certain \\(c\\) is the correct value.

A succesful experiment thus achieves a posterior with low entropy.
However, at the planning stage of the experiment,
we do not have access to the measurement \\(y\\).
We can only judge the quality of the design \\(D\\) on average over all possible measurements \\(y\\),
weighted by how likely we think those measurements are to occur using our prior:
$$
\argmin_D \left [ \int \left (-\int \ln p(\theta |y, D) p(\theta |y, D) d\theta \right) p(y|D) dy \right ].
$$
After a bunch of math tricks, this can be rewritten as the same equation in
[the already mentioned wiki article](https://en.wikipedia.org/wiki/Bayesian_experimental_design#Gain_in_Shannon_information_as_utility).
$$
\argmax_D \left [ \iint \ln p(y|\theta, D) p(\theta,y| D) d\theta dy  - \int \ln p(y|D) p(y|D) dy\right ].
$$
I am unsure why entropy is the default choice to quantify spread in Bayesian experimental design,
this is definitely not the case in frequentist experimental design, where variance is used.
I have heard arguments that the above rewritten expression is easier to calculate, since it does not rely on the posterior anymore.
For a criterion based on the variance, algebraic manipulations to get rid of the posterior are not (known to be?) possible.
However, I do not find this convincing since you still need to evaluate \\(\ln p(y|D)\\) in the rewritten expression,
which will still require solving the integral:
$$
p(y|D)  = \int p(y, \theta|D) d\theta.
$$
Thus, unfortunately, a nested numerical integration technique will still be required to generate Bayesian optimal designs.

For continuations that add sequential decision-making and the policy interpretation, see
[adaptive Bayesian experimental design]({{< ref "adaptive bayesian experimental design" >}})
and
[policy equivalence in adaptive Bayesian experimental design](/posts/policy-equivalence-in-adaptive-bayesian-experimental-design/).
