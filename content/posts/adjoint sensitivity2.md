---
draft: false
title: Notes on adjoint sensitivity analysis of dynamic systems part 2
date: 2022-04-30
lastmod: 2022-04-30
---
{{< katex >}}
We continue from [part 1]({{< ref "adjoint sensitivity" >}}) with a more rigorous version of the derivation of adjoint sensitivity analysis for continuous time systems,
$$\begin{align*}
    u(0) &= f_0(p)\\\\
    u(t) &= u(0) + \int_0^{t} f(u(q),p,q)dq\\\\
    c(t) &= g(u(t),p,t)\\\\
    G(c) &= \int_0^{t_e } c(s)ds,
\end{align*}$$
\\(u(t)\\) is the dynamic state, which evolution in time is described by the function \\(f\\).
\\(c(t)\\) is the cost at time \\(t\\) described by the function \\(g\\) and \\(G\\) is the total accumulated cost.
Both \\(g\\) and \\(f\\) are dependent on the parameters \\(p\\) and the time \\(t\\). We want to calculate the effect \\(p\\) has on \\(G\\) using backpropagation.
<!--more-->
Let us assume that we have already pulled back from time \\(t_e\\) to time \\(t\\).
We reparametrize \\(G\\) in terms of \\(p\\), \\(u(t)\\) and \\(c_{[0,t]}\\), which is the cost function restricted to the interval \\([0,t]\\),
$$\begin{align*}
    G(c_{[0,t]},u(t),p) = \int_0^t c(s)ds + &\int_t^{t_e} g(u(s),p,s)ds\\\\
    &\text{with}\qquad u(s) = u(t) + \int_t^s f(u(q),p,q)dq.
\end{align*}$$
If we assume that the partial derivative of \\(G(c_{[0,t]},u(t),p)\\) with regards to \\(u(t)\\) is equal to \\(\lambda(t)\\),

$$\frac{\partial G(c_{[0,t]},u(t),p)}{\partial u(t)} = \frac{\partial \int_t^{t_e} g(u(s),p,s)ds}{\partial u(t)} = \lambda(t),$$

then we can calculate the same partial derivative at a slightly further pulled back timepoint of \\(t-\Delta t\\).

$$\begin{align*}
	\frac{\partial G(c_{[0,t-\Delta t]},u(t-\Delta t),p)}{\partial u(t-\Delta t)}
	&=\frac{\partial \left( \int_0^{t-\Delta t} c(s) ds 
    + \int_{t-\Delta t}^{t_e} g(u(s),p,s)ds\right)}{\partial u(t-\Delta t)} \\\\
	&=\frac{\partial \left( \int_{t-\Delta t}^tg(u(s),p,s)ds
    + \int_t^{t_e} g(u(s),p,s)ds\right)}{\partial u(t-\Delta t)} \\\\
	&=\frac{\partial \int_{t-\Delta t}^tg(u(s),p,s)ds}{\partial u(t-\Delta t)} +
	\frac{\partial \int_t^{t_e} g(u(s),p,s)ds}{\partial u(t)}\frac{\partial u(t)}{\partial u(t-\Delta t)} \\\\
    &=\frac{\partial \int_{t-\Delta t}^tg(u(s),p,s)ds}{\partial u(t-\Delta t)} +
	\lambda(t)\frac{\partial u(t)}{\partial u(t-\Delta t)} \\\\
\end{align*}$$

Using the mean value theorem, we can write the second term as,

$$\begin{align*}
    \frac{\partial u(t)}{\partial u(t-\Delta t)} &=
    \frac{\partial \left( u(t-\Delta t) + \int_{t-\Delta t}^t f(u(q),p,q)dq\right) }{\partial u(t-\Delta t)}\\\\
    & = 1 + \frac{\partial \int_{t-\Delta t}^tf(u(q),p,q)dq}{\partial u(t-\Delta t)}\\\\
	& = 1 + \frac{\partial f(u(t-\Delta t_f),p,t-\Delta t_f)}{\partial u(t-\Delta t)}\Delta t \qquad \Delta t_f \in [0,\Delta t]\\\\
	& = 1 + \frac{\partial f(u(t-\Delta t_f),p,t-\Delta t_f)}{\partial u(t-\Delta t_f)}\frac{\partial u(t-\Delta t_f)}{\partial u(t-\Delta t)}\Delta t\\\\
	& = 1 + \frac{\partial f(u(t-\Delta t_f),p,t-\Delta t_f)}{\partial u(t-\Delta t_f)}\Delta t + \\\\
	& \qquad \frac{\partial f(u(t-\Delta t_{f_2}),p,t-\Delta t_{f_2})}{\partial u(t-\Delta t)}\Delta t(\Delta t - \Delta t_f) \qquad \Delta t_{f_2} \in [\Delta t_f,\Delta t],
\end{align*}$$

and the second term as,

$$\begin{align*}
    \frac{\partial \int_{t-\Delta t}^tg(u(s),p,s)ds}{\partial u(t-\Delta t)} &= 
    \frac{\partial g(u(t-\Delta t_g),p,t-\Delta t_g)}{\partial u(t-\Delta t)}\Delta t \qquad \Delta t_g \in [0,\Delta t] \\\\
	&= \frac{\partial g(u(t-\Delta t_g),p,t-\Delta t_g)}{\partial u(t-\Delta t_g)}\frac{\partial u(t-\Delta t_g)}{\partial u(t-\Delta t)}\Delta t\\\\
	&= \frac{\partial g(u(t-\Delta t_g),p,t-\Delta t_g)}{\partial u(t-\Delta t_g)}\Delta t +  \\\\
	& \qquad \frac{\partial f(u(t-\Delta t_{g_2}),p,t-\Delta t_{g_2})}{\partial u(t-\Delta t)}\Delta t(\Delta t - \Delta t_g) \qquad \Delta t_{g_2} \in [\Delta t_g,\Delta t].
\end{align*}$$


We can obtain a differential equation for \\(\lambda\\) by taking the limit,

$$\begin{align*}
    \frac{d\lambda}{dt} 
    & = \lim_{\Delta t \to 0} \frac{\frac{\partial G(c_{[0,t]},u(t),p)}{\partial u(t)}
    -\frac{\partial G(c_{[0,t-\Delta t]},u(t-\Delta t),p)}{\partial u(t-\Delta t)}}{\Delta t} \\\\
    & = \lim_{\Delta t \to 0} \left( -\frac{\partial g(u(t-\Delta t_g),p,t-\Delta t_g)}{\partial u(t-\Delta t_g)} - 
    \lambda(t)\frac{\partial f(u(t-\Delta t_f),p,t-\Delta t_f)}{\partial u(t-\Delta t_f)} + \right. \\\\
	& \qquad \left. \vphantom{\frac{\partial}{\partial}}\ldots (\Delta t - \Delta t_{g_2}) + \ldots (\Delta t - \Delta t_{f_2}) \right)\\\\
    & = -\frac{\partial g(u(t),p,t)}{\partial u(t)} -
    \lambda(t)\frac{\partial f(u(t),p,t)}{\partial u(t)}
\end{align*}$$
This differential equation is the same as the one found in [part 1]({{< ref "adjoint sensitivity" >}}).
It is a good exercise to try the same technique on \\(\frac{\partial G(c_{[0,t]},u(t),p)}{\partial p}\\).