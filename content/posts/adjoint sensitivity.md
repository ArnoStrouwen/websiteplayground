---
draft: false
title: Notes on adjoint sensitivity analysis of dynamic systems part 1
date: 2022-01-31
lastmod: 2022-04-30
---
{{< katex >}}
Gradients are useful for efficient parameter estimation and optimal control of dynamic systems.
Calculating these gradients requires sensitivity analysis.
Sensitivity analysis for dynamic systems comes in two flavors, forward mode and adjoint (reverse).
For systems with a large number of parameters adjoint sensitivity analysis is often more efficient
[[1]](https://ieeexplore.ieee.org/abstract/document/9622796).
I find that the traditional way of deriving adjoints for ordinary differential equations, such as
[[3]](https://book.sciml.ai/notes/11-Differentiable_Programming_and_Neural_Differential_Equations/),
leaves me with little intuition what these equations represent.
The goal of this blog post is to gain some intuition about these equations by deriving the adjoint equations in a different way.
<!--more-->
A prerequisite for understanding this post is being comfortable with the concept of backpropagation used in machine learning.
If you are not familiar with this, I recommend you first read
[[2]](https://book.sciml.ai/notes/10-Basic_Parameter_Estimation-Reverse-Mode_AD-and_Inverse_Problems/),
until you are comfortable with the backpropagation example of logistic regression.

To better understand adjoint sensitivity analysis for continuous time systems, we first start from the simpler case of discrete-time dynamic systems.
We want to the backpropagate the influence the parameters \\(p\\) have on the total cost \\(C\\) in the following computation,

$$\begin{align*}
    u_0 &= f_0(p)\\\\
    u_1 &= f_1(u_0,p)\\\\
    c_1 &= g_1(u_1,p)\\\\
    u_2 &= f_2(u_1,p)\\\\ 
    c_2 &= g_2(u_2,p)\\\\
    C   &= G(c_1, c_2) = c_1+c_2.
\end{align*}$$

![](../figadjoint/dynsys_transparant.png)
In these equations \\(u_0\\) is the initial state of the dynamic system,
which is a function of the parameters.
The transition from the initial state to the next state \\(u_1\\) is described by the function \\(f_1\\),
which is a function of the initial state and the parameters. Similarly, there is another transition \\(f_2\\) to \\(u_2\\).
For simplicity, we only consider a dynamic system with two steps.
For each of the two states \\(u_1\\) and \\(u_2\\) there is an associated cost function \\(c_1\\) and \\(c_2\\),
these costs can also be function of the parameters. The total cost \\(C\\) is the sum of the two individual costs.
Backpropagation will eventually lead to the gradient of \\(G\\) towards \\(p\\).

---
**NOTE**

In the following equations we will write the function \\(G\\) with different inputs.
\\(G\\) is the only function for which this is done, all other functions, such as \\(f_1\\),
will only be considered as functions of the inputs they are written as in the above system.

---

The gradient of \\(G\\) towards \\(c_1\\) and \\(c_2\\) is not difficult to calculate,

$$\nabla G(c_1, c_2) = \nabla(c_1+c_2) =  [1,1]^T.$$

Now let us take a step back and substitute \\(c_2\\) with \\(u_2\\) and \\(p\\),

$$
    \nabla G(c_1, u_2, p) =
    \nabla(c_1+g_2(u_2,p)) =
    \left[1,
    \frac{\partial g_2}{\partial u_2},
    \frac{\partial g_2}{\partial p}
    \right]^T.
$$

Call the second to last element of this vector,

$$\lambda_2  = \frac{\partial g_2}{\partial u_2},$$

and call the last element of this vector,

$$\phi_2  = \frac{\partial g_2}{\partial p}.$$

Now we pull back some more and substitute \\(c_1\\) with \\(u_1\\) and \\(p\\), and also substitute \\(u_2\\) with \\(u_1\\) and \\(p\\),

$$\begin{align*}
    \nabla G(u_1, p) &=
    \nabla(g_1(u_1,p)+g_2(f_2(u_1,p),p)) \\\\ 
    &= 
    \left[
    \frac{\partial g_1}{\partial u_1} + \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial u_1},
    \frac{\partial g_1}{\partial p} + \frac{\partial g_2}{\partial p} + \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial p}
    \right]^T.
\end{align*}$$

Now note that the second to last element of this vector can be written as,

$$\lambda_1  = \frac{\partial g_1}{\partial u_1} + \lambda_2 \frac{\partial f_2}{\partial u_1},$$

and the last element as,

$$\phi_1  = \lambda_2 \frac{\partial f_2}{\partial p} +  \frac{\partial g_1}{\partial p} + \phi_2.$$

Substituting \\(u_1\\) with \\(u_0\\) and \\(p\\) works the same,

$$\begin{align*}
    \nabla G(u_0, p) &=
    \nabla(g_1(f_1(u_0,p),p)+g_2(f_2(f_1(u_0,p),p),p) \\\\
    &=\left[
    \frac{\partial g_1}{\partial u_1}\frac{\partial f_1}{\partial u_0}+
    \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial u_1}\frac{\partial f_1}{\partial u_0},
    \frac{\partial g_1}{\partial p}
    + \frac{\partial g_1}{\partial u_1}\frac{\partial f_1}{\partial p}
    + \frac{\partial g_2}{\partial p}
    + \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial p}
    + \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial u_1}\frac{\partial f_1}{\partial p}
    \right]^T, 
\end{align*}$$

$$\lambda_0  = \lambda_1 \frac{\partial f_1}{\partial u_0},$$

$$\phi_0  = \lambda_1 \frac{\partial f_1}{\partial p} + \phi_1.$$

Substituing \\(u_0\\) with \\(p\\) gives the desired gradient,

$$\begin{align*}
    \nabla G(p) &=
    \nabla(g_1(f_1(f_0(p),p),p)+g_2(f_2(f_1(f_0(p),p),p),p)\\\\
    &=\left[
    \frac{\partial g_1}{\partial p}
    + \frac{\partial g_1}{\partial u_1}\frac{\partial f_1}{\partial p}
    + \frac{\partial g_1}{\partial u_1}\frac{\partial f_1}{\partial u_0}\frac{\partial f_0}{\partial u_p}
    + \frac{\partial g_2}{\partial p}
    + \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial p}
    + \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial u_1}\frac{\partial f_1}{\partial p}
    + \frac{\partial g_2}{\partial u_2}\frac{\partial f_2}{\partial u_1}\frac{\partial f_1}{\partial u_0}\frac{\partial f_0}{\partial u_p}
    \right]^T\\\\
    &=\left[
    \lambda_0 \frac{\partial f_0}{\partial p} + \phi_0 
    \right]^T.
\end{align*}$$

You should check that, for a high dimensional \\(p\\), this recursion based on \\(\lambda\\) and \\(\phi\\),
involves less costly matrix multiplications than the forward mode sensitivity analysis:

$$\begin{align*}
    \frac{\partial u_0}{\partial p} &= \frac{\partial f_0}{\partial p}\\\\
    \frac{\partial u_1}{\partial p} &= \frac{\partial f_1}{\partial u_0}\frac{\partial u_0}{\partial p} + \frac{\partial f_1}{\partial p}\\\\
    \frac{\partial c_1}{\partial p} &= \frac{\partial g_1}{\partial u_1}\frac{\partial u_1}{\partial p} + \frac{\partial g_1}{\partial p}\\\\
    \frac{\partial u_2}{\partial p} &= \frac{\partial f_2}{\partial u_1}\frac{\partial u_1}{\partial p} + \frac{\partial f_2}{\partial p}\\\\
    \frac{\partial c_2}{\partial p} &= \frac{\partial g_2}{\partial u_2}\frac{\partial u_2}{\partial p} + \frac{\partial g_2}{\partial p}\\\\
    \frac{\partial G}{\partial p} &= \frac{\partial c_1}{\partial p} + \frac{\partial c_2}{\partial p}.
\end{align*}$$

Now we know how to calculate the pullback of a difference equation.
Can we extend this intuition to continuous time systems?
If we assume that \\(f_k\\) comes from a forward euler discretization of a function \\(f_c\\),

$$f_{k+1}(u_k,p) = u_k + \Delta t f^c(u_k,p),$$

and we assume that \\(g_k\\) is the accumulation of a continuous cost function \\(g^c\\),
which can be considered constant in a short time-window \\(\Delta t\\),

$$g_k(u_k,p) = \Delta t g^c(u_k,p),$$

then we can calculate the recursion for \\(\lambda\\) as,

$$
	\lambda_{k}  = \frac{\partial g_k}{\partial u_k} + \lambda_{k+1} \frac{\partial f_{k+1}}{\partial u_k}
	= \Delta t\frac{\partial g^c}{\partial u_k} + \lambda_{k+1} \left(1+\Delta t\frac{\partial f^c}{\partial u_k}\right),
$$

which is the backwards euler solve of the differential equation,

$$\begin{align*}
	\frac{\lambda_{k+1} - \lambda_{k}}{\Delta t}  &= -\frac{\partial g^c}{\partial u_k}- \lambda_{k+1} \frac{\partial f^c}{\partial u_k},\\\\
	\frac{d\lambda}{dt}  &= -\frac{\partial g^c}{\partial u} - \lambda \frac{\partial f^c}{\partial u}.
\end{align*}$$

Similarly we can calculate the recursion for \\(\phi\\) as:

$$\begin{align*}
	\phi_k  &= \lambda_{k+1} \frac{\partial f_{k+1}}{\partial p} + \frac{\partial g_k}{\partial p} + \phi_{k+1},\\\\
	\frac{\phi_{k+1}-\phi_k }{\Delta t} &= -\lambda_{k+1} \frac{\partial f^c}{\partial p} - \frac{\partial g^c}{\partial p},\\\\
	\frac{d\phi}{dt}  &= -\lambda \frac{\partial f^c}{\partial p} - \frac{\partial g^c}{\partial p}.
\end{align*}$$

These are the same equations you find in the documentation of DifferentialEquations.jl [[4]](https://diffeq.sciml.ai/latest/extras/sensitivity_math/).
This, however, only gives us an intuition behind the equations for continuous time systems.
In [part 2]({{< ref "adjoint sensitivity2" >}}) we will see if a more rigorous version of this argument can be made for continuous time systems.
# References
1. [Y. Ma, V. Dixit, M. J. Innes, X. Guo and C. Rackauckas, "A Comparison of Automatic Differentiation and Continuous Sensitivity Analysis for Derivatives of Differential Equation Solutions," 2021 IEEE High Performance Extreme Computing Conference (HPEC), 2021, pp. 1-9. ](https://ieeexplore.ieee.org/abstract/document/9622796)
2. [Lecture 10 of the Parallel Computing and Scientific Machine Learning course](https://book.sciml.ai/notes/10-Basic_Parameter_Estimation-Reverse-Mode_AD-and_Inverse_Problems/)
3. [Lecture 11 of the Parallel Computing and Scientific Machine Learning course](https://book.sciml.ai/notes/11-Differentiable_Programming_and_Neural_Differential_Equations/)
4. [SciML sensitivity analysis documentation](https://docs.sciml.ai/SciMLSensitivity/stable/sensitivity_math/)
