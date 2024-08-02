---
draft: false
title: Dynamic experimental design in 100 lines of Julia code
date: 2021-09-30
lastmod: 2022-01-31
---
{{< katex >}}
Optimal experimental design is an area of statistics focused on constructing informative experiments.
In this tutorial we introduce the necessary tools to construct such informative experiments for dynamic systems using only 100 lines of Julia code.
We will work with a well-mixed fed-batch bioreactor as an example system. We have quite a bit of domain knowledge how to model the behavior of such a reactor.
The reactor has three dynamic states: the substrate concentration \\(C_s\\), the biomass concentration \\(C_x\\) and the volume of the reactor \\(V\\).
The evolution in time of these states is governed by the following differential equations:
<!--more-->

$$\begin{align*}
\frac{dC_s}{dt} &= -\sigma C_x + \frac{Q_{in}(t)}{V}(C_{S,in} - C_s)\\\\
\frac{dC_x}{dt} &= \mu C_x - \frac{Q_{in}(t)}{V}C_x\\\\
\frac{dV}{dt} &= Q_{in}(t),
\end{align*}$$
where, 

$$\begin{align*}
\mu &= \frac{\mu_{max}C_s}{K_s + C_s}\\\\
\sigma &= \frac{\mu}{y_{x,s}} + m.
\end{align*}$$

Not all parameters in these equations are exactly known.
We are unsure of the value of the maximal growth rate \\(\mu_{max}\\), and the half saturation constant \\(K_s\\).
We want to construct an experiment to learn about these parameters.
To reach this goal we can control the volume of the reactor by the volumetric flow rate \\(Q_{in}(t)\\).
And we will measure the two states \\(C_s\\) and \\(C_x\\).
The other parameters in the equation,
namely the substrate feed concentration \\(C_{S,in}\\), the maintenance factor \\(m\\) and yield \\(y_{x,s}\\), are considered to be exactly known.
All of the numerical values we will use are taken from
[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0098135414002695).


The packages that we will use:
```julia
using OrdinaryDiffEq
using DiffEqFlux
using ForwardDiff
using Distributions
using Quadrature
using Optim
using LinearAlgebra
using Plots
using Random
```



# Defining the system
First, we define the dynamics of the fed-batch reactor.
```julia
function dynamics!(du,u,p,t)
    C_s, C_x, V = u   
    μ_max, K_s = @view p[1:n_θ]
    control_par = @view p[n_θ+1:end]
    Q_in = control_network(t, control_par)[1]
    C_s_in, y_x_s, m = 50.0, 0.777, 0.0
    μ = μ_max*C_s/(K_s + C_s)
    σ = μ/y_x_s + m

    du[1] = -σ*C_x + Q_in/V*(C_s_in - C_s)
    du[2] = μ*C_x - Q_in/V*C_x
    du[3] = Q_in
    return nothing
end
```



Finding the optimal controls is an infinite dimensional optimization problem.
To reduce the complexity of this problem to a non-linear optimization one, we use a parametrized control function.
We call this parametrization \\(x\\), and in this case we use a neural network with three hidden layers, each with five neurons.
Time \\(t\\) is the only input to this neural network and the flow rate \\(Q_{in}\\) is the only output.
Note that the last activation function constrains the control input between 0 and 10.
You can fine-tune the amount of hidden layers and neurons as well as the different activation functions to achieve even better experiments than presented in this tutorial.
```julia
Random.seed!(45415)
const control_network = FastChain(FastDense(1, 5,gelu),
                                  FastDense(5, 5,swish),
                                  FastDense(5, 5,relu),
                                  FastDense(5, 1,z->10.0 .*sigmoid(z)))
const x_ini = randn(length(initial_params(control_network)))
const n_x = length(x_ini)
```



Next, we define the true value of the two uncertain parameters \\(\theta\\). We will use the true values for a simulation study to test the performance of our experiment.
Both the uncertain parameters and the control parameters will have to be passed together as the \\(p\\) argument of *dynamics!*.
```julia
const θ_t = [0.421, 0.439]
const n_θ = length(θ_t )
const p  = vcat(θ_t,x_ini)
```



Our experiment will last 15 hours. The initial conditions  of the dynamics, \\(u_0\\), are fixed.
```julia
const tspan = (0.0, 15.0)
const u_0 = [3.0, 0.25, 7.0]
const n_u = length(u_0)
```



We continue by simulating the true system.
```julia
prob = ODEProblem(dynamics!,u_0,tspan,p) 
sol = solve(prob,Tsit5(),reltol=1e-5,abstol=1e-5)
plot(sol;label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"],xlabel="t(h)",lw=3)
plot!(tickfontsize=12,guidefontsize=14,legendfontsize=14,grid=false, dpi=600)
```

{{< figure src="../figdynexptut/dynamic experimental design_7_1.png"  >}}

# Some optimal experimental design theory
Generally, we do not measure all the dynamic states continuously, but instead we take measurements \\(y_k\\), which are some function of the states \\(u\\) at discrete time points,

$$\begin{align*}
\frac{du}{dt} &= f(u,\theta,x,t)\\\\
y_k(\theta,x) &= h(u(\theta,x,t_k)).
\end{align*}$$

In our case we measure the substrate and biomass concentration every hour.
These measurements should inform us about the true value of the uncertain parameters.
Often these measurements are noisy, with covariance \\(\Sigma\\).
```julia
const tmeasurement = 0.0:1.0:15.0
const Σ = [0.001 0.0; 0.0 0.000625]
```



The Fisher information matrix (FIM) \\(F\\) is a popular way to quantify this information content of an experiment,

$$F(\theta,x) = \sum_k  \frac{\partial y_k'}{\partial\theta}\Sigma^{-1} \frac{\partial y_k}{\partial\theta}.$$

The intuition behind this formula is that in a good experiment the measurements should be sensitive towards the value of the uncertain model parameters.
If the experiment is such that the measurements are similar no matter what the value of the true parameter is,
then it will be hard to precisely determine that parameter value.
These sensitivities of the measurements towards the uncertain parameters can be further expanded,

$$\frac{\partial y_k(\theta,x)}{\partial\theta} = 
\frac{\partial h}{\partial u} \frac{\partial u(\theta,x,t_k)}{\partial\theta}.$$

Since we only measure the first two states, the first factor, \\(\frac{\partial h}{\partial u}\\), is equal to the matrix [1 0; 0 1; 0 0].
The second factor can be calculated from the forward sensitivity transform of the differential equation system,

$$\begin{align*}
\frac{d}{dt}\frac{\partial u(\theta,x,t)}{\partial\theta}
&= \frac{\partial}{\partial\theta}\frac{du(\theta,x,t)}{dt}
= \frac{\partial f(u,\theta,x,t)}{\partial u}\frac{\partial u}{\partial \theta} + 
\frac{\partial f(u,\theta,x,t)}{\partial \theta}\\\\
\frac{\partial u(\theta,x,t=0)}{\partial \theta} &= 0.
\end{align*}$$

In Julia this can be done using forward mode automatic differentiation.
We want the Fisher information matrix to be as large as possible, but what constitutes a large matrix?
The inverse of the FIM is related to confidence ellipsoids and we want these ellipsoids to be as small as possible volume wise, which turns out to be equivalent to maximizing the determinant of the FIM.
In the experimental design literature, these are called D-optimal experiments. In Julia this can be done in the following way:
```julia
function solve_wrap(θ,x)
    p  = [θ;x]
    prob = ODEProblem(dynamics!,u_0,tspan,p)
    sol = solve(prob,Tsit5(),saveat=tmeasurement,reltol=1e-5,abstol=1e-5)
    u = convert(Array,sol)
end
function D_criterion(θ,x)
    FIM = zeros(eltype(x),n_θ,n_θ)
    jac = ForwardDiff.jacobian(θ->solve_wrap(θ,x),θ)
    for k in 1:length(tmeasurement)
        du_dθ = jac[(k-1)*n_u+1:k*n_u,:]
        dy_θ = du_dθ[1:2,:]
        FIM += dy_θ'*inv(Σ)*dy_θ
    end
    return det(FIM)
end
```



The major issue with experimental design based on the Fisher information matrix, is the dependence of the optimal design on the true model parameter values that the experiment aims to learn about.
We can robustify this by averaging the D-criterion over a distribution of possible parameter values.
We specify the uncertainty on the model parameters as probability distributions provided by Distributions.jl.
We use truncated normal distributions centered around \\(4.0\\) for both uncertain parameters. The expectation is calculated rather inaccurately to speed up the tutorial. 
```julia
function robust_D_criterion(x)
    θ_dist = product_distribution([truncated(Normal(0.4,0.1),0.1,0.7),truncated(Normal(0.4,0.1),0.1,0.7)])
    integrand(θ,x) = D_criterion(θ,x)*pdf(θ_dist,θ)
    prob_quadrature = QuadratureProblem(integrand,minimum(θ_dist),maximum(θ_dist),x)
    sol_quadrature = solve(prob_quadrature,HCubatureJL(),reltol = 1e-2, abstol = 1e-2)[1]
end
```



# Optimal experiment
Finally, we get to optimizing the controls. A first order optimization technique is used. Optim.jl is capable of calculating the required gradient using automatic differentiation.
```julia
x_opt = optimize(x-> -robust_D_criterion(x), x_ini, BFGS(), Optim.Options(iterations=10), autodiff = :forward).minimizer
t_plot = 0.0:0.1:15
plot(t_plot,[control_network(t, x_ini)[1] for t in t_plot ],label="initial experiment",lw=3)       
plot!(t_plot,[control_network(t, x_opt)[1] for t in t_plot],label="optimized experiment",lw=3)
plot!(xlabel="t(h)", ylabel="Q(l/h)")
plot!(tickfontsize=12,guidefontsize=14,legendfontsize=14,grid=false,ylim=(-1,11), dpi=600)
```

{{< figure src="../figdynexptut/dynamic experimental design_11_1.png"  >}}

Now all that remains is showing the added value of the optimal experiment.
We simulate many experiments according to both the design we started with,
and the optimized design, using the true parameter values.
We then see how precisely, we can recover the true parameters from the simulated data sets.
The optimized experiment's parameter estimates are generally closer to the true parameter values
than the initial experiment we started with.
```julia
Random.seed!(454154)
function SSE(θ,x)
    prob = ODEProblem(dynamics!,u_0,tspan,vcat(θ,x)) 
    sol = solve(prob,Tsit5(),saveat=tmeasurement,reltol=1e-5)
    y_est = sol[1:2,:]
    errors = y_measure - y_est
    errors_scaled = sqrt(inv(Σ))*errors
    sum(errors_scaled.^2)
end

y_true  = solve(ODEProblem(dynamics!,u_0,tspan,vcat(θ_t,x_ini)),Tsit5(),
                saveat=tmeasurement,reltol=1e-5,abstol=1e-5)[1:2,:]
θ_estimates = zeros(n_θ,100)
for j = 1:100
    global y_measure = y_true + rand(MvNormal(Σ),length(tmeasurement))
    θ_estimates[:,j] = optimize(θ->SSE(θ,x_ini),  θ_t).minimizer
end
Plots.scatter(θ_estimates[1,:],θ_estimates[2,:],label="unoptimized experiment",ms=5)

y_true  = solve(ODEProblem(dynamics!,u_0,tspan,vcat(θ_t,x_opt)),Tsit5(),
                saveat=tmeasurement,reltol=1e-5,abstol=0.0)[1:2,:]
for j = 1:100
    global y_measure = y_true + rand(MvNormal(Σ),length(tmeasurement))
    θ_estimates[:,j] = optimize(θ->SSE(θ,x_opt),  θ_t).minimizer
end
Plots.scatter!(θ_estimates[1,:],θ_estimates[2,:],label="optimized experiment",ms=5)
plot!(xlabel="μₘₐₓ",ylabel="Kₛ")
plot!(tickfontsize=12,guidefontsize=14,legendfontsize=14,grid=false, dpi=600)
```

{{< figure src="../figdynexptut/dynamic experimental design_12_1.png"  >}}

The precisely estimated parameters can subsequently be used to obtain a better control of the bio-reactor to,
for example, grow as much biomass as possible.
Currently, the design optimization is implemented using forward over forward automatic differentiation.
It would be more efficient to instead do this with reverse over forward automatic differentiation,
but this is not yet easy to do in Julia.
# References
1. [Telen, Dries, et al. "Robustifying optimal experiment design for nonlinear, dynamic (bio) chemical systems." Computers & chemical engineering 71 (2014): 415-425.](https://www.sciencedirect.com/science/article/abs/pii/S0098135414002695)