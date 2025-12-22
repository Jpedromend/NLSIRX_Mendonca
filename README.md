# Effect of Containment Efforts on Epidemic Processes: Nonlinear SIR-X

This repository contains code to simulate and visualize the **nonlinear SIR-X epidemic model** introduced in

> **“Effect of containment efforts on epidemic processes: emergent cooperative phenomena”**

The model extends the standard SIR framework by including:
- A new compartment $X$ modeling isolated infected population,
- Two nonlinear (cooperative) containment rates that depend on the current number of infected individuals.

The codes:
- Integrate the nonlinear SIR-X equations,
- Plot epidemic curves (e.g. active cases),
- Process real COVID-19 data (Johns Hopkins University dataset) for comparison with the model,
- Fits real data by estimating the best NL-SIRX parameters via log-scale MSE.

If you use this code in your research, please cite the associated manuscript:
> Mendonça, J.P., Lira, S.A. & Lyra, M.L. Critical scaling and nonlinear adaptative containment responses in epidemic dynamics modeling. Eur. Phys. J. Spec. Top. (2025). https://doi.org/10.1140/epjs/s11734-025-02110-z

---

## 1. Model Overview

We consider four populations:
- $S(t)$: susceptible  
- $I(t)$: infected (active cases)  
- $R(t)$: removed (recovered/dead or otherwise no longer infectious)  
- $X(t)$: isolated (quarantined / contained) individuals

Parameters:
- $\beta/N$ (`rr`): effective infection rate
- $\gamma$ (`a`): removal rate  
- $\kappa$ (`k`): rate of specific (quarantine) social distancing acting on infectious individuals  
- $\kappa_0$ (`k0`): rate of general (confinement) social distancing acting on the mixing population (susceptible and infectious)
- $n, m$: exponents controlling **cooperative response** of containment to the number of infecteds

The nonlinear SIR-X ODEs used in the code are:

$$
\begin{aligned}
\frac{dS}{dt} &= - \frac{\beta S I}{N} - \kappa_0 I^m S, \\
\frac{dI}{dt} &= \frac{\beta S I}{N} - \gamma I - \kappa_0 I^m I - \kappa I^n I, \\
\frac{dR}{dt} &= \gamma I + \kappa_0 I^m S, \\
\frac{dX}{dt} &= \kappa_0 I^m I + \kappa I^n I.
\end{aligned}
$$

The total population
$S + I + R + X$
is conserved.

---

## 2. Repository Contents

- `nlsirx.py`  
  Contains the function `sirx_nonlinear_ode_solver(...)` that numerically integrates the nonlinear SIR-X equations using SciPy’s `solve_ivp`.

- `plot_nlsirx.py`  
  Loads a `.dat` file produced by the solver and plots the **active cases** $I(t)$ on a semilogarithmic scale.

- `process_jhu_data.py`  
  - Load the JHU global time-series for confirmed, deaths, and recovered cases,
  - Aggregate by country,
  - Compute **Active = Confirmed − Deaths − Recovered**,
  - Save the cleaned data,
  - Plot active cases for checking.

- `fit_nlsirx.py`
  - Loads processed country data,
  - Fits both **Linear** and **Nonlinear** SIR-X models to the active cases using least-squares optimization,
  - Compares the models via Normalized RMSE (per 100k population),
  - Generates a comparative plot of the fits against real data.
  > **Note on Reproducibility:** The fitting results are sensitive to user-defined inputs such as the start date and total population. Additionally, while the nonlinear model typically yields smaller errors, the optimization relies on randomized initial parameter guesses; consequently, generating multiple samples is recommended to avoid local minima and identify the global best fit.
---

## 3. Requirements

This code was developed and tested using **Python 3.x**, and the following libraries:

- [NumPy](https://numpy.org/) (tested on v2.3.3)
- [Pandas](https://pandas.pydata.org/) (tested on v2.3.3)
- [Matplotlib](https://matplotlib.org/) (tested on v3.10.6)
- [SciPy](https://scipy.org/) (tested on v1.16.2)

> The initially produced fitting was performed using real data series from the JHU repository ([https://github.com/CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19)), downloaded in May 2020, but any first-wave real data can be accommodated with minimal tweaks to `process_jhu_data.py`; in principle, this framework can be used to simulate general outbreaks where containment efforts are implemented.

