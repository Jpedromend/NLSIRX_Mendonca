import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
from nlsirx import sirx_nonlinear_ode_solver

def prepare_data_for_fitting(df, start_date=None, min_active_cases=100):
    """
    Filters and prepares the dataset for fitting based on date OR case count threshold
    """
    if not np.issubdtype(df['Date'].dtype, np.datetime64):
        df['Date'] = pd.to_datetime(df['Date'])

    if start_date:
        df_fit = df[df['Date'] >= pd.to_datetime(start_date)].copy()
    else:
        df_fit = df[df['Active'] >= min_active_cases].copy()
        
    if df_fit.empty:
        raise ValueError("No data points found matching criteria.")

    df_fit = df_fit.sort_values('Date').reset_index(drop=True)
    df_fit['Time'] = np.arange(len(df_fit)) 
    return df_fit

def get_residuals(params, t_data, log_y_data, pop_total, mode='nonlinear'):
    """
    Calculates residuals between the model simulation and observed data (log scale).
    
    Args:
        params: array-like, parameters to optimize.
                Linear: [beta, gamma, kappa, kappa0]
                Nonlinear: [beta, gamma, kappa, kappa0, n, m]
    """
    # Unpack parameters based on mode
    if mode == 'linear':
        beta, a, k, k0 = params
        n, m = 0.0, 0.0 
    else:
        beta, a, k, k0, n, m = params

    # Scale beta relative to population size
    rr = beta / pop_total

    # Define initial conditions based on data points
    I0 = np.exp(log_y_data[0])
    S0 = pop_total - I0
    
    # Run Simulation
    t_max_sim = t_data[-1] + 2.0
    
    try:
        df_sim = sirx_nonlinear_ode_solver(
            t_max=t_max_sim, rr=rr, S0=S0, I0=I0, R0=0, X0=0,
            a=a, k=k, k0=k0, n=n, m=m, h=0.1 
        )
        
        # Interpolate simulation results to match observed data timepoints
        model_I = np.interp(t_data, df_sim['t'], df_sim['I'])
        
        # Calculate log-residuals (with safety floor for log calculation)
        model_I = np.maximum(model_I, 1e-9)
        log_model = np.log(model_I)
        
        return log_model - log_y_data

    except Exception:
        # Return high penalty if ODE solver fails
        return np.ones(len(t_data)) * 100.0

def run_optimization(t_obs, y_obs_log, total_pop, mode='nonlinear'):
    """
    Sets bounds and initial guesses, then runs the least_squares optimization.
    """
    # Base parameters: beta, gamma, k, k0
    p0 = np.random.rand(4) * [1.0, 0.1, 0.1, 0.01]
    bounds_lower = [0.0, 0.0, 0.0, 0.0]
    bounds_upper = [1.0, 0.1, 0.1, 0.1]

    if mode == 'nonlinear':
        # Add n, m parameters
        p0 = np.concatenate(( p0, np.random.rand(2) * [0.2, 0.2] ))
        bounds_lower = np.concatenate((bounds_lower, [0.0, 0.0]))
        bounds_upper = np.concatenate((bounds_upper, [0.6, 0.6]))
    
    res = least_squares(
        get_residuals, p0, 
        bounds=(bounds_lower, bounds_upper), 
        args=(t_obs, y_obs_log, total_pop, mode),
        loss='soft_l1', f_scale=0.1
    )
    
    return res.x

def simulate_best_fit(p, t_max, I_start_data, total_pop, mode):
    """
    Runs the forward simulation using the optimized parameters.
    """
    if mode == 'linear':
        beta, a, k, k0 = p
        n, m = 0.0, 0.0
    else:
        beta, a, k, k0, n, m = p
        
    I0 = I_start_data
    rr = beta / total_pop
    
    return sirx_nonlinear_ode_solver(
        t_max=t_max, rr=rr, S0=total_pop-I0, I0=I0, R0=0, X0=0,
        a=a, k=k, k0=k0, n=n, m=m
    )

def calculate_errors(df_sim, t_obs, y_obs, pop_total):
    """
    Calculates RMSE and Normalized RMSE (per 100k population).
    """
    model_I = np.interp(t_obs, df_sim['t'], df_sim['I'])
    
    diff = model_I - y_obs
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    
    nrmse_per_100k = (rmse / pop_total) * 100000
    
    return rmse, nrmse_per_100k

def plot_results(df_data, sol_lin, sol_nl, err_lin, err_nl, country_name):
    """
    Plots the Real Data vs Linear and Nonlinear SIR-X fits.
    """
    # Prepare dates for simulation data
    start_date = df_data['Date'].iloc[0]
    dates_lin = [start_date + pd.Timedelta(days=t) for t in sol_lin['t']]
    dates_nl  = [start_date + pd.Timedelta(days=t) for t in sol_nl['t']]

    plt.figure(figsize=(10, 6), dpi=100)
    
    # Plot Real Data
    plt.plot(df_data['Date'], df_data['Active'], 'ko', fillstyle='none', markersize=5, label='Real Data')
    
    # Plot Linear Fit
    plt.plot(dates_lin, sol_lin['I'], 'r--', linewidth=2, label=f'Linear SIR-X (NRMSE={err_lin:.2f})')
    
    # Plot Nonlinear Fit
    plt.plot(dates_nl, sol_nl['I'], 'b-', linewidth=3, alpha=0.8, label=f'Nonlinear SIR-X (NRMSE={err_nl:.2f})')
    
    # Formatting
    plt.title(f'Model Fit: {country_name}', fontsize=16)
    plt.ylabel('Active Cases (Log Scale)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # X-Axis Date Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
print("--- COVID-19 SIR-X Model Fitting ---")

# 1. User Inputs
country_input = input("Enter Country Name (Default: Korea, South): ").strip()
COUNTRY = country_input if country_input else 'Korea, South'

pop_input = input("Enter Total Population (Default: 50,000,000): ").strip()
TOTAL_POP = int(pop_input) if pop_input else 50e6 

date_input = input("Enter Start Date YYYY-MM-DD (Default: None): ").strip()
START_DATE = date_input if date_input else None

# 2. Load & Prep Data
filename = f"data_{COUNTRY.lower()}.csv"
try:
    df_raw = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    exit()

print(f"\nProcessing data for {COUNTRY}...")
df_fit = prepare_data_for_fitting(df_raw, start_date=START_DATE)

t_obs = df_fit['Time'].values
y_obs = df_fit['Active'].values
y_obs_log = np.log(df_fit['Active'].values)

# 3. Run Optimizations
params_lin = run_optimization(t_obs, y_obs_log, TOTAL_POP, mode='linear')
params_nl  = run_optimization(t_obs, y_obs_log, TOTAL_POP, mode='nonlinear')

# 4. Simulate Best Fits for Plotting
t_max_plot = t_obs.max()
I_start_val = df_fit['Active'].iloc[0]

sol_lin = simulate_best_fit(params_lin, t_max_plot, I_start_val, TOTAL_POP, 'linear')
sol_nl  = simulate_best_fit(params_nl,  t_max_plot, I_start_val, TOTAL_POP, 'nonlinear')

# 5. Compute Errors (NRMSE per 100k)
_, mse_lin_nrm = calculate_errors(sol_lin, t_obs, y_obs, TOTAL_POP)
_, mse_nl_nrm  = calculate_errors(sol_nl,  t_obs, y_obs, TOTAL_POP)

# 6. Output Results
print("\nRESULTS (Normalized RMSE per 100k):")
print(f"  Linear Model:    {mse_lin_nrm:.5f}")
print(f"  Nonlinear Model: {mse_nl_nrm:.5f}")

print("\nOptimized Parameters:")
print(f"  Linear:    {np.round(params_lin, 4)}")
print(f"  Nonlinear: {np.round(params_nl, 4)}")

# 7. Visualization
plot_results(df_fit, sol_lin, sol_nl, mse_lin_nrm, mse_nl_nrm, COUNTRY)
