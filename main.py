"""
Author: Joe Samyn
Class: CST-305
Professor: Dr. Citro
Creation Date: 1.29.21
Last Revision Date: 1.31.21
Purpose: The purpose of this program is to solve the ordinary differential equation: y' = y/(e^x) - 1. The Equation
is solved using the Runge-Kutta algorithm and the ODEint package from SciPy. The results are compared and plotted using
the MatPlotLib library.
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.integrate import odeint as ode


def find_k_sum(y, x, dx, kn, prev_kval):
    """
    Finds the sum of all the K values in the Runge-Kutta 4th order algorithm

    Parameters
    ----------
    y: float
        Yn value
    x: float
        Xn value
    dx: float
        Step size
    kn: int
        K being solved (K1, K2, K3, or K4)
    prev_kval: float
        Value of the previous K (Kn-1)
    """
    # If calculating K4
    if kn >= 4:
        # Calculate K4
        k = (y + (dx*prev_kval))/(np.exp(x + dx) - 1)
        return k
    # If calculating K2 or K3
    elif kn == 2 | kn == 3:
        # Find dx midpoint
        dx_mid = dx/2
        # Calculate X at midpoint
        x_temp = x + dx_mid
        # Calculate Y at midpoint
        y_temp = y + (dx_mid * prev_kval)
        # Calculate K value with calculated X and Y values above
        k = (y_temp/(np.exp(x_temp) - 1))*2
        # Return K plus the next K value using recursion
        return k + find_k_sum(y, x, dx, kn + 1, k)
    # If calculating K1 just calculate using initial ODE
    else:
        k = y/(np.exp(x) - 1)
        # Return K plus sum of all other calculated K values
        return k + find_k_sum(y, x, dx, kn + 1, k)


def solve_rk(y, x, dx, n):
    """
    Solves the differential equation using the Runge-Kutta method
    ODE: y' = y/(e^x - 1)

    Parameters
    ----------
    y: int
        initial condition for y
    x: int
        initial condition for x
    dx: float
        step size
    n: int
        The number of x and y values to calculate using Runge-Kutta algorithm

    Returns
    ----------
    float[][]
        2D float containing all x & y values calculated
    """
    # Initialize the calculated XY list with Y0 and X0
    calculated_xy = [[x, y]]
    # Set curr_y to Y0
    curr_y = y
    # Set curr_x to X0
    curr_x = x
    # Loop through and solve RK for Xn and Yn
    for i in range(1, n):
        # Calculate Yn
        curr_y = curr_y + (dx / 6) * find_k_sum(curr_y, curr_x, dx, 1, 0)
        # Calculate Xn
        curr_x = curr_x + dx
        # Append Xn and Yn to list
        calculated_xy.append([curr_x, curr_y])
    # return results of RK algorithm in list
    return calculated_xy


def calculate_error(rk_calcs, odeint_calcs):
    """
    Calculates the error between the Runge-Kutta method of solving differential equations and
    the odeint method for solving differential equations.

    Parameters
    ----------
    rk_calcs: float[][]
        Calculations generated from the RK method
    odeint_calcs: float[][]
        Calculations generated from the ODEint method

    Returns
    ----------
    double[]
        Array containing error decimal for each Xn and Yn
    """
    # Initialize error list
    errors = []
    # Loop through rk_calc and odeint_calc and calculate the error between the two
    for i in range(1000):
        # Calculate error using percent error formula (experimental - theoretical)/theoretical
        err = np.abs((rk_calcs[i][1] - odeint_calcs[i][0])/odeint_calcs[i][0])
        # Add error to end of list
        errors.append(err)
    # Return error list when complete
    return errors


def model(y, x):
    """
    The differential equation model being solved by ODEint

    Parameters
    ----------
    y: float
        the starting y value from the inital condition (in this example Y0 = 5)
    x: float
        The x values to be used to find y. Each X value has step size of 0.02

    Returns
    ----------
    float
        dydx value for the differential equation at X and Y
    """
    dydx = y/(np.exp(x) - 1)
    return dydx


def calculate_stop_val_x(dx):
    """
    Calculates the ending value for X given that 1000 values are needed with a step size of 0.02.
    Should calculate that ending value is 21.

    Parameters
    ----------
    dx: float
        The delta X value or step size for the differential equation

    Returns
    ----------
    float
        The ending value of the range for a starting point of 1 using step size 1000 times.
    """
    return 1 + (dx * 1000)


def display_error(err, x, rk_calcs, odeint_calcs):
    """
    Plots the error for each step in the differential equation solving process using a line graph.
    Shows a table of the error for each step in the calculation.
    Shows the average error for the entire calculation.
        Y-Axis: error decimal
        X-Axis: Xn value that corresponds to the error

    Parameters
    ----------
    err: float[]
        The error array generated when calculating the error.
    x: float[]
        The Xn values
    Returns
    ----------
    None
    """
    # Plot the error
    plt.plot(x, err)
    # Set X label
    plt.xlabel('Xn Values')
    # Set Y label
    plt.ylabel('Error In Decimal')
    # Set title
    plt.title('Error Between Runge-Kutta and Odeint Calculations')
    # Show Plot
    plt.show()

    # Convert arrays to numpy arrays
    np_rk = np.array(rk_calcs)
    np_ode = np.array(odeint_calcs)
    np_err = np.array(err)
    # Create table from values using pandas
    error_table = pd.DataFrame({'X': x})
    error_table['Y_RK'] = np_rk[:, 1]
    error_table['Y_ODEint'] = np_ode[:, 0]
    error_table['Error_Decimal'] = np_err
    error_table['Error_Percentage'] = np_err*100
    # Print first few items in table
    print(error_table.head())
    # Convert the table to HTML file for easy viewing
    error_table.to_html('error_results.html')

    # Calculate the avg error for all calculations
    # Calculate sum of all error values
    err_sum = np_err.sum()
    # Divide sum by N
    avg_err = err_sum/1000
    # Display error
    print('The average error between the Runge-Kutta algorithm and ODEint is: ' + str(avg_err*100) + '%')

def plot_rk(rk_calc):
    """
    Plots the X and Y values for the Runge-Kutta solution on a line graph

    Parameters
    ----------
    rk_calc: double[][]
        2D array containing all the [Xn, Yn] values for the solution

    Returns
    ----------
    None
    """
    # Convert rk_calc to numpy array for plotting
    np_rk_values = np.array(rk_calc)
    # Plot the values
    plt.plot(np_rk_values[:, 0], np_rk_values[:, 1])
    # Set Y limit to 8 to match ODE graph
    plt.ylim([5.0, 8.0])
    # Set X label
    plt.xlabel('Xn Values (1 - 21)')
    # Set Y label
    plt.ylabel('Calculated Y Values For X')
    # Set title
    plt.title('Runge-Kutta Results for X0, Y0 to X1000, Y1000')
    # Show Plot
    plt.show()


def plot_odeint(ode_calc, x):
    """
    Plots the X and Y values from the ODEint solution on a line graph

    Parameters
    ----------
    ode_calc: double[][]
        Yn values calculated for the corresponding X
    x: float[]
        Xn values

    Returns
    ----------
    None
    """
    # Convert ode_calc to numpy array for plotting
    np_ode_calc = np.array(ode_calc[:,0])
    plt.plot(x, np_ode_calc)
    # Set X label
    plt.xlabel('Xn Values (1 - 21)')
    # Set Y label
    plt.ylabel('Calculated Y Values For X')
    # Set title
    plt.title('ODEint Results for X0, Y0 to X1000, Y1000')
    # Show Plot
    plt.show()


def plot_rk_odeint_overlapping(rk_calc, ode_calc, x):
    """
    Plots the Runge-Kutta results and the ODEint results on the same graph to demonstrate differences in the algorithms
    results

    Parameters
    ----------
    rk_calc: double[][]
        results from the Runge-Kutta calculations
    ode_calc: double[][]
        Yn results from the ODEint calculations
    x: float[]
        Xn values for the ODEint calculations

    Returns
    ----------
    None
    """
    # Convert ode_calc to numpy array for plotting
    np_ode_calc = np.array(ode_calc[:, 0])
    # Plot odeint values
    odeint_plt, = plt.plot(x, np_ode_calc, label='ODEint')
    # Convert rk_calc to numpy array for plotting
    np_rk_values = np.array(rk_calc)
    # Plot the values
    rk_plt, = plt.plot(np_rk_values[:, 0], np_rk_values[:, 1], label='Runge-Kutta')
    # Set X label
    plt.xlabel('Xn Values')
    # Set Y Label
    plt.ylabel('Yn Values')
    # Set Title
    plt.title('Runge-Kutta vs ODEint')
    # Show legend for plots
    plt.legend(handles=[rk_plt, odeint_plt])
    # Add some transparency to make graph more clear
    # Show Plots
    plt.show()


# Run the main program
if __name__ == '__main__':
    # initialize starting variables for RK algorithm
    n = 1000
    dx = 0.02
    y_init = 5
    x_init = 1

    # Grab start time of calculations
    start_time = dt.datetime.now()

    # Run Runge-Kutta calculation
    rk_calculations = solve_rk(y_init, x_init, dx, n)

    # Initialize starting variables for odeint calculations
    x_stop_val = calculate_stop_val_x(dx)
    # set x range
    x = np.arange(1, x_stop_val, dx)

    # Run ODEint calculation
    ode_calculations = ode(model, y_init, x)

    # Grab end time of calculations
    end_time = dt.datetime.now()
    # Calculate runtime of calculations for solving ODE with RK and ODEint
    print("Program Runtime: " + str((end_time - start_time).microseconds/1000))

    # Plot Results
    # Plot Runge-Kutta
    plot_rk(rk_calculations)
    # Plot ODEint
    plot_odeint(ode_calculations, x)
    # Plot both together on same graph
    plot_rk_odeint_overlapping(rk_calculations, ode_calculations, x)

    # calculate error between RK algorithm and odeint results
    err_calcs = calculate_error(rk_calculations, ode_calculations)
    # Display all error calculations
    display_error(err_calcs, x, rk_calculations, ode_calculations)

