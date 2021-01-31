# Project 2: Runge-Kutta for ODE

The program solves the ode: y' = y/(e^x) - 1
In order to run the program, you first must install the following packages using the pip installer tool on command line. The commands for installing the packages can be found below. 
## Steps For Installing Libraries
1. Navigate to the folder in the command line where the python script resides.
2. Enter The following commands into the terminal or command prompt to ensure you are in the proper directory:
	Windows: ``` dir ``` - ensure the python file is showing in the command prompt 
	Unix: ``` ls ``` - ensure the python file is showing in the terminal
3. Once you are in the proper directory, run the commands below to install the proper python packages.
### Installing Python Libraries
1. Install Pandas: ``` pip install pandas ```
2. Install MatPlotLib: ``` pip install matplotlib ```
3. Install SciPy: ``` pip install scipy ```

## Running the Code
To run the code enter one of the following commands below depending on your operating system.
### Windows
``` python main.py ```

### Unix (Mac & Linux)
``` python main.py ```

## Navigating to Next Plot
Once the program is running, the first plot will be displayed. To navigate to the next plot, close the plot that is 
being displayed. In order to present the error table in a cleaner fashion, an HTML file is generated in the directory 
in which the main.py file exists. Opening this file in a web browser will show the error calculations for every Yn. The
HTML file is name 'error_results.html'. 