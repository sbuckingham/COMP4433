# U.S. Energy Production and Consumption Analysis

## Project Overview
This project visualizes trends in U.S. energy production and consumption by state and energy type. The analysis explores differences in energy production, consumption, and expenditures across states, and provides a Dash application for interactive data exploration.

## Features
- **Data Cleaning and Preparation**: Merging and handling multiple datasets, removing null values.
- **Exploratory Data Analysis (EDA)**: Statistical summaries and visualizations of key energy metrics.
- **Dash Application**: Interactive components to visualize energy trends dynamically by state and energy type.

## Files Included
- COMP 4433 Project 2 Sarah Buckingham.ipynb: Contains all the code, data processing, exploratory data analysis, and Dash application code.
- states.csv: Contains state abbreviations and full names.
- SelectedStateRankingsData.csv: Data for state rankings in energy consumption, production, and expenditures.
- organised_Gen.csv: Contains monthly energy generation data for U.S. states from 2001 to 2022.

## Installation

### Requirements
The code requires Python 3.8 or above and the following libraries:
- Dash
- Plotly
- Pandas
- Seaborn
- Matplotlib

To install the dependencies, use the following command:
   ```bash
   pip install -r requirements.txt
   ```
Alternatively, you can use: 
  ```bash
   pip install dash plotly pandas seaborn matplotlib
   ```

## How to Run
Since the code is contained within a Jupyter Notebook:

1. Open "COMP 4433 Project 2 Sarah Buckingham.ipynb" in Jupyter Notebook or JupyterLab.
2. Run each cell sequentially to process the data and generate visualizations.
3. To launch the Dash app:
    - Run the final cell in the notebook containing app.run_server() (uncommented if necessary) to start the app.
    - Once running, visit http://127.0.0.1:8051 in your browser to access the interactive dashboard.
    
### Project Description
The project analyzes energy data trends by state over time, allowing users to:
- Compare state rankings in energy production, consumption, and expenditures.
- Explore energy generation by source across different states and years.
- Visualize data interactively using Dash.