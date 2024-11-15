#!/usr/bin/env python
# coding: utf-8

# ## COMP 4433: Project 2
# ### Sarah Buckingham

# In[1]:


import dash
#from dash import dcc, html, Input, Output
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ## Introduction 
# Data source: https://www.kaggle.com/datasets/kevinmorgado/us-energy-generation-2001-2022?resource=download  
# 
# This link was used to download the states.csv file and organised_Gen.csv file.
# 
# Data source: https://www.eia.gov/  
# 
# This link was used to download the SelectedStateRankingsData.csv file.
#         
# Using data from the U.S. Energy Information Administration, we can look at nationwide energy trends in terms of both production and consumption. We can perform an analysis and create a dash app the helps create an interactive investigation.

# ### Data Preparation
# First we can load in the datasets, merge to get state data in our energy generation as well as ranking/production datasets, and perform some preprocessing tasks. 

# In[2]:


# Load datasets
states_df = pd.read_csv('states.csv')
rankings_df = pd.read_csv('SelectedStateRankingsData.csv')
generation_df = pd.read_csv('organised_Gen.csv')

# Display the first few rows of each dataset to understand their structure
print("States Data:")
display(states_df.head())

print("\nState Rankings and Production Data:")
display(rankings_df.head())

print("\nGeneration Data:")
display(generation_df.head())


# In[3]:


# Check for missing values in each dataframe
print("Missing values in states_df:")
print(states_df.isnull().sum())

print("\nMissing values in rankings_df:")
print(rankings_df.isnull().sum())

print("\nMissing values in generation_df:")
print(generation_df.isnull().sum())


# In[4]:


# Drop irrelevant column
rankings_df = rankings_df.drop(columns=['Federal offshore production is not included in the Production Shares.'])

# Rename columns in generation_df and rankings_df for easier access
generation_df.rename(columns={'GENERATION (Megawatthours)': 'Generation_MWh'}, inplace=True)
rankings_df.rename(columns={
    'Production, U.S. Share': 'Production_Share',
    'Production, Rank': 'Production_Rank',
    'Consumption per Capita, Million Btu': 'Consumption_per_Capita_MMBtu',
    'Consumption per Capita, Rank': 'Consumption_Rank',
    'Expenditures per Capita, Dollars': 'Expenditures_per_Capita',
    'Expenditures per Capita, Rank': 'Expenditures_Rank'
}, inplace=True)


# Convert YEAR and MONTH columns in generation_df to datetime if needed
# Here, we combine YEAR and MONTH into a single 'Date' column for easier plotting
generation_df['Date'] = pd.to_datetime(generation_df['YEAR'].astype(str) + '-' + generation_df['MONTH'].astype(str) + '-01')



# In[5]:


# Merge generation data with states to get state names
generation_states = generation_df.merge(states_df[['State', 'Code']], left_on='STATE', right_on='Code', how='left')
generation_states.drop(columns=['Unnamed: 0'], inplace=True)  # Drop redundant columns
generation_states.rename(columns={'STATE': 'State_Abbrev'}, inplace=True)
generation_states.rename(columns={'State': 'State_Name'}, inplace=True)
generation_states.rename(columns={'ENERGY SOURCE': 'Energy Source'}, inplace=True)
generation_states.rename(columns={'TYPE OF PRODUCER': 'Type of Producer'}, inplace=True)

generation_states.head()


# In[6]:


# Merge rankings data with states to get state names
rankings_states = rankings_df.merge(states_df[['Code', 'State']], left_on='State', right_on='Code', how='left')
rankings_states.drop(columns=['Code'], inplace=True)  # Drop redundant columns

# Rename columns for clarity 
rankings_states.rename(columns={'State_x': 'State_Abbrev'}, inplace=True)
rankings_states.rename(columns={'State_y': 'State_Name'}, inplace=True)
rankings_states.head()


# ## Exploratory Data Analysis 

# We can start our EDA by looking at some summary statistics. First, we will explore the basic statistics of each dataset, focusing on key metrics like production share, consumption per capita, and expenditures.

# In[7]:


# Summary statistics for key columns in rankings_states
print("Summary Statistics for Rankings Data:")
print(rankings_states[['Production_Share', 'Consumption_per_Capita_MMBtu', 'Expenditures_per_Capita']].describe())

# Summary statistics for generation data
print("\nSummary Statistics for Generation Data:")
print(generation_states[['Generation_MWh']].describe())


# Now, we can look at some plots that will help visualize some of the trends that exist amongst our datasets. We will create a line plot of production for states of all of the different states over time, a countplot for a specific state's different energy sources (in this case, I have chosen my home state of Ohio), and a scatterplot to investigate any potential correlation between consumption and expenditure per capita. 

# In[8]:


# Line plot of general production trends for each state over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=generation_states, x='Date', y='Generation_MWh', hue='State_Name')
plt.title("Energy Generation Trends Over Time by State")
plt.xlabel("Date")
plt.ylabel("Generation (MWh)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[9]:


# Create countplot of energy source distribution for a specific state 
state = 'Ohio'
state_data = generation_states[generation_states['State_Name'] == state]
plt.figure(figsize=(10, 6))
sns.countplot(data=state_data, x='Energy Source')
plt.title(f"Energy Source Distribution in {state}")
plt.xticks(rotation=45)
plt.show()


# In[10]:


# Scatterplot of correlation between consumption per capita and expenditures per capita
plt.figure(figsize=(8, 6))
sns.scatterplot(data=rankings_states, x='Consumption_per_Capita_MMBtu', y='Expenditures_per_Capita')
plt.title("Correlation between Consumption per Capita and Expenditures per Capita")
plt.xlabel("Consumption per Capita (MMBtu)")
plt.ylabel("Expenditures per Capita (Dollars)")
plt.show()


# To showcase how production share varies across states, we will now create a choropleth map using Plotly.

# In[11]:


# Choropleth map of production share by state
fig = px.choropleth(rankings_states, 
                    locations='State_Abbrev', 
                    locationmode='USA-states', 
                    color='Production_Share',
                    hover_name='State_Name', 
                    color_continuous_scale="Viridis",
                    scope="usa",
                    title="Production Share by State (U.S. Share %)")
fig.show()


# To focus on the states with the highest production share, we’ll identify the top five and plot their monthly energy generation trends.

# In[12]:


# Select the top 5 states by production share
top_5_states = rankings_states.nlargest(5, 'Production_Share')['State_Name'].values
top_5_data = generation_states[generation_states['State_Name'].isin(top_5_states)]

fig = px.line(top_5_data, 
              x='Date', 
              y='Generation_MWh', 
              color='State_Name',
              title="Monthly Energy Generation Trends for Top 5 Producing States")
fig.show()


# It seems as though Texas is a huge producer of energy, especially relative to the other states in the U.S.

# It might be interesting to drill into the types of energy sources used in the U.S. This bar plot shows the distribution of different energy sources used across all states.

# In[13]:


# Exclude the "Total" energy source type from the data
filtered_generation_states = generation_states[generation_states['Energy Source'] != 'Total']

# Bar plot of energy sources used by all states, excluding "Total"
fig = px.histogram(filtered_generation_states, 
                   x='Energy Source', 
                   color='State_Name', 
                   title="Distribution of Energy Sources Across All States (Excluding 'Total')",
                   barmode='stack')
fig.show()


# It might also be interesting to look at energy source trends broken up by year over time, like the bar plot below. 

# In[14]:


yearly_generation = generation_states.groupby(['YEAR', 'Energy Source']).sum(numeric_only=True).reset_index()

fig = px.bar(yearly_generation, 
             x='YEAR', 
             y='Generation_MWh', 
             color='Energy Source', 
             title="Yearly Energy Generation by Source",
             labels={'Generation_MWh': 'Total Generation (MWh)'},
             barmode='stack')
fig.show()


# We can look at the states in terms of their relative average consumption per capita through the plot below. 

# In[15]:


# Sort rankings_states by consumption per capita and plot
sorted_consumption = rankings_states.sort_values(by='Consumption_per_Capita_MMBtu', ascending=False)
fig = px.bar(sorted_consumption, 
             x='State_Name', 
             y='Consumption_per_Capita_MMBtu', 
             title="Average Consumption per Capita by State",
             labels={'Consumption_per_Capita_MMBtu': 'Consumption per Capita (MMBtu)'})
fig.show()


# While we have identified states such as Texas to have very high production and consumption ranks, it might be interesting to investigate more of the states and compare their production and consumption ranks. This grouped bar plot can help showcase these kinds of insights here.

# In[16]:


fig = px.bar(rankings_states, 
             x='State_Name', 
             y=['Production_Rank', 'Consumption_Rank'], 
             title="State Rankings Comparison: Production vs. Consumption",
             labels={'value': 'Rank', 'variable': 'Metric'},
             barmode='group')
fig.update_layout(yaxis=dict(autorange='reversed'))
fig.show()


# ## Dash App Creation 

# Now that we have prepared our data and complete an initital cursory exploration of our data, we can create our dash app that allows for interactive investigation of the U.S. Energy data. 

# In[17]:


#app.run_server(debug=True, port=8050)


# In[18]:


# Initialize Dash app
app = dash.Dash(__name__)

# Layout for the Dash app
app.layout = html.Div([
    html.H1("U.S. Energy Production and Consumption Dashboard"),
    html.P("This interactive dashboard displays energy data for each U.S. state."),
    
    # Dropdown for state selection
    html.Label("Select a State:"),
    dcc.Dropdown(
        id='state-dropdown',
        options=[{'label': row['State_Name'], 'value': row['State_Abbrev']} for idx, row in rankings_states.iterrows()],
        value='OH'  # Default selection is Ohio
    ),
    
    # Radio buttons for selecting the dataset to display - either looking at 2022 individual state rankings and production/expenditures or yearly state generation data
    html.Label("Select Dataset:"),
    dcc.RadioItems(
        id='dataset-radio',
        options=[
            {'label': 'Production and Consumption (2022)', 'value': 'rankings'},
            {'label': 'Yearly Generation Data (2001-2022)', 'value': 'generation'}
        ],
        value='rankings'
    ),
    
    # Year slider for yearly data, shown only when the "generation" dataset is selected
    html.Div(id='year-slider-container', children=[
        html.Label("Select Year:"),
        dcc.Slider(
            id='year-slider',
            min=generation_states['YEAR'].min(),
            max=generation_states['YEAR'].max(),
            value=generation_states['YEAR'].min(),
            marks={str(year): str(year) for year in generation_states['YEAR'].unique()},
            step=None
        )
    ], style={'display': 'none'}),  # Initially hidden
    
    # Div container for plots: Scatter plot for 2022, pie chart, and rankings bar chart
    html.Div(id='plot-container', children=[
        html.Div(id='scatter-container', children=[dcc.Graph(id='energy-scatter-plot')], style={'width': '60%', 'display': 'inline-block'}),
        html.Div(id='line-container', children=[dcc.Graph(id='energy-line-plot')], style={'width': '100%', 'display': 'none'}),
        html.Div(id='pie-container', children=[dcc.Graph(id='energy-pie-plot')], style={'width': '50%', 'display': 'none'}),
        html.Div(id='rankings-bar-container', children=[dcc.Graph(id='rankings-bar-plot')], style={'width': '50%', 'display': 'none'})
    ], style={'display': 'flex'}),
    
    # Narrative and explanation for user guidance
    html.Div(id='narrative-container', children=[
        html.H2("Dash App Explanation and Walkthrough"),
        html.P("Welcome to the U.S. Energy Dashboard. Use the dropdown menu to select a state and choose a dataset view. You can explore either 2022 data for state production, consumption, and rankings data or historical generation data by year. For both views, you can select your state of interest. For yearly data, use the year slider and select energy sources of interest.")
    ])
])

# Callbacks for interactive components
@app.callback(
    [Output('energy-scatter-plot', 'figure'),
     Output('energy-line-plot', 'figure'),
     Output('energy-pie-plot', 'figure'),
     Output('rankings-bar-plot', 'figure'),
     Output('year-slider-container', 'style'),
     Output('scatter-container', 'style'),
     Output('line-container', 'style'),
     Output('pie-container', 'style'),
     Output('rankings-bar-container', 'style')],
    [Input('state-dropdown', 'value'),
     Input('dataset-radio', 'value'),
     Input('year-slider', 'value')]
)
def update_graph(selected_state, selected_dataset, selected_year): 
    # Default values in case dataset isn't selected yet
    scatter_fig = {'data': [], 'layout': {'title': 'No Data Selected'}}
    line_fig = {'data': [], 'layout': {'title': 'No Data Selected'}}
    pie_fig = {'data': [], 'layout': {'title': 'No Data Selected'}}
    rankings_fig = {'data': [], 'layout': {'title': 'No Data Selected'}}
    
    year_slider_style = {'display': 'none'}
    scatter_container_style = {'display': 'none'}
    line_container_style = {'display': 'none'}
    pie_container_style = {'display': 'none'}
    rankings_bar_container_style = {'display': 'none'}

    if selected_dataset == 'rankings':
        # Static 2022 data from rankings_states
        state_data = rankings_states[rankings_states['State_Abbrev'] == selected_state]
        
        # Scatter plot: Consumption vs Expenditures for all states
        scatter_data = rankings_states[['State_Name', 'Consumption_per_Capita_MMBtu', 'Expenditures_per_Capita']]
        scatter_fig = px.scatter(
            scatter_data,
            x='Consumption_per_Capita_MMBtu',
            y='Expenditures_per_Capita',
            color_discrete_sequence=['gray'],  # Set all points to gray
            title=f"Consumption vs Expenditures for All States in 2022",
            labels={'Consumption_per_Capita_MMBtu': 'Consumption per Capita (MMBTU)', 'Expenditures_per_Capita': 'Expenditures per Capita ($)'},
            hover_data={'State_Name': True, 'Consumption_per_Capita_MMBtu': True, 'Expenditures_per_Capita': True}  # Ensure state name appears on hover
        )
        
         # Add the selected state in green
        scatter_fig.add_scatter(
            x=[state_data['Consumption_per_Capita_MMBtu'].values[0]],
            y=[state_data['Expenditures_per_Capita'].values[0]],
            mode='markers',
            marker=dict(color='green', size=10),
            name=f"{state_data.iloc[0]['State_Name']} (Selected)",
            hovertemplate=f"<b>{state_data.iloc[0]['State_Name']}</b><br>Consumption per Capita: {state_data['Consumption_per_Capita_MMBtu'].values[0]}<br>Expenditures per Capita: {state_data['Expenditures_per_Capita'].values[0]}<br>"
       )
        
        
        # Update layout for the scatter plot with legend
        scatter_fig.update_layout(
        showlegend=False,  # Hide the legend as it is not necessary
        )
        
        # Rankings bar chart for Production, Consumption, and Expenditures Rank
        rankings = ['Production Rank', 'Consumption Rank', 'Expenditures Rank']
        ranking_values = [state_data['Production_Rank'].values[0], state_data['Consumption_Rank'].values[0], state_data['Expenditures_Rank'].values[0]]
        
        rankings_fig = px.bar(
            x=rankings,
            y=ranking_values,
            labels={'x': 'Rankings', 'y': 'Rank'},
            title=f"2022 Rankings for {state_data.iloc[0]['State_Name']}",
            color=rankings,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        rankings_fig.update_layout(
            yaxis=dict(range=[50, 0]),  # Fixed range from 50 to 0, lower rank is better
            showlegend=True,
            legend_title="Legend"  # Change legend title to 'Legend'
        )
        
        # Show scatter plot and rankings chart
        scatter_container_style = {'width': '60%', 'display': 'inline-block'}
        rankings_bar_container_style = {'width': '50%', 'display': 'inline-block'}
        
    else:
        # Dynamic yearly data from generation_states
        yearly_data = generation_states[(generation_states['State_Abbrev'] == selected_state) & 
                                        (generation_states['YEAR'] == selected_year)]
        
        # Filter out the "Total" energy source
        yearly_data = yearly_data[yearly_data['Energy Source'] != 'Total']
    
        # Plot stacked energy generation data for all sources
        line_fig = px.area(
            yearly_data,
            x='MONTH',
            y='Generation_MWh',
            color='Energy Source',
            labels={'MONTH': 'Month', 'Generation_MWh': 'Generation (MWh)'},
            title=f"Yearly Energy Generation Data for {yearly_data.iloc[0]['State_Name']} in {selected_year}",
            color_discrete_sequence=px.colors.qualitative.Set2  # Ensure consistent color palette
        )
        
        # Pie chart for production share by energy source
        pie_fig = px.pie(
            yearly_data,
            names='Energy Source',
            values='Generation_MWh',
            title=f"Energy Production Share in {selected_year}",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set2  # Ensure consistent color palette
        )
        
        # Update Pie chart to have fewer labels
        pie_fig.update_traces(textinfo='percent+label', insidetextorientation='radial', textposition='outside')
        pie_fig.update_layout(
            showlegend=True,  # Enable legend
            margin={"t": 105, "b": 20},  # Add extra margin at the bottom to avoid overlap
            title=dict(x=0.5, y=0.92),  # Move the title slightly upwards, within valid range
            annotations=[dict(
                x=0.5,
                y=-0.4,
                text="Energy Share by Source",
                showarrow=False,
                font=dict(size=12)
            )]
        )

        # Show year slider, line plot, and pie chart
        year_slider_style = {'display': 'block'}
        line_container_style = {'width': '100%', 'display': 'inline-block'}
        pie_container_style = {'width': '50%', 'display': 'inline-block'}

    return scatter_fig, line_fig, pie_fig, rankings_fig, year_slider_style, scatter_container_style, line_container_style, pie_container_style, rankings_bar_container_style


if __name__ == '__main__':
    app.run_server(debug=True)


# ## Conclusions

# This project aimed to visualize and analyze U.S. energy production and consumption trends using various datasets, providing insights through both exploratory data analysis and an interactive Dash application.  
# 
# EDA
# - The summary statistics of both the rankings and generation datasets provided an overview of the distributions and variability of energy-related metrics. For instance, the Production Share in the rankings dataset had a wide range, with values ranging from 0 to 25.5, indicating significant variation in energy production shares across states, indicating that some states produce a far larger portion of energy than others. Similarly, the Generation in MwH in the generation dataset displayed high variability, with some states generating orders of magnitude more energy than others. The statistical summaries also helped confirm that the datasets were appropriately processed for the analyses that follow.
# - A number of the visualizations in the EDA section of this project helped to identify interesting trends that could then be investigated further in the Dash App, such as looking more into some of the top energy producers and consumers in the U.S. to see a breakdown of their energy sources and a comparison of their production, consumption, and expenditure rankings.  
# 
# Dash Appplication  
# - The Dash app was developed with the goal of providing an interactive interface for users to explore energy data dynamically. The components, including dropdowns, radio buttons, and graphs, performed as expected, offering an intuitive user experience. Users can explore energy production and consumption data, with visualizations updating based on their selections. 
# - The line plot visualized generation trends effectively, showing how energy sources evolved over time. For example, Texas stood out as a consistently larger energy producer compared to other states. Additionally, the dominant energy sources for Texas shifted over time, moving from natural gas to petroleum and other biomass. This pattern highlighted the dynamic nature of the energy production landscape across the U.S. and underlined the importance of examining temporal trends to understand energy generation strategies.
# 
# In conclusion, this project successfully provides a comprehensive analysis of U.S. energy production and consumption trends, presenting clear visualizations that highlight the varying energy strategies of different states. The interactive Dash app enhanced the user experience, making the complex data more accessible and engaging. Further analysis could expand on specific energy sources and their impacts or explore additional variables such as energy efficiency measures or policy influences.

# In[19]:


# app.run_server()

