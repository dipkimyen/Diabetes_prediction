import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
url = "https://raw.githubusercontent.com/dipkimyen/Diabetes_prediction/master/diabetes_prediction_dataset.csv"
df = pd.read_csv(url)

# Create the Dash app
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Diabetes Prediction Dashboard"),
    
    # Dropdown to filter data by 'diabetes' column
    dcc.Dropdown(
        id='diabetes-filter',
        options=[{'label': x, 'value': x} for x in df['diabetes'].unique()],
        value=df['diabetes'].unique()[0],
        multi=False
    ),
    
    # Scatter plot to show the relationship between 'blood_glucose_level' and 'bmi'
    dcc.Graph(id='scatter-plot'),
])

# Define callback to update the scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('diabetes-filter', 'value')]
)
def update_scatter_plot(selected_diabetes):
    filtered_df = df[df['diabetes'] == selected_diabetes]
    
    fig = px.scatter(filtered_df, x='bmi', y='blood_glucose_level', color='age',
                     title=f"Scatter plot for {selected_diabetes} diabetes")
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
