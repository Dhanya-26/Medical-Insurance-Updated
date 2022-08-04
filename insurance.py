import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)

app = Dash(__name__)

# -- Import and clean data (importing csv into pandas)
# df = pd.read_csv("intro_bees.csv")
df = pd.read_csv("health-insurance-coverage-2019.csv")
df.dropna(inplace=True)
df = df.groupby(['LocationDesc', 'LocationAbbr', 'Insurance_Status'])[['Value']].mean()
df.reset_index(inplace=True)
# print(df[:5])

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Insurance Plans status", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_plan",
                 options=[
                    {"label": Insurance_Status, "value": Insurance_Status}
                    for Insurance_Status in df.Insurance_Status.unique()],
                 multi=False,
                 value="With Private Health Insurance",
                 style={'width': "40%"}
                 ),
    html.Br(),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='disease_dashboard', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='disease_dashboard', component_property='figure')],
    [Input(component_id='slct_plan', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The category chosen by user was: {}".format(option_slctd)

    dff = df.copy()
    dff = dff[dff["Insurance_Status"] == option_slctd]
    

    # Plotly Express
    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='LocationAbbr',
        scope="usa",
        color='Value',
        hover_data=['LocationDesc', 'LocationAbbr','Insurance_Status'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Value'},
        template='plotly_dark'
    )

    #Plotly Graph Objects (GO)
    """ fig = go.Figure(
       data=[go.Choropleth(
             locationmode='USA-states',
             locations=dff['LocationAbbr'],
             hoverinfo=['LocationDesc', 'LocationAbbr','Insurance_Status',''],
             z=dff["Value"].astype(float),
             colorscale='Reds',
        )]
    ) """
    
    fig.update_layout(
         title_text="Insurance plans ",
         title_xanchor="center",
         title_font=dict(size=24),
         title_x=0.5,
         geo=dict(scope='usa'),
     )

    return container, fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)