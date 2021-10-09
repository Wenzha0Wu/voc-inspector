import io
from base64 import b64encode
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
from datetime import date, timedelta, datetime
import math
import xarray as xr
import plotly.graph_objects as go
import PIL

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#PIL.Image.MAX_IMAGE_PIXELS=None
img = PIL.Image.open('./assets/yanshan_1-16.jpg')

nc_obs = xr.open_dataset('./data_obs_example.nc')
nc_pred = xr.open_dataset('./data_pred_example.nc')
df_obs = nc_obs['voc']
df_pred = nc_pred['voc']
data = {'obs' : df_obs, 'pred' : df_pred}
nhours=dict()
ndays=dict()
st_date=dict()
ed_date=dict()
lon=dict()
lat=dict()
for period in ['obs', 'pred']:
    nhours[period] = len(data[period]['time'])
    ndays[period] = nhours[period]//24
    st_date[period] = pd.to_datetime(data[period].time.data[0]).to_pydatetime()
    ed_date[period] = st_date[period]+timedelta(days=ndays[period])
    lon[period]=data[period].lon.data
    lat[period]=data[period].lat.data


# server
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
    html.H1(children='VOCs monitoring & alerting platform'),
    dcc.Tabs(id='tabs', value='obs', children=[
        dcc.Tab(label='Observation', value='obs'),
        dcc.Tab(label='Prediction', value='pred'),
    ]),
    html.Center([
        dcc.Graph(id="graph", style={'width': '70vw', 'height': '75vh'}),
        # Preview
        #dcc.Graph(id="preview"),
    ],),# {'columns': '2'}
    dcc.Slider(
        id="date-slider",
        min=0,
        max=nhours['obs']-1,
        value=0,
        step=1,
        updatemode="drag"
	),
    html.Div([
        html.Button('Prev Day', id='btn_prev', n_clicks=0, style={'marginRight':'10px'}),
        html.Button('Next Day', id='btn_next', n_clicks=0),
    ], style={'width': '30%', 'display': 'inline-block'}),
    html.Div([
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=st_date['obs'],
            max_date_allowed=ed_date['obs'],
            start_date=st_date['obs'],
            end_date=ed_date['obs'],
            display_format='Y-M-D',
            style={'marginRight':'10px'}
        ),
        html.Button('Download Animation', id='downloadbtn', n_clicks=0, style={'float': 'right', 'display': 'inline-block'}),
    ], style={'width': '54%', 'display': 'inline-block'}),
    dcc.Download(id='download-video'),
])

@app.callback(
    Output("download-video", "data"),
    Input("downloadbtn", "n_clicks"),
    Input("tabs", "value"),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    prevent_initial_call=True,
)
def func(n_clicks, select_mode, dump_st_date, dump_ed_date):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    encoded = ''
    if "downloadbtn" in changed_id:
        df = data[select_mode]
        st_frame = int((datetime.fromisoformat(dump_st_date) - st_date[select_mode]).total_seconds() // 3600)
        ed_frame = int((datetime.fromisoformat(dump_ed_date) - st_date[select_mode]).total_seconds() // 3600)
        frames=[]
        for tick in range(st_frame, ed_frame):
            frames.append(
                go.Frame(data=
                    go.Contour(
                        z= df[tick, :, :],
                        x=lon[select_mode],
                        y=lat[select_mode],
                        opacity=0.85,
                        contours_coloring='heatmap',
                        zmin=0,
                        zmax=6,
                        colorscale='RdBu_r',
                        colorbar_title=dict(text="VOCs(mg/m3)"),
                        hovertemplate="lat:%{y}<br>lon:%{x}<br>voc:%{z}<extra></extra>"
                    ),
                    layout=go.Layout(
                        title='VOCs concentration across area on {}'.format((st_date[select_mode] + timedelta(hours=tick)).isoformat(sep=' ')),
                        xaxis_title="lon(°)",
                        yaxis_title="lat(°)",
                    )
	    		)
            )
        fig = go.Figure(data =
            go.Contour(
                z= df[st_frame, :, :],
                x=lon[select_mode],
                y=lat[select_mode],
                opacity=0.85,
                contours_coloring='heatmap',
                zmin=0,
                zmax=6,
                colorscale='RdBu_r',
                colorbar_title=dict(text="VOCs(mg/m3)"),
                hovertemplate="lat:%{y}<br>lon:%{x}<br>voc:%{z}<extra></extra>"
            ),
            layout=go.Layout(
                title='VOCs concentration across area on {}'.format((st_date[select_mode] + timedelta(hours=st_frame)).isoformat(sep=' ')),
                xaxis_title="lon(°)",
                yaxis_title="lat(°)",
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
	    				    dict(
                                label="Play",
                                method="animate",
                                args=[None, {"frame":{"duration": 100}, "transition":{"duration":20}}]
                            ),
                            # Using Pause button
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"
                            }
                        ],
                        direction="left",
                        showactive=False,
                        x=1.1,
                        y=-0.07,
                    )
                ]
            ),
            frames=frames,
        )
        fig.add_layout_image(
                    dict(
                        source=img,
                        xref="x",
                        yref="y",
                        x=lon[select_mode][0],
                        y=lat[select_mode][-1],
                        sizex=lon[select_mode][-1] - lon[select_mode][0],
                        sizey=lat[select_mode][-1] - lat[select_mode][0],
                        sizing="stretch",
                        opacity=0.8,
                        layer="below")
        )
        buffer = io.StringIO()
        fig.write_html(buffer)
        encoded=buffer.getvalue()
        return dict(content=encoded, filename="voc_from{}to{}.html".format(dump_st_date, dump_ed_date))

''' Preview
@app.callback(
    Output("preview", "figure"), 
    Input("tabs", "value"),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
    )
def preview_heatmap(select_mode, dump_st_date, dump_ed_date):
    df = data[select_mode]
    st_frame = int((datetime.fromisoformat(dump_st_date) - st_date[select_mode]).total_seconds() // 3600)
    ed_frame = int((datetime.fromisoformat(dump_ed_date) - st_date[select_mode]).total_seconds() // 3600)
    frames=[]
    for tick in range(st_frame, ed_frame):
        frames.append(
            go.Frame(data=
                go.Contour(
                    z= df[tick, :, :],
                    x=lon[select_mode],
                    y=lat[select_mode],
                    opacity=1,
                    contours_coloring='heatmap',
                    zmin=0,
                    zmax=6,
                    colorscale='RdBu_r',
                    colorbar_title=dict(text="VOCs(mg/m3)"),
                    hovertemplate="lat:%{y}<br>lon:%{x}<br>voc:%{z}<extra></extra>"
                ),
                layout=go.Layout(
                    title='VOCs concentration across area on {}'.format((st_date[select_mode] + timedelta(hours=tick)).isoformat(sep=' ')),
                    xaxis_title="lon(°)",
                    yaxis_title="lat(°)",
                )
			)
        )
    fig = go.Figure(data =
        go.Contour(
            z= df[st_frame, :, :],
            x=lon[select_mode],
            y=lat[select_mode],
            opacity=1,
            contours_coloring='heatmap',
            zmin=0,
            zmax=6,
            colorscale='RdBu_r',
            colorbar_title=dict(text="VOCs(mg/m3)"),
            hovertemplate="lat:%{y}<br>lon:%{x}<br>voc:%{z}<extra></extra>"
        ),
        layout=go.Layout(
            title='VOCs concentration across area on {}'.format((st_date[select_mode] + timedelta(hours=st_frame)).isoformat(sep=' ')),
            xaxis_title="lon(°)",
            yaxis_title="lat(°)",
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
					    dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame":{"duration": 100}, "transition":{"duration":20}}]
                        ),
                        # Using Pause button
                        #{
                        #    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                        #                      "mode": "immediate",
                        #                      "transition": {"duration": 0}}],
                        #    "label": "Pause",
                        #    "method": "animate"
                        #}
                    ],
                    direction="left",
                    showactive=False,
                    x=1.1,
                    y=-0.07,
                )
            ]
        ),
        frames=frames,
    )
    return fig
'''


@app.callback(
    Output('date-picker-range', 'min_date_allowed'),
    Output('date-picker-range', 'max_date_allowed'),
    Output('date-picker-range', 'start_date'),
    Output('date-picker-range', 'end_date'),
    [Input('tabs', 'value')])
def set_slider_options(select_mode):
    return st_date[select_mode], ed_date[select_mode], st_date[select_mode], ed_date[select_mode] 

@app.callback(
    Output('date-slider', 'max'),
    [Input('tabs', 'value')])
def set_slider_options(select_mode):
    return nhours[select_mode] - 1

@app.callback(
    Output('date-slider', 'value'),
    State('date-slider', 'value'),
    State('tabs', 'value'),
    [Input('tabs', 'value'),
	Input("btn_prev", "n_clicks"),
    Input("btn_next", "n_clicks")])
def set_slider_step(cur_val, cur_mode, select_mode, stepforward, stepbackward):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    val = 0
    if "btn_prev" in changed_id:
        val = max(cur_val - 24, 0)
    if "btn_next" in changed_id:
        val = min(cur_val + 24, nhours[select_mode] - 1)
    if "tabs" in changed_id:
        prop = cur_val/nhours[cur_mode]
        val = math.floor((nhours[select_mode] - 1)*prop)
    return val  

@app.callback(
    Output("graph", "figure"), 
    [Input("date-slider", "value"), 
     Input("tabs", "value")])
def update_heatmap(select_hour, select_mode):
    df = data[select_mode]
    fig = go.Figure(data =
        go.Contour(
            z= df[select_hour, :, :],
            x=lon[select_mode],
            y=lat[select_mode],
            opacity=0.85,
            contours_coloring='heatmap',
            zmin=0,
            zmax=6,
            colorscale='RdBu_r',
            colorbar_title=dict(text="VOCs(mg/m3)"),
            hovertemplate="lat:%{y}<br>lon:%{x}<br>voc:%{z}<extra></extra>"
        ),
    )
    fig.add_layout_image(
                dict(
                    source=img,
                    xref="x",
                    yref="y",
                    x=lon[select_mode][0],
                    y=lat[select_mode][-1],
                    sizex=lon[select_mode][-1] - lon[select_mode][0],
                    sizey=lat[select_mode][-1] - lat[select_mode][0],
                    sizing="stretch",
                    opacity=0.8,
                    layer="below")
    )
    fig.update_layout(
        title='VOCs concentration across area on {}'.format((st_date[select_mode] + timedelta(hours=select_hour)).isoformat(sep=' ')),
        xaxis_title="lon(°)",
        yaxis_title="lat(°)",
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)

