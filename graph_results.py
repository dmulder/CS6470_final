#!/usr/bin/env python3
import plotly.graph_objects as go
from results import *

fig = go.Figure()
for activation in activation_outputs.keys():
    fig.add_scatter(
        name=activation.replace('_', ' ').title(),
        x=[i for i in range(1, len(activation_outputs[activation])+1)],
        y=activation_outputs[activation],
    )
fig.update_layout(title='Comparison of Activation Functions for Identifying Recyclables with 5 hidden layers',
                   xaxis_title='Epochs',
                   yaxis_title='Accuracy')
fig.show()

fig1 = go.Figure()
for layer in layer_outputs.keys():
    fig1.add_scatter(
        name='%d Layers' % layer,
        x=[i for i in range(1, len(layer_outputs[layer])+1)],
        y=layer_outputs[layer],
    )
fig1.update_layout(title='Comparison of elu Layers for Identifying Recyclables',
                   xaxis_title='Epochs',
                   yaxis_title='Accuracy')
fig1.show()

fig2 = go.Figure()
for layer in swish_layer_outputs.keys():
    fig2.add_scatter(
        name='%d Layers' % layer,
        x=[i for i in range(1, len(swish_layer_outputs[layer])+1)],
        y=swish_layer_outputs[layer],
    )
fig2.update_layout(title='Comparison of Swish Layers for Identifying Recyclables',
                   xaxis_title='Epochs',
                   yaxis_title='Accuracy')
fig2.show()


