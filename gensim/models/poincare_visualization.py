#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayantjain1992@gmail.com>
# Copyright (C) 2017 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Utilities for creating 2-D visualizations of Poincare models and Poincare distance heatmaps.

"""

import logging

from collections import Counter
import numpy as np
import plotly.graph_objs as go

from gensim.models.poincare import PoincareKeyedVectors


logger = logging.getLogger(__name__)


def poincare_2d_visualization(model, tree, figure_title, num_nodes=50, show_node_labels=[]):
    """Method to create a 2-d plot of the nodes and edges of a 2-d poincare embedding.

    Parameters
    ----------
    model : PoincareModel instance
        The model to visualize, model size must be 2.
    tree : set
        Set of tuples containing the direct edges present in the original dataset.
    figure_title : str
        Title of the plotted figure.
    num_nodes : int or None
        Number of nodes for which edges are to be plotted. If None, all edges are plotted.
        Helpful to limit this in case the data is too large to avoid a messy plot.
    show_node_labels : list/iterable
        List of nodes for which to show labels by default.

    Returns
    -------
    plotly.graph_objs.Figure instance

    """
    vectors = model.kv.syn0
    if vectors.shape[1] != 2:
        raise ValueError('Can only plot 2-D vectors')

    node_labels = model.kv.index2word
    nodes_x = list(vectors[:, 0])
    nodes_y = list(vectors[:, 1])
    nodes = go.Scatter(
        x=nodes_x, y=nodes_y,
        mode='markers',
        marker=dict(color='rgb(30, 100, 200)'),
        text=node_labels,
        textposition='bottom'
    )

    nodes_x, nodes_y, node_labels = [], [], []
    for node in show_node_labels:
        vector = model.kv[node]
        nodes_x.append(vector[0])
        nodes_y.append(vector[1])
        node_labels.append(node)
    nodes_with_labels = go.Scatter(
        x=nodes_x, y=nodes_y,
        mode='markers+text',
        marker=dict(color='rgb(200, 100, 200)'),
        text=node_labels,
        textposition='bottom'
    )

    node_out_degrees = Counter(hypernym_pair[1] for hypernym_pair in tree)
    if num_nodes is None:
        chosen_nodes = list(node_out_degrees.keys())
    else:
        chosen_nodes = list(sorted(node_out_degrees.keys(), key=lambda k: -node_out_degrees[k]))[:num_nodes]

    edges_x = []
    edges_y = []
    for u, v in tree:
        if not(u in chosen_nodes or v in chosen_nodes):
            continue
        vector_u = model.kv[u]
        vector_v = model.kv[v]
        edges_x += [vector_u[0], vector_v[0], None]
        edges_y += [vector_u[1], vector_v[1], None]
    edges = go.Scatter(
        x=edges_x, y=edges_y, mode="line", hoverinfo=False,
        line=dict(color='rgb(50,50,50)', width=1))

    layout = go.Layout(
        title=figure_title, showlegend=False, hovermode='closest', width=800, height=800)
    return go.Figure(data=[edges, nodes, nodes_with_labels], layout=layout)


def poincare_distance_heatmap(
    origin_point, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), num_points=100):
    """
    Creates a heatmap of Poincare distances from `origin_point` for each point (x, y),
    where x and y lie in `x_range` and `y_range` respectively, with `num_points` points
    chosen uniformly in both ranges.

    Parameters
    ----------
    origin_point : tuple (int, int)
        (x, y) from which distances are to be measured and plotted.
    x_range : tuple (int, int)
        Range for x-axis from which to choose `num_points` points.
    y_range : tuple (int, int)
        Range for y-axis from which to choose `num_points` points.
    num_points : int
        Number of points to choose from `x_range` and `y_range`.

    Returns
    -------
    plotly.graph_objs.Figure instance

    Notes
    -----
    Points outside the unit circle are ignored, since the Poincare distance is defined
    only for points inside the circle boundaries (exclusive of the boundary).

    """
    epsilon = 1e-8  # Can't choose (-1.0, -1.0) or (1.0, 1.0), distance undefined
    x_range, y_range = list(x_range), list(y_range)
    if x_range[0] == -1.0 and y_range[0] == -1.0:
        x_range[0] += epsilon
        y_range[0] += epsilon
    if x_range[0] == 1.0 and y_range[0] == 1.0:
        x_range[0] -= epsilon
        y_range[0] -= epsilon

    x_axis_values = np.linspace(x_range[0], x_range[1], num=num_points)
    y_axis_values = np.linspace(x_range[0], x_range[1], num=num_points)
    x, y = np.meshgrid(x_axis_values, y_axis_values)
    all_points = np.dstack((x, y)).swapaxes(1, 2).swapaxes(0, 1).reshape(2, num_points ** 2).T
    norms = np.linalg.norm(all_points, axis=1)
    all_points = all_points[norms < 1]

    origin_point = np.array(origin_point)
    all_distances = PoincareKeyedVectors.poincare_dists(origin_point, all_points)

    distances = go.Scatter(
        x=all_points[:, 0],
        y=all_points[:, 1],
        mode='markers',
        marker=dict(
            size='9',
            color=all_distances,
            colorscale='Viridis',
            showscale=True,
            colorbar=go.ColorBar(
                title='Poincare Distance'
            ),
        ),
        text=[
            'Distance from (%.2f, %.2f): %.2f' % (origin_point[0], origin_point[1], d)
            for d in all_distances],
        name='',  # To avoid the default 'trace 0'
    )

    origin = go.Scatter(
        x=[origin_point[0]],
        y=[origin_point[1]],
        name='Distance from (%.2f, %.2f)' % (origin_point[0], origin_point[1]),
        mode='markers+text',
        marker=dict(
            size='10',
            color='rgb(200, 50, 50)'
        )
    )

    layout = go.Layout(
        width=900,
        height=800,
        showlegend=False,
        title='Poincare Distances from (%.2f, %.2f)' % (origin_point[0], origin_point[1]),
        hovermode='closest',
    )

    return go.Figure(data=[distances, origin], layout=layout)
