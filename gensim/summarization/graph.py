#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains abstract class IGraph represents graphs interface and
class Graph (based on IGraph) which implements undirected graph.

Examples
--------

Create simple graph with 4 nodes.

>>> g = Graph()
>>> g.add_node('Felidae')
>>> g.add_node('Lion')
>>> g.add_node('Tiger')
>>> g.add_node('Wolf')
>>> sorted(g.nodes())
['Felidae', 'Lion', 'Tiger', 'Wolf']

Add some edges and check neighbours.

>>> g.add_edge(("Felidae", "Lion"))
>>> g.add_edge(("Felidae", "Tiger"))
>>> g.neighbors("Felidae")
['Lion', 'Tiger']

One node has no neighbours.

>>> g.neighbors("Wolf")
[]

"""

from abc import ABCMeta, abstractmethod


class IGraph(object):
    """Represents the interface or contract that the graph for TextRank
    should implement.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def nodes(self):
        """Returns all nodes of graph.

        Returns
        -------
        list of hashable
            Nodes of graph.

        """
        pass

    @abstractmethod
    def edges(self):
        """Returns all edges of graph.

        Returns
        -------
        list of (hashable, hashable)
            Edges of graph.

        """
        pass

    @abstractmethod
    def neighbors(self, node):
        """Return all nodes that are directly accessible from given node.

        Parameters
        ----------
        node : hashable
            Given node identifier.

        Returns
        -------
        list of hashable
            Nodes directly accessible from given `node`.

        """
        pass

    @abstractmethod
    def has_node(self, node):
        """Returns whether the requested node exists.

        Parameters
        ----------
        node : hashable
            Given node identifier.

        Returns
        -------
        bool
            True if `node` exists, False otherwise.

        """
        pass

    @abstractmethod
    def add_node(self, node, attrs=None):
        """Adds given node to the graph.

        Note
        ----
        While nodes can be of any type, it's strongly recommended to use only numbers and single-line strings
        as node identifiers if you intend to use write().

        Parameters
        ----------
        node : hashable
            Given node
        attrs : list, optional
            Node attributes specified as (attribute, value)

        """
        pass

    @abstractmethod
    def add_edge(self, edge, wt=1, label='', attrs=None):
        """Adds an edge to the graph connecting two nodes. An edge, here,
        is a tuple of two nodes.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.
        wt : float, optional
            Weight of new edge.
        label : str, optional
            Edge label.
        attrs : list, optional
            Node attributes specified as (attribute, value)

        """
        pass

    @abstractmethod
    def has_edge(self, edge):
        """Returns whether an edge exists.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        Returns
        -------
        bool
            True if `edge` exists, False otherwise.

        """
        pass

    @abstractmethod
    def edge_weight(self, edge):
        """Returns weigth of given edge.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        Returns
        -------
        float
            Edge weight.

        """
        pass

    @abstractmethod
    def del_node(self, node):
        """Removes node and its edges from the graph.

        Parameters
        ----------
        node : hashable
            Node to delete.

        """
        pass


class Graph(IGraph):
    """
    Implementation of an undirected graph, based on IGraph.

    Attributes
    ----------
    Graph.WEIGHT_ATTRIBUTE_NAME : str
        Name of weight attribute in graph.
    Graph.DEFAULT_WEIGHT : float
        Weight set by default.
    Graph.LABEL_ATTRIBUTE_NAME : str
        Default name of attribute. Not used.
    Graph.DEFAULT_LABEL : str
        Label set by default. Not used.

    """

    WEIGHT_ATTRIBUTE_NAME = "weight"
    DEFAULT_WEIGHT = 0

    LABEL_ATTRIBUTE_NAME = "label"
    DEFAULT_LABEL = ""

    def __init__(self):
        """Initializes object."""

        # Metadata about edges
        # Mapping: Edge -> Dict mapping, lablel-> str, wt->num
        self.edge_properties = {}
        # Key value pairs: (Edge -> Attributes)
        self.edge_attr = {}

        # Metadata about nodes
        # Pairing: Node -> Attributes
        self.node_attr = {}
        # Pairing: Node -> Neighbors
        self.node_neighbors = {}

    def has_edge(self, edge):
        """Returns whether an edge exists.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        Returns
        -------
        bool
            True if `edge` exists, False otherwise.

        """
        u, v = edge
        return (u, v) in self.edge_properties and (v, u) in self.edge_properties

    def edge_weight(self, edge):
        """Returns weight of given edge.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        Returns
        -------
        float
            Edge weight.

        """
        return self.get_edge_properties(edge).setdefault(self.WEIGHT_ATTRIBUTE_NAME, self.DEFAULT_WEIGHT)

    def neighbors(self, node):
        """Returns all nodes that are directly accessible from given node.

        Parameters
        ----------
        node : hashable
            Given node identifier.

        Returns
        -------
        list of hashable
            Nodes directly accessible from given `node`.

        """
        return self.node_neighbors[node]

    def has_node(self, node):
        """Returns whether the requested node exists.

        Parameters
        ----------
        node : hashable
            Given node.

        Returns
        -------
        bool
            True if `node` exists, False otherwise.

        """
        return node in self.node_neighbors

    def add_edge(self, edge, wt=1, label='', attrs=None):
        """Adds an edge to the graph connecting two nodes.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.
        wt : float, optional
            Weight of new edge.
        label : str, optional
            Edge label.
        attrs : list, optional
            Node attributes specified as (attribute, value).

        Raises
        ------
        ValueError
            If `edge` already exists in graph.

        """
        if attrs is None:
            attrs = []
        u, v = edge
        if v not in self.node_neighbors[u] and u not in self.node_neighbors[v]:
            self.node_neighbors[u].append(v)
            if u != v:
                self.node_neighbors[v].append(u)

            self.add_edge_attributes((u, v), attrs)
            self.set_edge_properties((u, v), label=label, weight=wt)
        else:
            raise ValueError("Edge (%s, %s) already in graph" % (u, v))

    def add_node(self, node, attrs=None):
        """Adds given node to the graph.

        Note
        ----
        While nodes can be of any type, it's strongly recommended
        to use only numbers and single-line strings as node identifiers if you
        intend to use write().

        Parameters
        ----------
        node : hashable
            Given node.
        attrs : list of (hashable, hashable), optional
            Node attributes specified as (attribute, value)

        Raises
        ------
        ValueError
            If `node` already exists in graph.

        """
        if attrs is None:
            attrs = []
        if node not in self.node_neighbors:
            self.node_neighbors[node] = []
            self.node_attr[node] = attrs
        else:
            raise ValueError("Node %s already in graph" % node)

    def nodes(self):
        """Returns all nodes of the graph.

        Returns
        -------
        list of hashable
            Nodes of graph.

        """
        return list(self.node_neighbors.keys())

    def edges(self):
        """Returns all edges of the graph.

        Returns
        -------
        list of (hashable, hashable)
            Edges of graph.

        """
        return [a for a in self.edge_properties.keys()]

    def del_node(self, node):
        """Removes given node and its edges from the graph.

        Parameters
        ----------
        node : hashable
            Given node.

        """
        for each in list(self.neighbors(node)):
            if each != node:
                self.del_edge((each, node))
        del self.node_neighbors[node]
        del self.node_attr[node]

    def get_edge_properties(self, edge):
        """Returns properties of given given edge. If edge doesn't exist
        empty dictionary will be returned.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        Returns
        -------
        dict
            Properties of graph.

        """
        return self.edge_properties.setdefault(edge, {})

    def add_edge_attributes(self, edge, attrs):
        """Adds attributes `attrs` to given edge, order of nodes in edge doesn't matter.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.
        attrs : list
            Provided attributes to add.

        """
        for attr in attrs:
            self.add_edge_attribute(edge, attr)

    def add_edge_attribute(self, edge, attr):
        """Adds attribute `attr` to given edge, order of nodes in edge doesn't matter.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        attr : object
            Provided attribute to add.

        """
        self.edge_attr[edge] = self.edge_attributes(edge) + [attr]

        if edge[0] != edge[1]:
            self.edge_attr[(edge[1], edge[0])] = self.edge_attributes((edge[1], edge[0])) + [attr]

    def edge_attributes(self, edge):
        """Returns attributes of given edge.

        Note
        ----
        In case of non existing edge returns empty list.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        Returns
        -------
        list
            Attributes of given edge.

        """
        try:
            return self.edge_attr[edge]
        except KeyError:
            return []

    def set_edge_properties(self, edge, **properties):
        """Adds `properties` to given edge, order of nodes in edge doesn't matter.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        properties : dict
            Properties to add.

        """
        self.edge_properties.setdefault(edge, {}).update(properties)
        if edge[0] != edge[1]:
            self.edge_properties.setdefault((edge[1], edge[0]), {}).update(properties)

    def del_edge(self, edge):
        """Removes given edges from the graph.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        """
        u, v = edge
        self.node_neighbors[u].remove(v)
        self.del_edge_labeling((u, v))
        if u != v:
            self.node_neighbors[v].remove(u)
            self.del_edge_labeling((v, u))

    def del_edge_labeling(self, edge):
        """Removes attributes and properties of given edge.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        """
        keys = [edge, edge[::-1]]

        for key in keys:
            for mapping in [self.edge_properties, self.edge_attr]:
                try:
                    del mapping[key]
                except KeyError:
                    pass
