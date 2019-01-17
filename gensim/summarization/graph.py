#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""This module contains abstract class IGraph represents graphs interface and
class Graph (based on IGraph) which implements undirected graph.

Examples
--------

Create simple graph with 4 nodes.

.. sourcecode:: pycon

    >>> g = Graph()
    >>> g.add_node('Felidae')
    >>> g.add_node('Lion')
    >>> g.add_node('Tiger')
    >>> g.add_node('Wolf')
    >>> sorted(g.nodes())
    ['Felidae', 'Lion', 'Tiger', 'Wolf']

Add some edges and check neighbours.

.. sourcecode:: pycon

    >>> g.add_edge(("Felidae", "Lion"))
    >>> g.add_edge(("Felidae", "Tiger"))
    >>> g.neighbors("Felidae")
    ['Lion', 'Tiger']

One node has no neighbours.

.. sourcecode:: pycon

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
    def __len__(self):
        """Returns number of nodes in graph"""
        pass

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
    def add_node(self, node):
        """Adds given node to the graph.

        Note
        ----
        While nodes can be of any type, it's strongly recommended to use only numbers and single-line strings
        as node identifiers if you intend to use write().

        Parameters
        ----------
        node : hashable
            Given node

        """
        pass

    @abstractmethod
    def add_edge(self, edge, wt=1):
        """Adds an edge to the graph connecting two nodes. An edge, here,
        is a tuple of two nodes.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.
        wt : float, optional
            Weight of new edge.

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
    Graph.DEFAULT_WEIGHT : float
        Weight set by default.

    """

    DEFAULT_WEIGHT = 0

    def __init__(self):
        """Initializes object."""
        # Pairing and metadata about edges
        # Mapping: Node->
        #    Dict mapping of Neighbor -> weight
        self.node_neighbors = {}

    def __len__(self):
        """Returns number of nodes in graph"""
        return len(self.node_neighbors)

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
        return (u in self.node_neighbors
            and v in self.node_neighbors
            and v in self.node_neighbors[u]
            and u in self.node_neighbors[v])

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
        u, v = edge
        return self.node_neighbors.get(u, {}).get(v, self.DEFAULT_WEIGHT)

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
        return list(self.node_neighbors[node])

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

    def add_edge(self, edge, wt=1):
        """Adds an edge to the graph connecting two nodes.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.
        wt : float, optional
            Weight of new edge.

        Raises
        ------
        ValueError
            If `edge` already exists in graph.

        """
        if wt == 0.0:
            # empty edge is similar to no edge at all or removing it
            if self.has_edge(edge):
                self.del_edge(edge)
            return
        u, v = edge
        if v not in self.node_neighbors[u] and u not in self.node_neighbors[v]:
            self.node_neighbors[u][v] = wt
            if u != v:
                self.node_neighbors[v][u] = wt
        else:
            raise ValueError("Edge (%s, %s) already in graph" % (u, v))

    def add_node(self, node):
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

        Raises
        ------
        ValueError
            If `node` already exists in graph.

        """
        if node in self.node_neighbors:
            raise ValueError("Node %s already in graph" % node)

        self.node_neighbors[node] = {}

    def nodes(self):
        """Returns all nodes of the graph.

        Returns
        -------
        list of hashable
            Nodes of graph.

        """
        return list(self.node_neighbors)

    def edges(self):
        """Returns all edges of the graph.

        Returns
        -------
        list of (hashable, hashable)
            Edges of graph.

        """
        return list(self.iter_edges())

    def iter_edges(self):
        """Returns iterator of all edges of the graph.

        Yields
        -------
        (hashable, hashable)
            Edges of graph.

        """
        for u in self.node_neighbors:
            for v in self.node_neighbors[u]:
                yield (u, v)

    def del_node(self, node):
        """Removes given node and its edges from the graph.

        Parameters
        ----------
        node : hashable
            Given node.

        """
        for each in self.neighbors(node):
            if each != node:
                self.del_edge((each, node))
        del self.node_neighbors[node]

    def del_edge(self, edge):
        """Removes given edges from the graph.

        Parameters
        ----------
        edge : (hashable, hashable)
            Given edge.

        """
        u, v = edge
        del self.node_neighbors[u][v]
        if u != v:
            del self.node_neighbors[v][u]
