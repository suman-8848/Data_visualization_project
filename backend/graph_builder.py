"""
Graph Builder: Constructs knowledge graphs from entities and relationships
Uses NetworkX for graph structure and computes layout coordinates
"""

import networkx as nx
from typing import List, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds and processes knowledge graphs"""
    
    def __init__(self):
        """Initialize graph builder"""
        self.graph = None
    
    def build_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        tokens: List[str] = None,
        attention: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from entities and relationships
        
        Args:
            entities: List of entity dictionaries
            relationships: List of relationship dictionaries
            tokens: Optional token list for attention mapping
            attention: Optional attention data for edge weighting
            
        Returns:
            Graph data with nodes, edges, and layout coordinates
        """
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Add nodes (entities)
        for entity in entities:
            # Extract entity data, renaming 'label' to 'entity_type' to avoid conflict
            entity_attrs = {k: v for k, v in entity.items() if k != 'label'}
            self.graph.add_node(
                entity["id"],
                label=entity["text"],
                entity_type=entity.get("label", "UNKNOWN"),
                **entity_attrs
            )
        
        # Add edges (relationships)
        for rel in relationships:
            if rel["source"] in self.graph and rel["target"] in self.graph:
                # Extract additional attributes (excluding already-used keys)
                extra_attrs = {k: v for k, v in rel.items() 
                              if k not in ["source", "target", "relation", "confidence"]}
                
                self.graph.add_edge(
                    rel["source"],
                    rel["target"],
                    relation=rel["relation"],
                    confidence=rel.get("confidence", 1.0),
                    **extra_attrs
                )
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Compute layout
        layout = self._compute_layout()
        
        # Compute graph metrics
        metrics = self._compute_metrics()
        
        # Prepare visualization data
        graph_data = self._prepare_visualization_data(layout, metrics, attention)
        
        return graph_data
    
    def _compute_layout(self) -> Dict[str, tuple]:
        """
        Compute graph layout using force-directed algorithm
        
        Returns:
            Dictionary mapping node IDs to (x, y) coordinates
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Use spring layout (force-directed)
        try:
            layout = nx.spring_layout(
                self.graph,
                k=2.0,  # Optimal distance between nodes
                iterations=50,
                seed=42  # For reproducibility
            )
        except:
            # Fallback to circular layout if spring fails
            layout = nx.circular_layout(self.graph)
        
        # Scale coordinates to reasonable range (0-1000)
        scaled_layout = {}
        for node, (x, y) in layout.items():
            scaled_layout[node] = (
                (x + 1) * 500,  # Scale from [-1, 1] to [0, 1000]
                (y + 1) * 500
            )
        
        return scaled_layout
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """
        Compute graph metrics for visualization
        
        Returns:
            Dictionary of graph metrics
        """
        if self.graph.number_of_nodes() == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0,
                "avg_degree": 0
            }
        
        # Basic metrics
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (for connected graphs)
        try:
            betweenness = nx.betweenness_centrality(self.graph)
        except:
            betweenness = {node: 0 for node in self.graph.nodes()}
        
        # PageRank
        try:
            pagerank = nx.pagerank(self.graph)
        except:
            pagerank = {node: 1.0/num_nodes for node in self.graph.nodes()}
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "avg_degree": sum(dict(self.graph.degree()).values()) / num_nodes if num_nodes > 0 else 0,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness,
            "pagerank": pagerank
        }
    
    def _prepare_visualization_data(
        self,
        layout: Dict[str, tuple],
        metrics: Dict[str, Any],
        attention: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Prepare graph data for D3.js visualization
        
        Args:
            layout: Node positions
            metrics: Graph metrics
            attention: Optional attention data for mapping
            
        Returns:
            Visualization-ready graph data
        """
        nodes = []
        edges = []
        
        # Prepare nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            x, y = layout.get(node_id, (500, 500))
            entity_type = node_data.get("entity_type", "UNKNOWN")
            
            nodes.append({
                "id": node_id,
                "label": node_data.get("label", str(node_id)),
                "entity_type": entity_type,
                "type": entity_type,  # Frontend expects 'type' field
                "x": float(x),  # Convert numpy float64 to Python float
                "y": float(y),
                "degree": int(self.graph.degree(node_id)),  # Convert to Python int
                "centrality": float(metrics.get("degree_centrality", {}).get(node_id, 0)),
                "pagerank": float(metrics.get("pagerank", {}).get(node_id, 0)),
                "start_token": node_data.get("start_token"),
                "end_token": node_data.get("end_token")
            })
        
        # Prepare edges
        for source, target in self.graph.edges():
            edge_data = self.graph.edges[source, target]
            
            edges.append({
                "id": edge_data.get("id", f"{source}_{target}"),
                "source": source,
                "target": target,
                "relation": edge_data.get("relation", "related"),
                "relation_text": edge_data.get("relation_text", ""),
                "confidence": edge_data.get("confidence", 1.0),
                "weight": edge_data.get("confidence", 1.0)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "num_nodes": metrics["num_nodes"],
                "num_edges": metrics["num_edges"],
                "density": metrics["density"],
                "avg_degree": metrics["avg_degree"]
            }
        }
    
    def get_subgraph(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Extract subgraph containing specified nodes
        
        Args:
            node_ids: List of node IDs to include
            
        Returns:
            Subgraph data
        """
        if not self.graph:
            return {"nodes": [], "edges": []}
        
        # Get subgraph
        subgraph = self.graph.subgraph(node_ids)
        
        # Compute layout for subgraph
        layout = self._compute_layout()
        metrics = self._compute_metrics()
        
        return self._prepare_visualization_data(layout, metrics)
