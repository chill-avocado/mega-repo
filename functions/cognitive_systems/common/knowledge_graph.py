# Common knowledge_graph component for cognitive_systems
class KnowledgeGraph:
    """Simple knowledge graph implementation."""
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id, properties=None):
        """
        Add a node to the knowledge graph.
        
        Args:
            node_id (str): Unique identifier for the node.
            properties (dict, optional): Node properties.
        
        Returns:
            bool: True if the node was added, False if it already existed.
        """
        if node_id in self.nodes:
            return False
        
        self.nodes[node_id] = properties or {}
        return True
    
    def add_edge(self, source_id, target_id, relation_type, properties=None):
        """
        Add an edge between two nodes.
        
        Args:
            source_id (str): Source node ID.
            target_id (str): Target node ID.
            relation_type (str): Type of relation.
            properties (dict, optional): Edge properties.
        
        Returns:
            bool: True if the edge was added, False if nodes don't exist.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        
        edge = {
            'source': source_id,
            'target': target_id,
            'relation': relation_type,
            'properties': properties or {}
        }
        
        self.edges.append(edge)
        return True
    
    def get_node(self, node_id):
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edges(self, source_id=None, target_id=None, relation_type=None):
        """
        Get edges matching the specified criteria.
        
        Args:
            source_id (str, optional): Filter by source node ID.
            target_id (str, optional): Filter by target node ID.
            relation_type (str, optional): Filter by relation type.
        
        Returns:
            list: Matching edges.
        """
        result = []
        
        for edge in self.edges:
            if source_id and edge['source'] != source_id:
                continue
            if target_id and edge['target'] != target_id:
                continue
            if relation_type and edge['relation'] != relation_type:
                continue
            
            result.append(edge)
        
        return result
    
    def query(self, query_func):
        """
        Query the knowledge graph using a custom function.
        
        Args:
            query_func (callable): Function that takes a node or edge and returns a boolean.
        
        Returns:
            dict: Dictionary with 'nodes' and 'edges' that match the query.
        """
        matching_nodes = {}
        matching_edges = []
        
        # Query nodes
        for node_id, properties in self.nodes.items():
            node = {'id': node_id, 'properties': properties}
            if query_func(node):
                matching_nodes[node_id] = properties
        
        # Query edges
        for edge in self.edges:
            if query_func(edge):
                matching_edges.append(edge)
        
        return {
            'nodes': matching_nodes,
            'edges': matching_edges
        }
