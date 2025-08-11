# Common system_connector component for integration
class SystemConnector:
    """Connector for integrating different systems."""
    
    def __init__(self):
        """Initialize the system connector."""
        self.systems = {}
        self.connections = []
    
    def register_system(self, system_id, system_obj, system_type=None):
        """
        Register a system with the connector.
        
        Args:
            system_id (str): Unique identifier for the system.
            system_obj (object): The system object.
            system_type (str, optional): Type of the system.
        
        Returns:
            bool: True if the system was registered, False if it already exists.
        """
        if system_id in self.systems:
            return False
        
        self.systems[system_id] = {
            'object': system_obj,
            'type': system_type,
            'interfaces': {}
        }
        
        return True
    
    def register_interface(self, system_id, interface_id, interface_func):
        """
        Register an interface for a system.
        
        Args:
            system_id (str): System identifier.
            interface_id (str): Interface identifier.
            interface_func (callable): Function implementing the interface.
        
        Returns:
            bool: True if the interface was registered, False otherwise.
        """
        if system_id not in self.systems:
            return False
        
        self.systems[system_id]['interfaces'][interface_id] = interface_func
        return True
    
    def connect(self, source_system_id, source_interface_id, 
                target_system_id, target_interface_id, 
                transform_func=None):
        """
        Connect two systems through their interfaces.
        
        Args:
            source_system_id (str): Source system identifier.
            source_interface_id (str): Source interface identifier.
            target_system_id (str): Target system identifier.
            target_interface_id (str): Target interface identifier.
            transform_func (callable, optional): Function to transform data between interfaces.
        
        Returns:
            bool: True if the connection was established, False otherwise.
        """
        # Check if systems and interfaces exist
        if (source_system_id not in self.systems or 
            target_system_id not in self.systems or
            source_interface_id not in self.systems[source_system_id]['interfaces'] or
            target_interface_id not in self.systems[target_system_id]['interfaces']):
            return False
        
        # Create connection
        connection = {
            'source_system': source_system_id,
            'source_interface': source_interface_id,
            'target_system': target_system_id,
            'target_interface': target_interface_id,
            'transform': transform_func
        }
        
        self.connections.append(connection)
        return True
    
    def send(self, source_system_id, source_interface_id, data):
        """
        Send data from a source interface to all connected target interfaces.
        
        Args:
            source_system_id (str): Source system identifier.
            source_interface_id (str): Source interface identifier.
            data: Data to send.
        
        Returns:
            dict: Results from each target interface.
        """
        results = {}
        
        # Find all connections from this source
        for connection in self.connections:
            if (connection['source_system'] == source_system_id and 
                connection['source_interface'] == source_interface_id):
                
                # Get target system and interface
                target_system_id = connection['target_system']
                target_interface_id = connection['target_interface']
                target_system = self.systems[target_system_id]
                target_interface = target_system['interfaces'][target_interface_id]
                
                # Transform data if needed
                transformed_data = data
                if connection['transform']:
                    transformed_data = connection['transform'](data)
                
                # Send data to target interface
                try:
                    result = target_interface(transformed_data)
                    results[f"{target_system_id}.{target_interface_id}"] = result
                except Exception as e:
                    results[f"{target_system_id}.{target_interface_id}"] = {
                        'error': str(e)
                    }
        
        return results
