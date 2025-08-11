#!/usr/bin/env python3
"""
Script to extract common code patterns from repositories in each category
and create reusable components.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict

# Base directories
REPO_DIR = Path('repos')
FUNCTIONS_DIR = Path('functions')

# Common component templates for each category
COMPONENT_TEMPLATES = {
    'agent_frameworks': {
        'agent_base': """
class Agent:
    \"\"\"Base agent class that provides common functionality for all agents.\"\"\"
    
    def __init__(self, name, config=None):
        \"\"\"Initialize the agent with a name and optional configuration.\"\"\"
        self.name = name
        self.config = config or {}
        self.memory = []
        self.tools = []
        self.state = "idle"
    
    def add_tool(self, tool):
        \"\"\"Add a tool to the agent's toolkit.\"\"\"
        self.tools.append(tool)
    
    def add_to_memory(self, item):
        \"\"\"Add an item to the agent's memory.\"\"\"
        self.memory.append(item)
    
    def plan(self, goal):
        \"\"\"Create a plan to achieve a goal.\"\"\"
        # Implementation depends on specific agent architecture
        raise NotImplementedError("Subclasses must implement plan()")
    
    def execute(self, plan):
        \"\"\"Execute a plan.\"\"\"
        # Implementation depends on specific agent architecture
        raise NotImplementedError("Subclasses must implement execute()")
    
    def run(self, goal):
        \"\"\"Run the agent to achieve a goal.\"\"\"
        plan = self.plan(goal)
        return self.execute(plan)
""",
        'memory_system': """
class Memory:
    \"\"\"Memory system for agents to store and retrieve information.\"\"\"
    
    def __init__(self, capacity=100):
        \"\"\"Initialize the memory system with a capacity.\"\"\"
        self.capacity = capacity
        self.items = []
    
    def add(self, item):
        \"\"\"Add an item to memory.\"\"\"
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)  # Remove oldest item if capacity exceeded
    
    def get_all(self):
        \"\"\"Get all items in memory.\"\"\"
        return self.items
    
    def search(self, query):
        \"\"\"Search memory for items matching a query.\"\"\"
        # Simple string matching for now
        return [item for item in self.items if query in str(item)]
    
    def clear(self):
        \"\"\"Clear all items from memory.\"\"\"
        self.items = []
"""
    },
    'user_interfaces': {
        'chat_interface': """
class ChatInterface:
    \"\"\"Basic chat interface component.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the chat interface.\"\"\"
        self.messages = []
    
    def add_message(self, sender, content, timestamp=None):
        \"\"\"Add a message to the chat.\"\"\"
        import datetime
        timestamp = timestamp or datetime.datetime.now()
        message = {
            "sender": sender,
            "content": content,
            "timestamp": timestamp
        }
        self.messages.append(message)
        return message
    
    def get_messages(self, limit=None):
        \"\"\"Get recent messages, optionally limited to a certain number.\"\"\"
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def clear_messages(self):
        \"\"\"Clear all messages.\"\"\"
        self.messages = []
"""
    },
    'os_interaction': {
        'screen_capture': """
def capture_screen(region=None):
    \"\"\"
    Capture the screen or a region of the screen.
    
    Args:
        region (tuple, optional): Region to capture as (x, y, width, height).
            If None, captures the entire screen.
    
    Returns:
        image: The captured screen image.
    \"\"\"
    try:
        # Try to use PIL first
        from PIL import ImageGrab
        if region:
            return ImageGrab.grab(bbox=region)
        else:
            return ImageGrab.grab()
    except ImportError:
        try:
            # Fall back to pyautogui if PIL is not available
            import pyautogui
            if region:
                return pyautogui.screenshot(region=region)
            else:
                return pyautogui.screenshot()
        except ImportError:
            raise ImportError("Either PIL or pyautogui is required for screen capture")
""",
        'keyboard_control': """
def send_keys(text, interval=0.0):
    \"\"\"
    Send keystrokes to the active window.
    
    Args:
        text (str): The text to type.
        interval (float, optional): Interval between keystrokes in seconds.
    \"\"\"
    try:
        import pyautogui
        pyautogui.write(text, interval=interval)
    except ImportError:
        try:
            # Fall back to pynput if pyautogui is not available
            from pynput.keyboard import Controller
            keyboard = Controller()
            for char in text:
                keyboard.press(char)
                keyboard.release(char)
                if interval > 0:
                    import time
                    time.sleep(interval)
        except ImportError:
            raise ImportError("Either pyautogui or pynput is required for keyboard control")
"""
    },
    'browser_automation': {
        'browser_control': """
class BrowserControl:
    \"\"\"Basic browser control functionality.\"\"\"
    
    def __init__(self, headless=False):
        \"\"\"
        Initialize the browser controller.
        
        Args:
            headless (bool): Whether to run the browser in headless mode.
        \"\"\"
        self.headless = headless
        self.browser = None
    
    def start(self):
        \"\"\"Start the browser.\"\"\"
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            if self.headless:
                options.add_argument('--headless')
            
            self.browser = webdriver.Chrome(options=options)
            return True
        except ImportError:
            raise ImportError("Selenium is required for browser automation")
    
    def navigate(self, url):
        \"\"\"Navigate to a URL.\"\"\"
        if not self.browser:
            self.start()
        self.browser.get(url)
    
    def find_element(self, selector, by_type='css'):
        \"\"\"Find an element on the page.\"\"\"
        if not self.browser:
            raise RuntimeError("Browser not started")
        
        from selenium.webdriver.common.by import By
        
        by_map = {
            'css': By.CSS_SELECTOR,
            'id': By.ID,
            'name': By.NAME,
            'xpath': By.XPATH,
            'link_text': By.LINK_TEXT,
            'tag': By.TAG_NAME,
            'class': By.CLASS_NAME
        }
        
        by_type = by_type.lower()
        if by_type not in by_map:
            raise ValueError(f"Invalid selector type: {by_type}")
        
        return self.browser.find_element(by_map[by_type], selector)
    
    def close(self):
        \"\"\"Close the browser.\"\"\"
        if self.browser:
            self.browser.quit()
            self.browser = None
"""
    },
    'code_execution': {
        'code_executor': """
class CodeExecutor:
    \"\"\"Safe code execution environment.\"\"\"
    
    def __init__(self, timeout=30):
        \"\"\"
        Initialize the code executor.
        
        Args:
            timeout (int): Maximum execution time in seconds.
        \"\"\"
        self.timeout = timeout
        self.globals = {}
        self.locals = {}
    
    def execute(self, code):
        \"\"\"
        Execute Python code safely.
        
        Args:
            code (str): Python code to execute.
        
        Returns:
            dict: Execution results with stdout, stderr, and return value.
        \"\"\"
        import sys
        import io
        import traceback
        import threading
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        # Prepare result
        result = {
            'stdout': '',
            'stderr': '',
            'return_value': None,
            'error': None,
            'timed_out': False
        }
        
        # Create execution function
        def exec_code():
            sys_stdout = sys.stdout
            sys_stderr = sys.stderr
            
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                
                # Execute the code
                exec(code, self.globals, self.locals)
                
                # Get the last expression's value if it exists
                if '_' in self.locals:
                    result['return_value'] = self.locals['_']
            except Exception as e:
                result['error'] = str(e)
                result['stderr'] = traceback.format_exc()
            finally:
                sys.stdout = sys_stdout
                sys.stderr = sys_stderr
        
        # Execute with timeout
        thread = threading.Thread(target=exec_code)
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            # Timeout occurred
            result['timed_out'] = True
            result['stderr'] = f"Execution timed out after {self.timeout} seconds"
            
            # Try to terminate the thread (not guaranteed)
            try:
                import ctypes
                thread_id = thread.ident
                if thread_id:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread_id),
                        ctypes.py_object(SystemExit)
                    )
            except:
                pass
        else:
            # Execution completed
            result['stdout'] = stdout.getvalue()
            result['stderr'] = stderr.getvalue()
        
        return result
"""
    },
    'cognitive_systems': {
        'knowledge_graph': """
class KnowledgeGraph:
    \"\"\"Simple knowledge graph implementation.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize an empty knowledge graph.\"\"\"
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id, properties=None):
        \"\"\"
        Add a node to the knowledge graph.
        
        Args:
            node_id (str): Unique identifier for the node.
            properties (dict, optional): Node properties.
        
        Returns:
            bool: True if the node was added, False if it already existed.
        \"\"\"
        if node_id in self.nodes:
            return False
        
        self.nodes[node_id] = properties or {}
        return True
    
    def add_edge(self, source_id, target_id, relation_type, properties=None):
        \"\"\"
        Add an edge between two nodes.
        
        Args:
            source_id (str): Source node ID.
            target_id (str): Target node ID.
            relation_type (str): Type of relation.
            properties (dict, optional): Edge properties.
        
        Returns:
            bool: True if the edge was added, False if nodes don't exist.
        \"\"\"
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
        \"\"\"Get a node by ID.\"\"\"
        return self.nodes.get(node_id)
    
    def get_edges(self, source_id=None, target_id=None, relation_type=None):
        \"\"\"
        Get edges matching the specified criteria.
        
        Args:
            source_id (str, optional): Filter by source node ID.
            target_id (str, optional): Filter by target node ID.
            relation_type (str, optional): Filter by relation type.
        
        Returns:
            list: Matching edges.
        \"\"\"
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
        \"\"\"
        Query the knowledge graph using a custom function.
        
        Args:
            query_func (callable): Function that takes a node or edge and returns a boolean.
        
        Returns:
            dict: Dictionary with 'nodes' and 'edges' that match the query.
        \"\"\"
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
"""
    },
    'evolution_optimization': {
        'genetic_algorithm': """
class GeneticAlgorithm:
    \"\"\"Simple genetic algorithm implementation.\"\"\"
    
    def __init__(self, population_size=100, mutation_rate=0.01, crossover_rate=0.7):
        \"\"\"
        Initialize the genetic algorithm.
        
        Args:
            population_size (int): Size of the population.
            mutation_rate (float): Probability of mutation (0-1).
            crossover_rate (float): Probability of crossover (0-1).
        \"\"\"
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.generation = 0
    
    def initialize(self, create_individual):
        \"\"\"
        Initialize the population.
        
        Args:
            create_individual (callable): Function that creates a random individual.
        \"\"\"
        self.population = [create_individual() for _ in range(self.population_size)]
        self.generation = 0
    
    def evolve(self, fitness_func, selection_func, crossover_func, mutation_func, generations=1):
        \"\"\"
        Evolve the population for a number of generations.
        
        Args:
            fitness_func (callable): Function that evaluates an individual's fitness.
            selection_func (callable): Function that selects parents based on fitness.
            crossover_func (callable): Function that creates offspring from parents.
            mutation_func (callable): Function that mutates an individual.
            generations (int): Number of generations to evolve.
        
        Returns:
            tuple: Best individual and its fitness.
        \"\"\"
        import random
        
        for _ in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness_func(individual) for individual in self.population]
            
            # Find best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            best_individual = self.population[best_idx]
            best_fitness = fitness_scores[best_idx]
            
            # Create new population
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(best_individual)
            
            # Create the rest of the new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = selection_func(self.population, fitness_scores)
                parent2 = selection_func(self.population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = crossover_func(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = mutation_func(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = mutation_func(offspring2)
                
                # Add to new population
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Update population
            self.population = new_population
            self.generation += 1
        
        # Evaluate final fitness
        fitness_scores = [fitness_func(individual) for individual in self.population]
        best_idx = fitness_scores.index(max(fitness_scores))
        
        return self.population[best_idx], fitness_scores[best_idx]
"""
    },
    'integration': {
        'system_connector': """
class SystemConnector:
    \"\"\"Connector for integrating different systems.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the system connector.\"\"\"
        self.systems = {}
        self.connections = []
    
    def register_system(self, system_id, system_obj, system_type=None):
        \"\"\"
        Register a system with the connector.
        
        Args:
            system_id (str): Unique identifier for the system.
            system_obj (object): The system object.
            system_type (str, optional): Type of the system.
        
        Returns:
            bool: True if the system was registered, False if it already exists.
        \"\"\"
        if system_id in self.systems:
            return False
        
        self.systems[system_id] = {
            'object': system_obj,
            'type': system_type,
            'interfaces': {}
        }
        
        return True
    
    def register_interface(self, system_id, interface_id, interface_func):
        \"\"\"
        Register an interface for a system.
        
        Args:
            system_id (str): System identifier.
            interface_id (str): Interface identifier.
            interface_func (callable): Function implementing the interface.
        
        Returns:
            bool: True if the interface was registered, False otherwise.
        \"\"\"
        if system_id not in self.systems:
            return False
        
        self.systems[system_id]['interfaces'][interface_id] = interface_func
        return True
    
    def connect(self, source_system_id, source_interface_id, 
                target_system_id, target_interface_id, 
                transform_func=None):
        \"\"\"
        Connect two systems through their interfaces.
        
        Args:
            source_system_id (str): Source system identifier.
            source_interface_id (str): Source interface identifier.
            target_system_id (str): Target system identifier.
            target_interface_id (str): Target interface identifier.
            transform_func (callable, optional): Function to transform data between interfaces.
        
        Returns:
            bool: True if the connection was established, False otherwise.
        \"\"\"
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
        \"\"\"
        Send data from a source interface to all connected target interfaces.
        
        Args:
            source_system_id (str): Source system identifier.
            source_interface_id (str): Source interface identifier.
            data: Data to send.
        
        Returns:
            dict: Results from each target interface.
        \"\"\"
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
"""
    },
    'nlp': {
        'text_processor': """
class TextProcessor:
    \"\"\"Basic text processing functionality.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the text processor.\"\"\"
        self.stop_words = set([
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
            'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
            'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on', 'at', 'by', 'with'
        ])
    
    def tokenize(self, text):
        \"\"\"
        Tokenize text into words.
        
        Args:
            text (str): Text to tokenize.
        
        Returns:
            list: List of tokens.
        \"\"\"
        import re
        # Simple tokenization by splitting on non-alphanumeric characters
        return re.findall(r'\\w+', text.lower())
    
    def remove_stop_words(self, tokens):
        \"\"\"
        Remove stop words from a list of tokens.
        
        Args:
            tokens (list): List of tokens.
        
        Returns:
            list: Tokens with stop words removed.
        \"\"\"
        return [token for token in tokens if token not in self.stop_words]
    
    def extract_entities(self, text):
        \"\"\"
        Extract entities from text using simple pattern matching.
        
        Args:
            text (str): Text to analyze.
        
        Returns:
            dict: Dictionary of entity types and their values.
        \"\"\"
        import re
        
        entities = {
            'emails': re.findall(r'[\\w\\.-]+@[\\w\\.-]+', text),
            'urls': re.findall(r'https?://[\\w\\.-/]+', text),
            'phone_numbers': re.findall(r'\\+?\\d{1,3}[\\s-]?\\(?\\d{3}\\)?[\\s-]?\\d{3}[\\s-]?\\d{4}', text),
            'dates': re.findall(r'\\d{1,2}[/\\-]\\d{1,2}[/\\-]\\d{2,4}', text)
        }
        
        return entities
    
    def get_sentiment(self, text):
        \"\"\"
        Get sentiment of text using a simple lexicon-based approach.
        
        Args:
            text (str): Text to analyze.
        
        Returns:
            float: Sentiment score (-1 to 1).
        \"\"\"
        # Simple sentiment lexicon
        positive_words = {
            'good', 'great', 'excellent', 'positive', 'wonderful', 'fantastic',
            'amazing', 'love', 'happy', 'best', 'better', 'awesome', 'nice',
            'perfect', 'pleasant', 'enjoy', 'liked', 'like', 'joy'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'negative', 'horrible', 'worst',
            'worse', 'hate', 'sad', 'disappointing', 'poor', 'unpleasant',
            'dislike', 'disliked', 'unfortunate', 'unhappy', 'sorry'
        }
        
        # Tokenize and remove stop words
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        
        # Count positive and negative words
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        # Calculate sentiment score
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
"""
    }
}

def extract_common_components():
    """Extract common components for each category."""
    # Load the functionality index
    try:
        with open(FUNCTIONS_DIR / 'functionality_index.json', 'r') as f:
            functionality_index = json.load(f)
    except FileNotFoundError:
        print("Functionality index not found. Run analyze_functionality.py first.")
        return
    
    # Create common components for each category
    for category, template_dict in COMPONENT_TEMPLATES.items():
        category_dir = FUNCTIONS_DIR / category / 'common'
        category_dir.mkdir(exist_ok=True)
        
        # Create __init__.py to make it a proper package
        with open(category_dir / '__init__.py', 'w') as f:
            f.write(f"# Common components for {category}\n")
        
        # Create each component
        for component_name, template in template_dict.items():
            file_name = f"{component_name}.py"
            with open(category_dir / file_name, 'w') as f:
                f.write(f"# Common {component_name} component for {category}\n")
                f.write(template.strip())
                f.write("\n")
            
            print(f"Created common component: {category}/common/{file_name}")
    
    print("Common components extraction complete!")

if __name__ == "__main__":
    extract_common_components()