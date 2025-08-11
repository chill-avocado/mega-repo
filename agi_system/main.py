#!/usr/bin/env python3
"""
Main script for the AGI system.

This script demonstrates the AGI system by creating an instance and running it on a simple task.
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional

from agi_system.core.agi_system import AGISystem


def setup_logging():
    """Set up logging for the AGI system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_agi_system() -> AGISystem:
    """Create and initialize the AGI system."""
    # Create the AGI system
    config = {
        'cognitive_architecture': {
            'executive_function': {
                'planning_horizon': 10
            },
            'working_memory': {
                'capacity': 10
            },
            'long_term_memory': {
                'retrieval_threshold': 0.5
            },
            'attention': {
                'focus_strength': 0.8
            },
            'meta_cognition': {
                'monitoring_interval': 5
            }
        },
        'neural_symbolic': {
            'neural_model': {
                'model_type': 'transformer',
                'hidden_dim': 128
            },
            'symbolic_knowledge': {
                'facts': [
                    'sky is blue',
                    'grass is green',
                    'water is wet'
                ],
                'rules': [
                    {
                        'antecedent': 'sky is blue',
                        'consequent': 'it is daytime',
                        'confidence': 0.8
                    },
                    {
                        'antecedent': 'grass is green',
                        'consequent': 'plants are healthy',
                        'confidence': 0.7
                    }
                ],
                'ontology': {
                    'sky': {
                        'is_a': 'natural_object',
                        'color': 'blue',
                        'location': 'above'
                    },
                    'grass': {
                        'is_a': 'plant',
                        'color': 'green',
                        'location': 'ground'
                    }
                }
            }
        },
        'predictive_processing': {
            'hierarchy': [
                {
                    'input_dim': 10,
                    'hidden_dim': 20,
                    'output_dim': 10
                },
                {
                    'input_dim': 10,
                    'hidden_dim': 30,
                    'output_dim': 10
                },
                {
                    'input_dim': 10,
                    'hidden_dim': 40,
                    'output_dim': 10
                }
            ]
        },
        'self_improvement': {
            'code_improvement': {
                'modules': [
                    'agi_system.core.agi_system',
                    'agi_system.core.cognitive_architecture'
                ]
            },
            'model_improvement': {
                'models': [
                    'neural_model',
                    'predictive_model'
                ]
            },
            'knowledge_improvement': {
                'seed_knowledge': {
                    'facts': [
                        'AGI is artificial general intelligence',
                        'learning is important for intelligence',
                        'reasoning is important for intelligence'
                    ],
                    'rules': [
                        {
                            'antecedent': 'learning is important for intelligence',
                            'consequent': 'AGI must learn',
                            'confidence': 0.9
                        },
                        {
                            'antecedent': 'reasoning is important for intelligence',
                            'consequent': 'AGI must reason',
                            'confidence': 0.9
                        }
                    ],
                    'concepts': {
                        'AGI': {
                            'is_a': 'artificial_intelligence',
                            'capability': 'general',
                            'goal': 'human_level_intelligence'
                        },
                        'learning': {
                            'is_a': 'cognitive_process',
                            'purpose': 'acquire_knowledge',
                            'types': ['supervised', 'unsupervised', 'reinforcement']
                        },
                        'reasoning': {
                            'is_a': 'cognitive_process',
                            'purpose': 'draw_conclusions',
                            'types': ['deductive', 'inductive', 'abductive']
                        }
                    }
                }
            }
        }
    }
    
    agi_system = AGISystem(config)
    
    # Initialize the AGI system with capabilities
    capabilities = [
        'language',
        'reasoning',
        'learning',
        'planning',
        'self_model',
        'causal',
        'meta_learning',
        'self_improvement'
    ]
    
    agi_system.initialize(capabilities)
    
    return agi_system


def run_example_task(agi_system: AGISystem):
    """Run an example task using the AGI system."""
    # Set a goal for the AGI system
    goal = "Understand the concept of artificial general intelligence and explain its key components"
    agi_system.set_goal(goal)
    
    # Run the AGI system
    print(f"\nRunning AGI system with goal: {goal}")
    print("=" * 80)
    
    start_time = time.time()
    results = agi_system.run(max_cycles=50)
    duration = time.time() - start_time
    
    print(f"\nAGI system completed in {duration:.2f} seconds")
    print("=" * 80)
    
    # Print results
    if results['success']:
        print(f"Goal achieved: {results['goal_achieved']}")
        print(f"Cycles run: {results['cycles_run']}")
        
        # Get the final state
        state = agi_system.get_state()
        
        # Print capabilities
        print("\nCapabilities:")
        for capability in state['capabilities']:
            print(f"- {capability}")
        
        # Print cognitive architecture state
        print("\nCognitive Architecture State:")
        cognitive_state = state['cognitive_architecture']
        print(f"- Cognitive cycles: {cognitive_state['cognitive_cycle_count']}")
        print(f"- Current goal: {cognitive_state['current_goal']}")
        
        # Print working memory contents
        if 'components' in cognitive_state and 'working_memory' in cognitive_state['components']:
            working_memory = cognitive_state['components']['working_memory']
            if 'memory_contents' in working_memory:
                print("\nWorking Memory Contents:")
                for key, value in working_memory['memory_contents'].items():
                    print(f"- {key}: {value}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")


def demonstrate_self_improvement(agi_system: AGISystem):
    """Demonstrate the self-improvement capability of the AGI system."""
    print("\nDemonstrating self-improvement capability")
    print("=" * 80)
    
    # Improve the AGI system
    start_time = time.time()
    improvement_results = agi_system.improve()
    duration = time.time() - start_time
    
    print(f"\nSelf-improvement completed in {duration:.2f} seconds")
    print("=" * 80)
    
    # Print improvement results
    if improvement_results['success']:
        # Print code improvement results
        if 'code_improvement' in improvement_results:
            code_results = improvement_results['code_improvement']
            print("\nCode Improvement Results:")
            if code_results['success']:
                print(f"Modules improved: {code_results.get('modules_improved', 0)}")
                for module, result in code_results.get('results', {}).items():
                    if result['success']:
                        print(f"- {module}: {len(result.get('improvements', []))} improvements")
                    else:
                        print(f"- {module}: Error - {result.get('error', 'Unknown error')}")
            else:
                print(f"Error: {code_results.get('error', 'Unknown error')}")
        
        # Print model improvement results
        if 'model_improvement' in improvement_results:
            model_results = improvement_results['model_improvement']
            print("\nModel Improvement Results:")
            if model_results['success']:
                print(f"Models improved: {model_results.get('models_improved', 0)}")
                for model, result in model_results.get('results', {}).items():
                    if result['success']:
                        print(f"- {model}: {len(result.get('improvements', []))} improvements")
                    else:
                        print(f"- {model}: Error - {result.get('error', 'Unknown error')}")
            else:
                print(f"Error: {model_results.get('error', 'Unknown error')}")
        
        # Print knowledge improvement results
        if 'knowledge_improvement' in improvement_results:
            knowledge_results = improvement_results['knowledge_improvement']
            print("\nKnowledge Improvement Results:")
            if knowledge_results['success']:
                print(f"Improvements made: {knowledge_results.get('improvements_made', 0)}")
                
                # Print knowledge statistics
                if 'knowledge_stats' in knowledge_results:
                    stats = knowledge_results['knowledge_stats']
                    print(f"- Facts: {stats['facts']}")
                    print(f"- Rules: {stats['rules']}")
                    print(f"- Concepts: {stats['concepts']}")
                
                # Print improvement details
                for improvement_type, result in knowledge_results.get('results', {}).items():
                    if result['success']:
                        print(f"- {improvement_type}: {len(result.get('improvements', []))} improvements")
                    else:
                        print(f"- {improvement_type}: Error - {result.get('error', 'Unknown error')}")
            else:
                print(f"Error: {knowledge_results.get('error', 'Unknown error')}")
    else:
        print(f"Error: {improvement_results.get('error', 'Unknown error')}")


def demonstrate_neural_symbolic_integration(agi_system: AGISystem):
    """Demonstrate the neural-symbolic integration capability of the AGI system."""
    print("\nDemonstrating neural-symbolic integration capability")
    print("=" * 80)
    
    # Get the neural-symbolic component
    components = agi_system.cognitive_architecture.registry.get_all()
    neural_symbolic = components.get('neural_symbolic')
    
    if not neural_symbolic:
        print("Neural-symbolic component not found")
        return
    
    # Demonstrate reasoning
    query = {
        'symbolic_query': 'sky is blue',
        'neural_query': {'text': 'What color is the sky?'},
        'integrated_query': {
            'text': 'What color is the sky?',
            'symbolic': 'sky is blue'
        }
    }
    
    print("\nPerforming reasoning with neural-symbolic integration")
    reasoning_results = neural_symbolic.reason(query)
    
    if reasoning_results['success']:
        print("\nReasoning Results:")
        
        # Print symbolic reasoning results
        if 'symbolic' in reasoning_results['results']:
            symbolic_results = reasoning_results['results']['symbolic']
            print("\nSymbolic Reasoning:")
            for result in symbolic_results:
                if 'fact' in result:
                    print(f"- Fact: {result['fact']} (confidence: {result['confidence']})")
                elif 'rule' in result:
                    print(f"- Rule: {result['rule']['antecedent']} -> {result['rule']['consequent']} (confidence: {result['confidence']})")
        
        # Print neural reasoning results
        if 'neural' in reasoning_results['results']:
            neural_results = reasoning_results['results']['neural']
            print("\nNeural Reasoning:")
            print(f"- Predictions: {neural_results['predictions']}")
        
        # Print integrated reasoning results
        if 'integrated' in reasoning_results['results']:
            integrated_results = reasoning_results['results']['integrated']
            print("\nIntegrated Reasoning:")
            print(f"- Confidence: {integrated_results['confidence']}")
            print(f"- Neural embedding: {integrated_results['neural_embedding'][:3]}... (truncated)")
            print(f"- Symbolic results count: {len(integrated_results['symbolic_results'])}")
    else:
        print(f"Error: {reasoning_results.get('error', 'Unknown error')}")
    
    # Demonstrate generation
    spec = {
        'neural_spec': {'prompt': 'Explain what AGI is'},
        'symbolic_spec': {'query': 'AGI is artificial general intelligence'},
        'integrated_spec': {
            'prompt': 'Explain what AGI is',
            'query': 'AGI is artificial general intelligence'
        }
    }
    
    print("\nGenerating content with neural-symbolic integration")
    generation_results = neural_symbolic.generate(spec)
    
    if generation_results['success']:
        print("\nGeneration Results:")
        
        # Print neural generation results
        if 'neural' in generation_results['results']:
            neural_results = generation_results['results']['neural']
            print("\nNeural Generation:")
            print(f"- Generated text: {neural_results.get('generated_text', 'None')}")
        
        # Print symbolic generation results
        if 'symbolic' in generation_results['results']:
            symbolic_results = generation_results['results']['symbolic']
            print("\nSymbolic Generation:")
            print(f"- Generated text: {symbolic_results.get('generated_text', 'None')}")
        
        # Print integrated generation results
        if 'integrated' in generation_results['results']:
            integrated_results = generation_results['results']['integrated']
            print("\nIntegrated Generation:")
            print(f"- Integrated text: {integrated_results.get('integrated_text', 'None')}")
    else:
        print(f"Error: {generation_results.get('error', 'Unknown error')}")


def demonstrate_predictive_processing(agi_system: AGISystem):
    """Demonstrate the hierarchical predictive processing capability of the AGI system."""
    print("\nDemonstrating hierarchical predictive processing capability")
    print("=" * 80)
    
    # Get the predictive processing component
    components = agi_system.cognitive_architecture.registry.get_all()
    predictive_processing = components.get('predictive_processing')
    
    if not predictive_processing:
        print("Predictive processing component not found")
        return
    
    # Generate random input data
    import random
    input_data = [random.random() for _ in range(10)]
    
    print("\nProcessing input through hierarchical predictive processing")
    processing_results = predictive_processing.process(input_data)
    
    if processing_results['success']:
        print("\nProcessing Results:")
        
        # Print bottom-up results
        print("\nBottom-up Pass:")
        for result in processing_results['bottom_up_results']:
            print(f"- Level {result['level']}: Predictions: {result['predictions'][:3]}... (truncated)")
        
        # Print top-down results
        print("\nTop-down Pass:")
        for result in processing_results['top_down_results']:
            print(f"- Level {result['level']}: Prediction error: {result['prediction_error']:.6f}")
        
        # Print free energy and attention weights
        print("\nFree Energy (Prediction Error):")
        for i, fe in enumerate(processing_results['free_energy']):
            print(f"- Level {i}: {fe:.6f}")
        
        print("\nAttention Weights:")
        for i, weight in enumerate(processing_results['attention_weights']):
            print(f"- Level {i}: {weight:.6f}")
    else:
        print(f"Error: {processing_results.get('error', 'Unknown error')}")
    
    # Generate a sequence
    print("\nGenerating sequence using hierarchical predictive processing")
    generation_results = predictive_processing.generate(level=1, steps=5)
    
    if generation_results['success']:
        print("\nGeneration Results:")
        print(f"- Level: {generation_results['level']}")
        print(f"- Steps: {generation_results['steps']}")
        print("\nGenerated Sequence:")
        for i, step in enumerate(generation_results['generated_sequence']):
            print(f"- Step {i}: {step[:3]}... (truncated)")
    else:
        print(f"Error: {generation_results.get('error', 'Unknown error')}")
    
    # Get abstractions
    print("\nGetting abstractions from hierarchical predictive processing")
    abstraction_results = predictive_processing.get_abstractions(level=2)
    
    if abstraction_results['success']:
        print("\nAbstraction Results:")
        print(f"- Level: {abstraction_results['level']}")
        print("\nAbstractions:")
        abstractions = abstraction_results['abstractions']
        print(f"- Input weights shape: {len(abstractions['weights_input'])}x{len(abstractions['weights_input'][0])}")
        print(f"- Output weights shape: {len(abstractions['weights_output'])}x{len(abstractions['weights_output'][0])}")
        print(f"- Hidden bias length: {len(abstractions['bias_hidden'])}")
        print(f"- Output bias length: {len(abstractions['bias_output'])}")
    else:
        print(f"Error: {abstraction_results.get('error', 'Unknown error')}")


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    
    # Create the AGI system
    print("Creating AGI system...")
    agi_system = create_agi_system()
    
    # Run an example task
    run_example_task(agi_system)
    
    # Demonstrate self-improvement
    demonstrate_self_improvement(agi_system)
    
    # Demonstrate neural-symbolic integration
    demonstrate_neural_symbolic_integration(agi_system)
    
    # Demonstrate hierarchical predictive processing
    demonstrate_predictive_processing(agi_system)
    
    print("\nAGI system demonstration completed")


if __name__ == "__main__":
    main()