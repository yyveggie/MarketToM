# -*- coding: utf-8 -*-

import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not installed. Run 'pip install graphviz' to enable graph generation.")

@dataclass
class MentalState:
    """Mental state data class"""
    belief: str
    intent: str 
    emotion: str
    timestamp: str
    is_corrected: bool = False
    original_state: Optional['MentalState'] = None

@dataclass 
class StrategyUpdate:
    """Strategy update data class"""
    level: str  # belief, intent, emotion
    decision_type: str  # CREATE, MODIFY
    original_id: Optional[str]
    content: str
    timestamp: str

@dataclass
class BackwardInferenceResult:
    """Backward inference result data class"""
    timestamp: str
    predicted_action: str
    actual_action: str
    strategy_updates: List[StrategyUpdate]
    original_inference_timestamp: str

@dataclass
class InferenceStep:
    """Inference step data class"""
    timestamp: str
    environmental_state: str
    mental_states: MentalState
    strategies_used: Dict[str, List]
    backward_inference: Optional[BackwardInferenceResult] = None

class MentalStateVisualizer:
    """Mental state visualizer"""
    
    def __init__(self, storage_dir: str = "./storage"):
        """
        Initialize visualizer
        
        Args:
            storage_dir: Storage directory path
        """
        self.storage_dir = storage_dir
        self.inference_logs_dir = os.path.join(storage_dir, "inference_logs")
        self.strategy_database_dir = os.path.join(storage_dir, "strategy_database")
        self.output_dir = os.path.join(storage_dir, "visualizations")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not GRAPHVIZ_AVAILABLE:
            self.logger.warning("Graphviz unavailable, will skip graph generation")

    def load_inference_log(self, log_file: str) -> Optional[InferenceStep]:
        """
        Load inference log file
        
        Args:
            log_file: Log file path
            
        Returns:
            InferenceStep object or None
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Parse mental states
            mental_states = MentalState(
                belief=data['mental_states']['belief'],
                intent=data['mental_states']['intent'],
                emotion=data['mental_states']['emotion'],
                timestamp=data['timestamp']
            )
            
            # Create inference step
            step = InferenceStep(
                timestamp=data['timestamp'],
                environmental_state=data['environmental_state'],
                mental_states=mental_states,
                strategies_used=data.get('strategies_used', {})
            )
            
            return step
            
        except Exception as e:
            self.logger.error(f"Failed to load inference log {log_file}: {e}")
            return None

    def load_all_inference_logs(self) -> List[InferenceStep]:
        """
        Load all inference logs
        
        Returns:
            List of inference steps
        """
        steps = []
        
        if not os.path.exists(self.inference_logs_dir):
            self.logger.warning(f"Inference logs directory does not exist: {self.inference_logs_dir}")
            return steps
        
        # Get all JSON files
        log_files = [f for f in os.listdir(self.inference_logs_dir) 
                    if f.endswith('.json')]
        log_files.sort()  # Sort by timestamp
        
        self.logger.info(f"Found {len(log_files)} inference log files")
        
        for log_file in log_files:
            file_path = os.path.join(self.inference_logs_dir, log_file)
            step = self.load_inference_log(file_path)
            if step:
                steps.append(step)
        
        return steps

    def load_strategy_database(self) -> Dict[str, List[Dict]]:
        """
        Load strategy database
        
        Returns:
            Strategy database dictionary
        """
        strategies = {"belief": [], "emotion": [], "intent": []}
        
        for strategy_type in strategies.keys():
            strategy_file = os.path.join(self.strategy_database_dir, f"{strategy_type}_strategies.json")
            
            if os.path.exists(strategy_file):
                try:
                    with open(strategy_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Handle different data formats
                        if isinstance(data, dict):
                            strategies[strategy_type] = data.get("strategies", [])
                        elif isinstance(data, list):
                            strategies[strategy_type] = data
                        else:
                            strategies[strategy_type] = []
                    
                    self.logger.info(f"Loaded {len(strategies[strategy_type])} {strategy_type} strategies")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load strategy file {strategy_file}: {e}")
            else:
                self.logger.warning(f"Strategy file does not exist: {strategy_file}")
        
        return strategies

    def create_strategy_evolution_graph(self, strategy_type: str = "belief") -> Optional[str]:
        """
        Create strategy evolution graph
        
        Args:
            strategy_type: Strategy type (belief, intent, emotion)
            
        Returns:
            Generated graph file path or None
        """
        if not GRAPHVIZ_AVAILABLE:
            return None
        
        strategies = self.load_strategy_database()
        strategy_list = strategies.get(strategy_type, [])
        
        if not strategy_list:
            self.logger.warning(f"No {strategy_type} strategy data found")
            return None
        
        # Sort strategies by time
        strategy_list.sort(key=lambda x: x.get("timestamp", ""))
        
        # Create directed graph
        dot = graphviz.Digraph(comment=f'{strategy_type} strategy evolution', format='png')
        dot.attr(rankdir='TB', size='12,10')
        dot.attr('node', shape='box', style='rounded,filled')
        
        # Set graph attributes
        dot.attr(label=f'MarketToM {strategy_type.upper()} Strategy Evolution\\nTotal Strategies: {len(strategy_list)}', 
                labelloc='t', fontsize='16')
        
        # Create nodes for each strategy version
        for i, strategy in enumerate(strategy_list):
            strategy_id = strategy.get("id", f"strategy_{i}")
            content = strategy.get("strategy", "")
            timestamp = strategy.get("timestamp", "")
            version = strategy.get("version", 1)
            
            # Extract complete strategy content
            content_short = self.format_full_text(content, 400, line_width=35)
            
            # Determine color (based on version number)
            if version == 1:
                color = 'lightgreen'  # Original strategy
            else:
                color = 'lightyellow'  # Modified strategy
            
            # Create node
            dot.node(strategy_id, 
                    f'{strategy_type.upper()} v{version}\\n{content_short}\\n{timestamp[:10]}',
                    fillcolor=color)
            
            # Connect to previous strategy (if exists)
            if i > 0:
                prev_strategy = strategy_list[i-1]
                prev_id = prev_strategy.get("id", f"strategy_{i-1}")
                dot.edge(prev_id, strategy_id, 'Evolve')
        
        # Generate file
        output_path = os.path.join(self.output_dir, f'{strategy_type}_strategy_evolution')
        
        try:
            dot.render(output_path, cleanup=True)
            self.logger.info(f"Generated strategy evolution graph: {output_path}.png")
            return f"{output_path}.png"
        except Exception as e:
            self.logger.error(f"Failed to generate graph: {e}")
            return None

    def create_backward_inference_graph(self, step: InferenceStep) -> Optional[str]:
        """
        Create mental state graph with backward inference
        
        Args:
            step: Inference step containing backward inference results
            
        Returns:
            Generated graph file path or None
        """
        if not GRAPHVIZ_AVAILABLE or not step.backward_inference:
            return None
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Backward inference correction graph', format='png')
        dot.attr(rankdir='TB', size='14,10')
        dot.attr('node', shape='box', style='rounded,filled')
        
        # Set graph attributes
        backward = step.backward_inference
        dot.attr(label=f'MarketToM Backward Inference Correction Flow\\n' +
                      f'Predicted: {backward.predicted_action} → Actual: {backward.actual_action}\\n' +
                      f'Time: {step.timestamp}', 
                labelloc='t', fontsize='16')
        
        # Original inference flow nodes
        env_text = self.format_full_text(step.environmental_state, 300, line_width=50)
        dot.node('env', f'Environmental State\\n{env_text}', 
                fillcolor='lightblue')
        
        belief_text = self.format_full_text(step.mental_states.belief, 400, line_width=40)
        intent_text = self.format_full_text(step.mental_states.intent, 400, line_width=40)
        emotion_text = self.format_full_text(step.mental_states.emotion, 400, line_width=40)
        
        dot.node('belief', f'Belief State\\n{belief_text}', 
                fillcolor='lightgreen')
        dot.node('intent', f'Intention State\\n{intent_text}', 
                fillcolor='lightyellow')
        dot.node('emotion', f'Emotion State\\n{emotion_text}', 
                fillcolor='lightcoral')
        
        # Predicted and actual actions
        dot.node('predicted_action', f'Predicted Action\\n{backward.predicted_action}', 
                fillcolor='lightgray')
        dot.node('actual_action', f'Actual Action\\n{backward.actual_action}', 
                fillcolor='lightcyan')
        
        # Original causal relationship edges
        dot.edge('env', 'belief', 'Influence')
        dot.edge('belief', 'intent', 'Lead to')
        dot.edge('env', 'emotion', 'Co-influence')
        dot.edge('belief', 'emotion', '')
        dot.edge('intent', 'predicted_action', 'Drive')
        dot.edge('emotion', 'predicted_action', '')
        
        # Error indication
        dot.edge('predicted_action', 'actual_action', 'Error', 
                color='red', style='dashed')
        
        # Strategy update nodes
        strategy_updates = backward.strategy_updates
        for i, update in enumerate(strategy_updates):
            update_id = f'update_{i}'
            update_text = self.format_full_text(update.content, 300, line_width=35)
            
            color = 'orange' if update.decision_type == 'MODIFY' else 'lightpink'
            
            dot.node(update_id, 
                    f'{update.decision_type}\\n{update.level.upper()} Strategy\\n{update_text}',
                    fillcolor=color)
            
            # Connect to corresponding mental state
            if update.level == 'belief':
                dot.edge('belief', update_id, 'Strategy Update', 
                        color='blue', style='dashed')
            elif update.level == 'intent':
                dot.edge('intent', update_id, 'Strategy Update', 
                        color='blue', style='dashed')
            elif update.level == 'emotion':
                dot.edge('emotion', update_id, 'Strategy Update', 
                        color='blue', style='dashed')
        
        # Generate file
        timestamp_str = step.timestamp.replace(':', '-').replace('.', '-')
        output_path = os.path.join(self.output_dir, f'backward_inference_{timestamp_str}')
        
        try:
            dot.render(output_path, cleanup=True)
            self.logger.info(f"Generated backward inference graph: {output_path}.png")
            return f"{output_path}.png"
        except Exception as e:
            self.logger.error(f"Failed to generate graph: {e}")
            return None

    def format_full_text(self, text: str, max_length: int = 500, line_width: int = 50) -> str:
        """
        Format complete text with smart line wrapping for display in graphs
        
        Args:
            text: Original text to display
            max_length: Maximum display length (will truncate at sentence boundary if possible)
            line_width: Maximum characters per line for wrapping
            
        Returns:
            Complete formatted text with line breaks
        """
        if not text:
            return ""
        
        # Remove extra spaces and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Truncate if too long, but keep much more content
        if len(text) > max_length:
            # Find a good truncation point (end of sentence)
            truncated = text[:max_length-3]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.7:  # If we can find a sentence end in the latter part
                text = text[:last_period+1] + "..."
            else:
                text = truncated + "..."
        
        # Add smart line wrapping
        return self._wrap_text(text, line_width)
    
    def _wrap_text(self, text: str, line_width: int = 50) -> str:
        """
        Wrap text to specified line width with smart word breaking
        
        Args:
            text: Text to wrap
            line_width: Maximum characters per line
            
        Returns:
            Text with line breaks
        """
        if len(text) <= line_width:
            return text
        
        words = text.split(' ')
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # If adding this word would exceed line width
            if current_length + len(word) + 1 > line_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\\n'.join(lines)

    def create_mental_state_graph(self, step: InferenceStep, show_details: bool = True) -> Optional[str]:
        """
        Create mental state graph for single inference step
        
        Args:
            step: Inference step
            show_details: Whether to show detailed information
            
        Returns:
            Generated graph file path or None
        """
        if not GRAPHVIZ_AVAILABLE:
            self.logger.warning("Graphviz unavailable, cannot generate graph")
            return None
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Mental state inference graph', format='png')
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='box', style='rounded,filled')
        
        # Set graph attributes
        dot.attr(label=f'MarketToM Mental State Inference Flow\\nTime: {step.timestamp}', 
                labelloc='t', fontsize='16')
        
        # Environmental state node
        env_text = self.format_full_text(step.environmental_state, 500, line_width=50)
        dot.node('env', f'Environmental State\\n{env_text}', 
                fillcolor='lightblue')
        
        # Mental state nodes
        belief_text = self.format_full_text(step.mental_states.belief, 600, line_width=40)
        intent_text = self.format_full_text(step.mental_states.intent, 600, line_width=40)
        emotion_text = self.format_full_text(step.mental_states.emotion, 600, line_width=40)
        
        dot.node('belief', f'Belief State\\n{belief_text}', 
                fillcolor='lightgreen')
        dot.node('intent', f'Intention State\\n{intent_text}', 
                fillcolor='lightyellow')
        dot.node('emotion', f'Emotion State\\n{emotion_text}', 
                fillcolor='lightcoral')
        
        # Action node (prediction)
        dot.node('action', 'Action\\n[To be predicted]', 
                fillcolor='lightgray')
        
        # Add causal relationship edges
        dot.edge('env', 'belief', 'Influence')
        dot.edge('belief', 'intent', 'Lead to')
        dot.edge('env', 'emotion', 'Co-influence')
        dot.edge('belief', 'emotion', '')
        dot.edge('intent', 'action', 'Drive')
        dot.edge('emotion', 'action', '')
        
        # If backward inference result exists, add correction node
        if step.backward_inference:
            dot.node('corrected', 'Strategy Correction', fillcolor='orange')
            dot.edge('action', 'corrected', 'Backward Learning', style='dashed')
        
        # Generate file
        timestamp_str = step.timestamp.replace(':', '-').replace('.', '-')
        output_path = os.path.join(self.output_dir, f'mental_state_{timestamp_str}')
        
        try:
            dot.render(output_path, cleanup=True)
            self.logger.info(f"Generated mental state graph: {output_path}.png")
            return f"{output_path}.png"
        except Exception as e:
            self.logger.error(f"Failed to generate graph: {e}")
            return None

    def create_causal_network_graph(self) -> Optional[str]:
        """
        Create causal Bayesian network structure graph
        
        Returns:
            Generated graph file path or None
        """
        if not GRAPHVIZ_AVAILABLE:
            self.logger.warning("Graphviz unavailable, cannot generate graph")
            return None
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Causal Bayesian network', format='png')
        dot.attr(rankdir='TB', size='10,8')
        dot.attr('node', shape='ellipse', style='filled')
        
        # Set graph attributes
        dot.attr(label='MarketToM Causal Bayesian Network (CBN)', 
                labelloc='t', fontsize='16')
        
        # Define nodes
        dot.node('env', 'Environmental State', 
                fillcolor='lightblue')
        dot.node('belief', 'Belief', 
                fillcolor='lightgreen')
        dot.node('intent', 'Intention', 
                fillcolor='lightyellow')
        dot.node('emotion', 'Emotion', 
                fillcolor='lightcoral')
        dot.node('action', 'Action', 
                fillcolor='lightgray')
        
        # Define causal relationships
        dot.edge('env', 'belief', label='Causal Influence')
        dot.edge('belief', 'intent', label='Determine')
        dot.edge('env', 'emotion', label='Co-influence')
        dot.edge('belief', 'emotion')
        dot.edge('intent', 'action', label='Drive')
        dot.edge('emotion', 'action', label='Modulate')
        
        # Generate file
        output_path = os.path.join(self.output_dir, 'causal_bayesian_network')
        
        try:
            dot.render(output_path, cleanup=True)
            self.logger.info(f"Generated causal network graph: {output_path}.png")
            return f"{output_path}.png"
        except Exception as e:
            self.logger.error(f"Failed to generate graph: {e}")
            return None

    def create_inference_timeline(self, steps: List[InferenceStep], max_steps: int = 10) -> Optional[str]:
        """
        Create inference timeline graph
        
        Args:
            steps: List of inference steps
            max_steps: Maximum number of steps to display
            
        Returns:
            Generated graph file path or None
        """
        if not GRAPHVIZ_AVAILABLE or not steps:
            return None
        
        # Limit number of displayed steps
        display_steps = steps[-max_steps:] if len(steps) > max_steps else steps
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Inference timeline', format='png')
        dot.attr(rankdir='LR', size='16,10')
        dot.attr('node', shape='record', style='filled')
        
        # Set graph attributes
        dot.attr(label=f'MarketToM Inference Timeline (Recent {len(display_steps)} Inferences)', 
                labelloc='t', fontsize='16')
        
        # Create node for each step
        for i, step in enumerate(display_steps):
            timestamp = datetime.fromisoformat(step.timestamp.replace('Z', '+00:00'))
            time_str = timestamp.strftime('%m-%d %H:%M')
            
            # Extract complete information with line wrapping
            belief_key = self.format_full_text(step.mental_states.belief, 200, line_width=25)
            emotion_key = self.format_full_text(step.mental_states.emotion, 200, line_width=25)
            
            node_label = f"{{Time: {time_str}|Belief: {belief_key}|Emotion: {emotion_key}}}"
            
            dot.node(f'step_{i}', node_label, 
                    fillcolor='lightblue')
            
            # Connect adjacent steps
            if i > 0:
                dot.edge(f'step_{i-1}', f'step_{i}')
        
        # Generate file
        output_path = os.path.join(self.output_dir, 'inference_timeline')
        
        try:
            dot.render(output_path, cleanup=True)
            self.logger.info(f"Generated timeline graph: {output_path}.png")
            return f"{output_path}.png"
        except Exception as e:
            self.logger.error(f"Failed to generate graph: {e}")
            return None

    def generate_summary_report(self, steps: List[InferenceStep]) -> str:
        """
        Generate inference summary report
        
        Args:
            steps: List of inference steps
            
        Returns:
            Summary report text
        """
        if not steps:
            return "No inference data available"
        
        report = f"""
MarketToM Mental State Inference Summary Report
==============================================

Inference Statistics:
- Total inferences: {len(steps)}
- Time range: {steps[0].timestamp} to {steps[-1].timestamp}

Recent inference results:
"""
        
        # Display key information from recent inferences
        recent_steps = steps[-5:] if len(steps) > 5 else steps
        
        for i, step in enumerate(recent_steps, 1):
            timestamp = datetime.fromisoformat(step.timestamp.replace('Z', '+00:00'))
            time_str = timestamp.strftime('%Y-%m-%d %H:%M')
            
            belief_summary = self.format_full_text(step.mental_states.belief, 300, line_width=30)
            emotion_summary = self.format_full_text(step.mental_states.emotion, 300, line_width=30)
            
            report += f"""
{i}. Inference time: {time_str}
   Market belief: {belief_summary}
   Market emotion: {emotion_summary}
   Strategies used: {len(step.strategies_used.get('belief', []))} belief strategies, 
                   {len(step.strategies_used.get('emotion', []))} emotion strategies
"""
        
        # Add visualization file information
        report += f"""

Generated visualization files:
- Causal Bayesian network graph: {self.output_dir}/causal_bayesian_network.png
- Inference timeline graph: {self.output_dir}/inference_timeline.png
- Individual inference graphs: {self.output_dir}/mental_state_*.png

Recommendations:
1. View causal network graph to understand MarketToM's overall architecture
2. Use timeline graph to observe evolution trends of market mental states
3. Analyze individual inference graphs for detailed market cognition at specific moments
"""
        
        return report

    def visualize_all(self, max_individual_graphs: int = 5) -> Dict[str, Any]:
        """
        Generate complete visualization report
        
        Args:
            max_individual_graphs: Maximum number of individual graphs
            
        Returns:
            Dictionary containing all generated file paths and summary
        """
        self.logger.info("Starting to generate complete mental state visualization report...")
        
        # Load all inference logs
        steps = self.load_all_inference_logs()
        
        if not steps:
            self.logger.warning("No inference logs found, cannot generate visualization")
            return {"error": "No inference data available"}
        
        result = {
            "summary": "",
            "causal_network": None,
            "timeline": None,
            "individual_graphs": [],
            "strategy_evolution_graphs": [],
            "backward_inference_graphs": [],
            "total_steps": len(steps)
        }
        
        # Generate causal network graph
        result["causal_network"] = self.create_causal_network_graph()
        
        # Generate timeline graph
        result["timeline"] = self.create_inference_timeline(steps)
        
        # Generate strategy evolution graphs
        for strategy_type in ['belief', 'intent', 'emotion']:
            evolution_graph = self.create_strategy_evolution_graph(strategy_type)
            if evolution_graph:
                result["strategy_evolution_graphs"].append(evolution_graph)
        
        # Generate individual inference graphs (recent ones)
        recent_steps = steps[-max_individual_graphs:] if len(steps) > max_individual_graphs else steps
        
        for step in recent_steps:
            # Regular inference graph
            graph_path = self.create_mental_state_graph(step)
            if graph_path:
                result["individual_graphs"].append(graph_path)
            
            # If backward inference results exist, generate backward inference graph
            if step.backward_inference:
                backward_graph = self.create_backward_inference_graph(step)
                if backward_graph:
                    result["backward_inference_graphs"].append(backward_graph)
        
        # Generate summary report
        result["summary"] = self.generate_summary_report(steps)
        
        # Save summary report to file
        report_path = os.path.join(self.output_dir, "summary_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(result["summary"])
        
        result["report_file"] = report_path
        
        self.logger.info(f"Completed visualization generation, processed {len(steps)} inference steps")
        self.logger.info(f"Strategy evolution graphs: {len(result['strategy_evolution_graphs'])}")
        self.logger.info(f"Backward inference graphs: {len(result['backward_inference_graphs'])}")
        self.logger.info(f"Results saved in: {self.output_dir}")
        
        return result

    def create_latest_complete_inference_graph(self) -> Optional[str]:
        """
        Create detailed flowchart of the latest complete inference process
        Including forward inference, strategy retrieval, prediction results and backward correction
        
        Returns:
            Generated graph file path or None
        """
        if not GRAPHVIZ_AVAILABLE:
            return None
        
        # Get latest inference log
        steps = self.load_all_inference_logs()
        if not steps:
            self.logger.error("No inference logs found")
            return None
        
        latest_step = steps[-1]  # Latest inference step
        
        # Load strategy database to get detailed strategy content
        strategies_db = self.load_strategy_database()
        
        # Load corresponding prediction results
        prediction_result = self._load_prediction_result(latest_step.timestamp)
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Latest complete inference flow', format='png')
        dot.attr(rankdir='TB', size='24,20', splines='false', nodesep='0.5', ranksep='0.8')
        dot.attr('node', shape='box', style='rounded,filled', margin='0.2')
        dot.attr('edge', color='black')
        
        # Set graph attributes (no title)
        
        # 1. Environmental State
        env_text = self.format_full_text(latest_step.environmental_state, 600, line_width=60)
        dot.node('env', f'Environmental State\\n{env_text}', 
                fillcolor='lightblue', fontsize='10')
        
        # 2. Forward inference mental states
        belief_text = self.format_full_text(latest_step.mental_states.belief, 800, line_width=45)
        intent_text = self.format_full_text(latest_step.mental_states.intent, 800, line_width=45)  
        emotion_text = self.format_full_text(latest_step.mental_states.emotion, 800, line_width=45)
        
        dot.node('belief', f'Belief State\\n{belief_text}', 
                fillcolor='lightgreen', fontsize='10')
        dot.node('intent', f'Intention State\\n{intent_text}', 
                fillcolor='lightyellow', fontsize='10')
        dot.node('emotion', f'Emotion State\\n{emotion_text}', 
                fillcolor='lightcoral', fontsize='10')
        
        # 3. Show retrieved strategies
        strategies_used = latest_step.strategies_used
        
        # Belief strategy
        if strategies_used.get('belief'):
            strategy_id = strategies_used['belief'][0]
            strategy_content = self._get_strategy_content(strategies_db['belief'], strategy_id)
            strategy_text = self.format_full_text(strategy_content, 600, line_width=40)
            dot.node('belief_strategy', f'Retrieved Strategy (Belief)\\nID: {strategy_id}\\n{strategy_text}', 
                    fillcolor='lightgreen', fontsize='9')
            dot.edge('belief_strategy', 'belief', 'Guide Inference', style='dashed', color='green')
        
        # Intent strategy
        if strategies_used.get('intent'):
            strategy_id = strategies_used['intent'][0]
            strategy_content = self._get_strategy_content(strategies_db['intent'], strategy_id)
            strategy_text = self.format_full_text(strategy_content, 600, line_width=40)
            dot.node('intent_strategy', f'Retrieved Strategy (Intent)\\nID: {strategy_id}\\n{strategy_text}', 
                    fillcolor='lightyellow', fontsize='9')
            dot.edge('intent_strategy', 'intent', 'Guide Inference', style='dashed', color='orange')
        
        # Emotion strategy
        if strategies_used.get('emotion'):
            strategy_id = strategies_used['emotion'][0]
            strategy_content = self._get_strategy_content(strategies_db['emotion'], strategy_id)
            strategy_text = self.format_full_text(strategy_content, 600, line_width=40)
            dot.node('emotion_strategy', f'Retrieved Strategy (Emotion)\\nID: {strategy_id}\\n{strategy_text}', 
                    fillcolor='lightcoral', fontsize='9')
            dot.edge('emotion_strategy', 'emotion', 'Guide Inference', style='dashed', color='red')
        
        # 4. Causal relationship edges
        dot.edge('env', 'belief', 'Influence')
        dot.edge('belief', 'intent', 'Lead to')
        dot.edge('env', 'emotion', 'Co-influence')
        dot.edge('belief', 'emotion', '')
        
        # 5. Prediction results
        if prediction_result:
            predicted_action = "Up" if prediction_result.get('predicted_up', False) else "Down"
            actual_action = "Up" if prediction_result.get('label', 0) == 1 else "Down"
            probability = prediction_result.get('probability', 0.0)
            is_correct = prediction_result.get('correct', False)
            
            # Expert prediction analysis (separate, centered node)
            dot.node('expert_analysis', f'Multi-Perspective\\nSampling', 
                    fillcolor='lightsteelblue', fontsize='8')
            
            # Create a horizontal layout for prediction results section
            with dot.subgraph(name='cluster_prediction') as pred_cluster:
                pred_cluster.attr(rank='same')
                pred_cluster.attr(style='invis')  # Make cluster invisible
                
                # Predicted action
                pred_color = 'lightgreen' if is_correct else 'salmon'
                pred_cluster.node('prediction', f'Predicted\\n{predicted_action}\\nP: {probability:.3f}', 
                        fillcolor=pred_color, fontsize='8')
                
                # Actual result
                pred_cluster.node('actual', f'Actual\\n{actual_action}', 
                        fillcolor='lightcyan', fontsize='8')
            
            # Connections
            dot.edge('intent', 'expert_analysis', 'Drive')
            dot.edge('emotion', 'expert_analysis', 'Modulate')
            dot.edge('expert_analysis', 'prediction', 'Generate')
            
            # Prediction result comparison
            if is_correct:
                dot.edge('prediction', 'actual', 'Correct')
            else:
                dot.edge('prediction', 'actual', 'Error', style='dashed')
                
                # 6. If prediction error, show backward inference strategy updates
                backward_updates = self._load_backward_inference_updates(latest_step.timestamp)
                self.logger.info(f"Loaded {len(backward_updates)} backward inference updates for latest step")
                if backward_updates:
                    dot.node('backward_analysis', 'Backward Inference Analysis', 
                            fillcolor='orange')
                    dot.edge('actual', 'backward_analysis', 'Trigger Learning')
                    
                    # Show specific strategy updates
                    for i, update in enumerate(backward_updates):
                        update_id = f'update_{update["level"]}_{i}'
                        update_text = self.format_full_text(update['content'], 500, line_width=35)
                        update_type = update['decision_type']
                        
                        # Use colors consistent with mental state nodes
                        level = update['level'].lower()
                        if level == 'belief':
                            color = 'lightgreen'
                        elif level == 'intent':
                            color = 'lightyellow'  
                        elif level == 'emotion':
                            color = 'lightcoral'
                        else:
                            color = 'lightgray'  # fallback color
                        
                        dot.node(update_id, 
                                f'{update_type} {update["level"].upper()} Strategy\\n{update_text}',
                                fillcolor=color, fontsize='9')
                        
                        dot.edge('backward_analysis', update_id, 'Update', 
                                style='dashed')
        
        # Generate file
        output_path = os.path.join(self.output_dir, 'latest_complete_inference')
        
        try:
            dot.render(output_path, cleanup=True)
            self.logger.info(f"Generated latest complete inference graph: {output_path}.png")
            return f"{output_path}.png"
        except Exception as e:
            self.logger.error(f"Failed to generate graph: {e}")
            return None

    def _get_strategy_content(self, strategy_list: List[Dict], strategy_id: str) -> str:
        """
        Get strategy content by ID from strategy list
        
        Args:
            strategy_list: Strategy list
            strategy_id: Strategy ID
            
        Returns:
            Strategy content string
        """
        for strategy in strategy_list:
            if isinstance(strategy, dict):
                item = strategy.get('item', {})
                if item.get('id') == strategy_id:
                    return item.get('strategy', 'Strategy content not found')
        
        return f'Strategy {strategy_id} not found'

    def _load_prediction_result(self, timestamp: str) -> Optional[Dict]:
        """
        Load prediction results for corresponding timestamp
        
        Args:
            timestamp: Inference timestamp
            
        Returns:
            Prediction result dictionary or None
        """
        prediction_file = os.path.join(self.storage_dir, "..", "prediction_results.json")
        
        if not os.path.exists(prediction_file):
            self.logger.warning("Prediction result file does not exist")
            return None
        
        try:
            with open(prediction_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find prediction result with closest timestamp
            predictions = data.get('predictions', [])
            target_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            closest_prediction = None
            min_time_diff = float('inf')
            
            for pred in predictions:
                pred_time = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                time_diff = abs((target_time - pred_time).total_seconds())
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_prediction = pred
            
            return closest_prediction
            
        except Exception as e:
            self.logger.error(f"Failed to load prediction results: {e}")
            return None

    def _load_backward_inference_updates(self, timestamp: str) -> List[Dict]:
        """
        Load strategy updates from backward inference
        
        Args:
            timestamp: Inference timestamp
            
        Returns:
            List of strategy updates
        """
        backward_logs_dir = os.path.join(self.storage_dir, "backward_inference_logs")
        
        if not os.path.exists(backward_logs_dir):
            self.logger.debug(f"Backward inference logs directory not found: {backward_logs_dir}")
            return []
        
        # Find backward inference log closest to given timestamp
        target_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        best_match = None
        min_time_diff = None
        
        try:
            for filename in os.listdir(backward_logs_dir):
                if filename.startswith('backward_') and filename.endswith('.json'):
                    filepath = os.path.join(backward_logs_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                        
                        # Compare original inference timestamp
                        log_timestamp = log_data.get('timestamp', '')
                        if log_timestamp:
                            log_time = datetime.fromisoformat(log_timestamp.replace('Z', '+00:00'))
                            time_diff = abs((target_time - log_time).total_seconds())
                            
                            if min_time_diff is None or time_diff < min_time_diff:
                                min_time_diff = time_diff
                                best_match = log_data
                                
                    except (json.JSONDecodeError, ValueError) as e:
                        self.logger.warning(f"Error reading backward inference log {filename}: {e}")
                        continue
            
            if best_match and min_time_diff is not None and min_time_diff < 3600:  # Match within 1 hour
                strategy_updates = best_match.get('strategy_updates', [])
                self.logger.info(f"Found {len(strategy_updates)} backward inference updates for timestamp {timestamp}")
                return strategy_updates
            else:
                self.logger.debug(f"No matching backward inference updates found for timestamp {timestamp}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error loading backward inference updates: {e}")
            return []
