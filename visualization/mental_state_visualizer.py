# -*- coding: utf-8 -*-

import json
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not installed. Run 'pip install graphviz' to enable graph generation.")

logger = logging.getLogger(__name__)

# ── Agent display colors ──
AGENT_COLORS = {
    "Retail":        {"fill": "#E8F5E9", "border": "#4CAF50"},   # green
    "Institutional": {"fill": "#E3F2FD", "border": "#2196F3"},   # blue
    "Arbitrageur":   {"fill": "#FFF3E0", "border": "#FF9800"},   # orange
}
DEFAULT_AGENT_COLOR = {"fill": "#F5F5F5", "border": "#9E9E9E"}

STATE_COLORS = {
    "belief":  "#C8E6C9",   # light green
    "intent":  "#FFF9C4",   # light yellow
    "emotion": "#FFCDD2",   # light red/coral
}


# ── Data classes ──
@dataclass
class AgentMentalState:
    agent_role: str
    belief: str = ""
    intent: str = ""
    emotion: str = ""


@dataclass
class InferenceStep:
    timestamp: str
    environmental_state: str
    agents: List[AgentMentalState] = field(default_factory=list)
    strategies_used: Dict[str, Any] = field(default_factory=dict)
    # Legacy compat: single mental_states dict
    legacy_mental_states: Optional[Dict[str, str]] = None


@dataclass
class BackwardUpdate:
    failing_agent: str
    level: str           # belief / intent / emotion
    decision_type: str   # CREATE / MODIFY
    content: str
    strategy_id: Optional[str] = None


# ── Main Visualizer ──
class MentalStateVisualizer:

    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = storage_dir
        self.inference_logs_dir = os.path.join(storage_dir, "inference_logs")
        self.backward_logs_dir = os.path.join(storage_dir, "backward_inference_logs")
        self.strategy_database_dir = os.path.join(storage_dir, "strategy_database")
        self.output_dir = os.path.join(storage_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz unavailable — will skip graph generation")

    # ────────────────────────────────────────────
    # 1. Data Loading
    # ────────────────────────────────────────────

    def load_inference_log(self, log_file: str) -> Optional[InferenceStep]:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            env_state = data.get('environmental_state', '')
            timestamp = data.get('timestamp', '')

            agents: List[AgentMentalState] = []
            legacy_ms: Optional[Dict[str, str]] = None

            # ── New multi-agent format ──
            if 'agent_results' in data:
                for role, states in data['agent_results'].items():
                    agents.append(AgentMentalState(
                        agent_role=role,
                        belief=self._extract_desc(states.get('belief', '')),
                        intent=self._extract_desc(states.get('intent', '')),
                        emotion=self._extract_desc(states.get('emotion', '')),
                    ))
            # ── Legacy single-agent format ──
            elif 'mental_states' in data:
                ms = data['mental_states']
                legacy_ms = {
                    'belief': self._extract_desc(ms.get('belief', '')),
                    'intent': self._extract_desc(ms.get('intent', '')),
                    'emotion': self._extract_desc(ms.get('emotion', '')),
                }
                # Convert to single-agent representation
                agents.append(AgentMentalState(
                    agent_role='Market',
                    belief=legacy_ms['belief'],
                    intent=legacy_ms['intent'],
                    emotion=legacy_ms['emotion'],
                ))

            return InferenceStep(
                timestamp=timestamp,
                environmental_state=env_state,
                agents=agents,
                strategies_used=data.get('strategies_used', {}),
                legacy_mental_states=legacy_ms,
            )
        except Exception as e:
            logger.error(f"Failed to load inference log {log_file}: {e}")
            return None

    def load_all_inference_logs(self) -> List[InferenceStep]:
        steps: List[InferenceStep] = []
        if not os.path.exists(self.inference_logs_dir):
            return steps
        for fname in os.listdir(self.inference_logs_dir):
            if fname.endswith('.json'):
                step = self.load_inference_log(os.path.join(self.inference_logs_dir, fname))
                if step:
                    steps.append(step)
        steps.sort(key=lambda s: s.timestamp)
        return steps

    def load_strategy_database(self) -> Dict[str, List[Dict]]:
        strategies: Dict[str, List[Dict]] = {"belief": [], "emotion": [], "intent": []}
        for stype in strategies:
            spath = os.path.join(self.strategy_database_dir, f"{stype}_strategies.json")
            if os.path.exists(spath):
                try:
                    with open(spath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        strategies[stype] = data.get("strategies", [])
                    elif isinstance(data, list):
                        strategies[stype] = data
                except Exception as e:
                    logger.error(f"Failed to load {spath}: {e}")
        return strategies

    def _load_prediction_result(self, timestamp: str) -> Optional[Dict]:
        pred_file = os.path.join(self.storage_dir, "..", "prediction_results.json")
        if not os.path.exists(pred_file):
            return None
        try:
            with open(pred_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            target = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            best, min_diff = None, float('inf')
            for p in data.get('predictions', []):
                pt = datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00'))
                diff = abs((target - pt).total_seconds())
                if diff < min_diff:
                    min_diff, best = diff, p
            return best
        except Exception:
            return None

    def _load_backward_updates(self, timestamp: str) -> List[BackwardUpdate]:
        if not os.path.exists(self.backward_logs_dir):
            return []
        target = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        best_data, min_diff = None, float('inf')

        for fname in os.listdir(self.backward_logs_dir):
            if not fname.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.backward_logs_dir, fname), 'r', encoding='utf-8') as f:
                    d = json.load(f)
                ts = d.get('timestamp', '')
                if ts:
                    lt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    diff = abs((target - lt).total_seconds())
                    if diff < min_diff:
                        min_diff, best_data = diff, d
            except Exception:
                continue

        if best_data is None or min_diff > 3600:
            return []

        updates: List[BackwardUpdate] = []
        failing_agent = best_data.get('failing_agent', 'Unknown')
        raw_updates = best_data.get('strategy_updates', {})

        if isinstance(raw_updates, dict):
            for level, items in raw_updates.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    updates.append(BackwardUpdate(
                        failing_agent=failing_agent,
                        level=level,
                        decision_type=item.get('type', item.get('decision_type', 'UPDATE')),
                        content=item.get('content', item.get('strategy', '')),
                        strategy_id=item.get('id', None),
                    ))
        elif isinstance(raw_updates, list):
            for item in raw_updates:
                updates.append(BackwardUpdate(
                    failing_agent=failing_agent,
                    level=item.get('level', ''),
                    decision_type=item.get('type', item.get('decision_type', 'UPDATE')),
                    content=item.get('content', item.get('strategy', '')),
                    strategy_id=item.get('id', None),
                ))
        return updates

    # ────────────────────────────────────────────
    # 2. Text Helpers
    # ────────────────────────────────────────────

    @staticmethod
    def _extract_desc(text: str) -> str:
        try:
            if isinstance(text, str) and text.strip().startswith('{'):
                data = json.loads(text)
                if isinstance(data, dict) and 'mental state description' in data:
                    return data['mental state description']
        except Exception:
            pass
        return text

    @staticmethod
    def format_full_text(text: str, max_length: int = 500, line_width: int = 50) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        if len(text) > max_length:
            cut = text[:max_length - 3]
            last_dot = cut.rfind('.')
            if last_dot > max_length * 0.7:
                text = text[:last_dot + 1] + "..."
            else:
                text = cut + "..."
        return MentalStateVisualizer._wrap_text(text, line_width)

    @staticmethod
    def _wrap_text(text: str, width: int = 50) -> str:
        if len(text) <= width:
            return text
        words = text.split(' ')
        lines, cur, cur_len = [], [], 0
        for w in words:
            if cur_len + len(w) + 1 > width and cur:
                lines.append(' '.join(cur))
                cur, cur_len = [w], len(w)
            else:
                cur.append(w)
                cur_len += len(w) + (1 if cur else 0)
        if cur:
            lines.append(' '.join(cur))
        return '\\n'.join(lines)

    @staticmethod
    def _get_strategy_content(strategy_list: List[Dict], strategy_id: str) -> str:
        for s in strategy_list:
            if isinstance(s, dict):
                item = s.get('item', s)
                if item.get('id') == strategy_id:
                    return item.get('strategy', item.get('content', 'Not found'))
        return f'Strategy {strategy_id} not found'

    def _agent_color(self, role: str) -> Dict[str, str]:
        return AGENT_COLORS.get(role, DEFAULT_AGENT_COLOR)

    # ────────────────────────────────────────────
    # 3. Graph: Multi-Agent CCN (Architecture)
    # ────────────────────────────────────────────

    def create_causal_network_graph(self) -> Optional[str]:
        if not GRAPHVIZ_AVAILABLE:
            return None

        dot = graphviz.Digraph(comment='Multi-Agent CCN', format='png')
        dot.attr(rankdir='TB', size='14,10')
        dot.attr('node', shape='box', style='rounded,filled', fontsize='11')
        dot.attr(label='MarketToM Multi-Agent Causal Cognitive Network (CCN)',
                 labelloc='t', fontsize='16', fontname='Helvetica-Bold')

        # Environment
        dot.node('env', 'Environmental State\n(Price + Social Media)',
                 fillcolor='lightblue', shape='ellipse')

        # Per-agent sub-graphs
        agent_roles = list(AGENT_COLORS.keys())
        for role in agent_roles:
            c = self._agent_color(role)
            with dot.subgraph(name=f'cluster_{role}') as sg:
                sg.attr(label=f'{role} Agent', style='dashed',
                        color=c['border'], fontcolor=c['border'], fontsize='13')
                sg.node(f'{role}_B', f'Belief\n({role})', fillcolor=c['fill'])
                sg.node(f'{role}_I', f'Intent\n({role})', fillcolor=c['fill'])
                sg.node(f'{role}_E', f'Emotion\n({role})', fillcolor=c['fill'])
                sg.node(f'{role}_A', f'Action p_up\n({role})', fillcolor=c['fill'])
                # CCN edges within agent
                sg.edge(f'{role}_B', f'{role}_I', 'Lead to')
                sg.edge(f'{role}_B', f'{role}_E')
                sg.edge(f'{role}_I', f'{role}_A', 'Drive')
                sg.edge(f'{role}_E', f'{role}_A', 'Modulate')
            dot.edge('env', f'{role}_B', 'Influence')
            dot.edge('env', f'{role}_E', 'Co-influence')

        # Second-order ToM edges (inter-agent)
        dot.edge('Retail_I', 'Institutional_B', '2nd-order ToM',
                 style='dashed', color='purple')
        dot.edge('Institutional_I', 'Arbitrageur_B', '2nd-order ToM',
                 style='dashed', color='purple')

        # Aggregation
        dot.node('agg', 'Dynamic Weighted\nAggregation\nW_k = Softmax((αA + γC)/T)',
                 fillcolor='#E1BEE7', shape='ellipse')
        for role in agent_roles:
            dot.edge(f'{role}_A', 'agg')
        dot.node('final', 'P(up)', fillcolor='#F8BBD0', shape='doubleoctagon')
        dot.edge('agg', 'final', 'Aggregate')

        # CEP
        dot.node('cep', 'Per-Agent CEP\n(Strategy Database)',
                 fillcolor='#FFECB3', shape='cylinder')
        for role in agent_roles:
            dot.edge('cep', f'{role}_B', 'Retrieve', style='dotted', color='gray')
        dot.node('backward', 'Inter-Agent\nBackward Learning',
                 fillcolor='#FFCCBC', shape='hexagon')
        dot.edge('final', 'backward', 'If wrong', style='dashed', color='red')
        dot.edge('backward', 'cep', 'Update strategies', style='dashed', color='red')

        output = os.path.join(self.output_dir, 'multi_agent_ccn')
        try:
            dot.render(output, cleanup=True)
            return f"{output}.png"
        except Exception as e:
            logger.error(f"Failed to render CCN graph: {e}")
            return None

    # ────────────────────────────────────────────
    # 4. Graph: Latest Complete Inference
    # ────────────────────────────────────────────

    def create_latest_complete_inference_graph(self) -> Optional[str]:
        if not GRAPHVIZ_AVAILABLE:
            return None

        steps = self.load_all_inference_logs()
        if not steps:
            logger.error("No inference logs found")
            return None

        latest = steps[-1]
        strategies_db = self.load_strategy_database()
        prediction = self._load_prediction_result(latest.timestamp)

        dot = graphviz.Digraph(comment='Latest inference flow', format='png')
        dot.attr(rankdir='TB', size='28,22', splines='false',
                 nodesep='0.6', ranksep='0.8')
        dot.attr('node', shape='box', style='rounded,filled', margin='0.2', fontsize='10')

        # ── 1. Environmental State ──
        env_text = self.format_full_text(latest.environmental_state, 600, 60)
        dot.node('env', f'Environmental State\\n{env_text}', fillcolor='lightblue')

        is_multi = len(latest.agents) > 1 or (latest.legacy_mental_states is None)

        # ── 2. Per-agent mental states ──
        for agent in latest.agents:
            role = agent.agent_role
            c = self._agent_color(role)
            prefix = role.lower()[:4]

            # Determine per-agent strategies
            agent_strats = latest.strategies_used.get(role, latest.strategies_used)

            with dot.subgraph(name=f'cluster_{role}') as sg:
                sg.attr(label=f'{role} Agent', style='dashed',
                        color=c['border'], fontcolor=c['border'], fontsize='12')

                b_text = self.format_full_text(agent.belief, 600, 42)
                i_text = self.format_full_text(agent.intent, 600, 42)
                e_text = self.format_full_text(agent.emotion, 600, 42)

                sg.node(f'{prefix}_belief', f'Belief ({role})\\n{b_text}',
                        fillcolor=STATE_COLORS['belief'])
                sg.node(f'{prefix}_intent', f'Intent ({role})\\n{i_text}',
                        fillcolor=STATE_COLORS['intent'])
                sg.node(f'{prefix}_emotion', f'Emotion ({role})\\n{e_text}',
                        fillcolor=STATE_COLORS['emotion'])

                # CCN edges
                sg.edge(f'{prefix}_belief', f'{prefix}_intent', 'Lead to')
                sg.edge(f'{prefix}_belief', f'{prefix}_emotion')

            # Env → agent
            dot.edge('env', f'{prefix}_belief', 'Influence')
            dot.edge('env', f'{prefix}_emotion', 'Co-influence')

            # ── 2b. Strategy nodes per agent ──
            for stype in ['belief', 'intent', 'emotion']:
                strat_ids = agent_strats.get(stype, [])
                if strat_ids:
                    sid = strat_ids[0] if isinstance(strat_ids[0], str) else str(strat_ids[0])
                    s_content = self._get_strategy_content(strategies_db.get(stype, []), sid)
                    s_text = self.format_full_text(s_content, 400, 38)
                    node_id = f'{prefix}_strat_{stype}'
                    dot.node(node_id,
                             f'CEP Strategy ({stype})\\nID: {sid}\\n{s_text}',
                             fillcolor='#FFECB3', fontsize='9')
                    dot.edge(node_id, f'{prefix}_{stype}', 'Guide',
                             style='dashed', color='gray')

        # ── 3. Prediction / Aggregation ──
        if prediction:
            predicted_up = prediction.get('predicted_up', False)
            label_val = prediction.get('label', 0)
            p_up = prediction.get('probability', 0.0)
            is_correct = prediction.get('correct', False)
            pred_action = "Up ↑" if predicted_up else "Down ↓"
            actual_action = "Up ↑" if label_val == 1 else "Down ↓"
            weights = prediction.get('agent_weights', {})

            if is_multi:
                # Dynamic Weighted Aggregation node
                weight_lines = '\\n'.join(
                    f'{r}: w={w:.3f}' for r, w in weights.items()
                ) if weights else ''
                dot.node('agg',
                         f'Dynamic Weighted Aggregation\\n{weight_lines}',
                         fillcolor='#E1BEE7', shape='ellipse', fontsize='10')
                for agent in latest.agents:
                    prefix = agent.agent_role.lower()[:4]
                    dot.edge(f'{prefix}_intent', 'agg', 'Drive')
                    dot.edge(f'{prefix}_emotion', 'agg', 'Modulate')
                dot.edge('agg', 'prediction', 'Aggregate')
            else:
                prefix = latest.agents[0].agent_role.lower()[:4]
                dot.edge(f'{prefix}_intent', 'prediction', 'Drive')
                dot.edge(f'{prefix}_emotion', 'prediction', 'Modulate')

            pred_color = '#C8E6C9' if is_correct else '#FFCDD2'
            dot.node('prediction',
                     f'Predicted: {pred_action}\\nP(up) = {p_up:.4f}',
                     fillcolor=pred_color)
            dot.node('actual', f'Actual: {actual_action}', fillcolor='lightcyan')

            if is_correct:
                dot.edge('prediction', 'actual', 'Correct ✓', color='green')
            else:
                dot.edge('prediction', 'actual', 'Error ✗',
                         style='dashed', color='red')

                # ── 4. Backward learning ──
                bk_updates = self._load_backward_updates(latest.timestamp)
                if bk_updates:
                    dot.node('backward', 'Inter-Agent Backward Learning',
                             fillcolor='#FFCCBC', shape='hexagon')
                    dot.edge('actual', 'backward', 'Trigger Learning')

                    for idx, upd in enumerate(bk_updates):
                        uid = f'upd_{idx}'
                        upd_text = self.format_full_text(upd.content, 400, 35)
                        color = STATE_COLORS.get(upd.level, '#F5F5F5')
                        dot.node(uid,
                                 f'{upd.decision_type} {upd.level.upper()} Strategy\\n'
                                 f'Agent: {upd.failing_agent}\\n{upd_text}',
                                 fillcolor=color, fontsize='9')
                        dot.edge('backward', uid, 'Update', style='dashed')

        # Render
        ts_str = latest.timestamp.replace(':', '').replace('-', '').replace('.', '_')
        output = os.path.join(self.output_dir, f'inference_flow_{ts_str}')
        try:
            dot.render(output, cleanup=True)
            return f"{output}.png"
        except Exception as e:
            logger.error(f"Failed to render inference graph: {e}")
            return None

    # ────────────────────────────────────────────
    # 5. Graph: Single-Step Mental State
    # ────────────────────────────────────────────

    def create_mental_state_graph(self, step: InferenceStep) -> Optional[str]:
        if not GRAPHVIZ_AVAILABLE:
            return None

        dot = graphviz.Digraph(comment='Mental state graph', format='png')
        dot.attr(rankdir='TB', size='16,12')
        dot.attr('node', shape='box', style='rounded,filled', fontsize='10')
        dot.attr(label=f'MarketToM Inference — {step.timestamp}',
                 labelloc='t', fontsize='14', fontname='Helvetica-Bold')

        env_text = self.format_full_text(step.environmental_state, 400, 50)
        dot.node('env', f'Environmental State\\n{env_text}', fillcolor='lightblue')

        for agent in step.agents:
            role = agent.agent_role
            c = self._agent_color(role)
            pfx = role.lower()[:4]

            b = self.format_full_text(agent.belief, 500, 40)
            i = self.format_full_text(agent.intent, 500, 40)
            e = self.format_full_text(agent.emotion, 500, 40)

            with dot.subgraph(name=f'cluster_{role}') as sg:
                sg.attr(label=f'{role}', style='dashed',
                        color=c['border'], fontcolor=c['border'])
                sg.node(f'{pfx}_B', f'Belief\\n{b}', fillcolor=STATE_COLORS['belief'])
                sg.node(f'{pfx}_I', f'Intent\\n{i}', fillcolor=STATE_COLORS['intent'])
                sg.node(f'{pfx}_E', f'Emotion\\n{e}', fillcolor=STATE_COLORS['emotion'])
                sg.edge(f'{pfx}_B', f'{pfx}_I', 'Lead to')
                sg.edge(f'{pfx}_B', f'{pfx}_E')
            dot.edge('env', f'{pfx}_B', 'Influence')
            dot.edge('env', f'{pfx}_E', 'Co-influence')

        if len(step.agents) > 1:
            dot.node('agg', 'Dynamic Weighted\nAggregation',
                     fillcolor='#E1BEE7', shape='ellipse')
            for agent in step.agents:
                pfx = agent.agent_role.lower()[:4]
                dot.edge(f'{pfx}_I', 'agg', 'Drive')
                dot.edge(f'{pfx}_E', 'agg')
            dot.node('action', 'P(up)', fillcolor='#F8BBD0',
                     shape='doubleoctagon')
            dot.edge('agg', 'action')
        else:
            pfx = step.agents[0].agent_role.lower()[:4]
            dot.node('action', 'Action', fillcolor='lightgray')
            dot.edge(f'{pfx}_I', 'action', 'Drive')
            dot.edge(f'{pfx}_E', 'action')

        ts_str = step.timestamp.replace(':', '-').replace('.', '-')
        output = os.path.join(self.output_dir, f'mental_state_{ts_str}')
        try:
            dot.render(output, cleanup=True)
            return f"{output}.png"
        except Exception as e:
            logger.error(f"Failed to render graph: {e}")
            return None

    # ────────────────────────────────────────────
    # 6. Graph: Strategy Evolution
    # ────────────────────────────────────────────

    def create_strategy_evolution_graph(self, strategy_type: str = "belief") -> Optional[str]:
        if not GRAPHVIZ_AVAILABLE:
            return None
        strategies = self.load_strategy_database()
        items = strategies.get(strategy_type, [])
        if not items:
            return None

        items.sort(key=lambda x: x.get("timestamp", ""))

        dot = graphviz.Digraph(comment=f'{strategy_type} strategy evolution', format='png')
        dot.attr(rankdir='TB', size='14,10')
        dot.attr('node', shape='box', style='rounded,filled')
        dot.attr(label=f'MarketToM {strategy_type.upper()} Strategy Evolution '
                       f'({len(items)} strategies)',
                 labelloc='t', fontsize='16')

        for i, s in enumerate(items):
            sid = s.get("id", f"s_{i}")
            content = self.format_full_text(s.get("strategy", ""), 400, 35)
            ts = s.get("timestamp", "")[:10]
            ver = s.get("version", 1)
            agent = s.get("agent_role", "")
            color = '#C8E6C9' if ver == 1 else '#FFF9C4'
            label_str = f'{strategy_type.upper()} v{ver}'
            if agent:
                label_str += f' ({agent})'
            dot.node(sid, f'{label_str}\\n{content}\\n{ts}', fillcolor=color)
            if i > 0:
                prev_id = items[i - 1].get("id", f"s_{i - 1}")
                dot.edge(prev_id, sid, 'Evolve')

        output = os.path.join(self.output_dir, f'{strategy_type}_strategy_evolution')
        try:
            dot.render(output, cleanup=True)
            return f"{output}.png"
        except Exception as e:
            logger.error(f"Failed to render evolution graph: {e}")
            return None

    # ────────────────────────────────────────────
    # 7. Graph: Inference Timeline
    # ────────────────────────────────────────────

    def create_inference_timeline(self, steps: List[InferenceStep],
                                  max_steps: int = 10) -> Optional[str]:
        if not GRAPHVIZ_AVAILABLE or not steps:
            return None
        display = steps[-max_steps:]

        dot = graphviz.Digraph(comment='Inference timeline', format='png')
        dot.attr(rankdir='LR', size='18,10')
        dot.attr('node', shape='record', style='filled')
        dot.attr(label=f'MarketToM Inference Timeline (last {len(display)})',
                 labelloc='t', fontsize='14')

        for i, step in enumerate(display):
            try:
                ts = datetime.fromisoformat(step.timestamp.replace('Z', '+00:00'))
                ts_str = ts.strftime('%m-%d %H:%M')
            except Exception:
                ts_str = step.timestamp[:16]

            agent_count = len(step.agents)
            roles = ', '.join(a.agent_role for a in step.agents)
            label = f"{{Time: {ts_str}|Agents: {agent_count} ({roles})}}"
            dot.node(f'step_{i}', label, fillcolor='lightblue')
            if i > 0:
                dot.edge(f'step_{i - 1}', f'step_{i}')

        output = os.path.join(self.output_dir, 'inference_timeline')
        try:
            dot.render(output, cleanup=True)
            return f"{output}.png"
        except Exception as e:
            logger.error(f"Failed to render timeline: {e}")
            return None

    # ────────────────────────────────────────────
    # 8. Summary Report
    # ────────────────────────────────────────────

    def generate_summary_report(self, steps: List[InferenceStep]) -> str:
        if not steps:
            return "No inference data available"

        report = f"""
MarketToM Multi-Agent Inference Summary
=======================================

Statistics:
- Total inferences : {len(steps)}
- Time range       : {steps[0].timestamp} → {steps[-1].timestamp}

Recent inferences:
"""
        for idx, step in enumerate(steps[-5:], 1):
            try:
                ts = datetime.fromisoformat(step.timestamp.replace('Z', '+00:00'))
                ts_str = ts.strftime('%Y-%m-%d %H:%M')
            except Exception:
                ts_str = step.timestamp[:16]

            agents_str = ', '.join(a.agent_role for a in step.agents)
            report += f"\n{idx}. {ts_str}  agents=[{agents_str}]\n"
            for a in step.agents:
                report += f"   [{a.agent_role}] belief : {a.belief[:80]}...\n"
                report += f"   [{a.agent_role}] emotion: {a.emotion[:80]}...\n"

        report += f"""
Visualisation files → {self.output_dir}/
- multi_agent_ccn.png          : Architecture overview
- inference_flow_*.png         : Per-sample detailed flows
- inference_timeline.png       : Timeline
- *_strategy_evolution.png     : Strategy evolution graphs
"""
        return report

    # ────────────────────────────────────────────
    # 9. Entry Point — Generate Everything
    # ────────────────────────────────────────────

    def visualize_all(self, max_individual_graphs: int = 5) -> Dict[str, Any]:
        logger.info("Starting multi-agent visualisation...")

        steps = self.load_all_inference_logs()
        if not steps:
            return {"error": "No inference data available"}

        result: Dict[str, Any] = {
            "summary": "",
            "ccn_graph": None,
            "timeline": None,
            "individual_graphs": [],
            "strategy_evolution_graphs": [],
            "total_steps": len(steps),
        }

        result["ccn_graph"] = self.create_causal_network_graph()
        result["timeline"] = self.create_inference_timeline(steps)

        for stype in ['belief', 'intent', 'emotion']:
            g = self.create_strategy_evolution_graph(stype)
            if g:
                result["strategy_evolution_graphs"].append(g)

        recent = steps[-max_individual_graphs:]
        for step in recent:
            g = self.create_mental_state_graph(step)
            if g:
                result["individual_graphs"].append(g)

        result["summary"] = self.generate_summary_report(steps)

        report_path = os.path.join(self.output_dir, "summary_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(result["summary"])
        result["report_file"] = report_path

        logger.info(f"Visualisation complete — {len(steps)} steps processed")
        return result
