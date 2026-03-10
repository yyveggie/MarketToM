# -*- coding: utf-8 -*-
"""
Inter-Agent Learning via Backward Inference.

When a prediction is wrong, identifies which agents predicted correctly
(successful peers) and which failed. For each failing agent, the backward
inference compares the failing trace with all successful peers' traces,
then updates the failing agent's CEP strategies.

Update rule:
    Pi_A^updates = LLM_Learn(Context, Pi_A^retrieved, {M_t^k}_{k in S_t}, actual_action)
"""
import json
import os
import time
import logging
from typing import Dict, Optional, Any, List, Tuple
import traceback
from openai import OpenAI
import random
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator
from core.cep import CognitiveEnhancementPlugin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_analysis.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.ErrorAnalysis')

COLOR_TITLE = "\033[1;36m"
COLOR_SUCCESS = "\033[1;32m"
COLOR_WARNING = "\033[1;33m"
COLOR_ERROR = "\033[1;31m"
COLOR_INFO = "\033[0;34m"
COLOR_VALUE = "\033[1;35m"
COLOR_DEBUG = "\033[0;90m"
COLOR_PHASE = "\033[1;94m"
COLOR_BELIEF = "\033[1;35m"
COLOR_INTENT = "\033[1;33m"
COLOR_EMOTION = "\033[1;36m"
COLOR_CREATE = "\033[1;32m"
COLOR_MODIFY = "\033[1;34m"
COLOR_RESET = "\033[0m"

DEFAULT_AGENT_ROLES = ["Retail", "Institutional", "Arbitrageur"]

last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 20.
DEFAULT_COOLDOWN = 20.0
MAX_JITTER = 1.0


def rate_limit_api_call(func):
    def wrapper(*args, **kwargs):
        global last_api_request_time
        now = datetime.now()
        elapsed = (now - last_api_request_time).total_seconds()
        if elapsed < MIN_REQUEST_INTERVAL:
            wait = MIN_REQUEST_INTERVAL - elapsed + random.uniform(0, MAX_JITTER)
            time.sleep(wait)
        last_api_request_time = datetime.now()
        result = func(*args, **kwargs)
        cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
        time.sleep(cooldown)
        return result
    return wrapper


class StrategyUpdateItem(BaseModel):
    """One strategy update from backward inference."""
    target_state_type: str = Field(..., description="Belief, Intention, or Emotion")
    decision_type: str = Field(..., description="MODIFY or CREATE")
    target_strategy_id: Optional[str] = Field(None)
    updated_strategy_content: str = Field(...)
    tom_components: List[dict] = Field(default_factory=list)
    justification: str = Field("")

    @field_validator('target_state_type')
    @classmethod
    def validate_state_type(cls, v):
        normalized = v.lower().strip()
        mapping = {"belief": "belief", "intention": "intent", "intent": "intent", "emotion": "emotion"}
        if normalized not in mapping:
            raise ValueError(f"Invalid target_state_type: {v}")
        return mapping[normalized]

    @field_validator('decision_type')
    @classmethod
    def validate_decision_type(cls, v):
        if v.upper() not in ['CREATE', 'MODIFY']:
            raise ValueError(f"decision_type must be CREATE or MODIFY, got: {v}")
        return v.upper()


class BackwardInferenceResponse(BaseModel):
    """Complete backward inference response."""
    failing_agent: str = Field(...)
    peer_insight: str = Field("")
    strategy_updates: List[StrategyUpdateItem] = Field(default_factory=list)


class BackwardInference:
    """Inter-Agent Learning: failed agents learn from successful peers."""

    def __init__(self,
                 cep: CognitiveEnhancementPlugin,
                 llm_client: OpenAI,
                 llm_model: str,
                 backward_template_abs_path: str,
                 inference_logs_abs_path: str,
                 agent_roles: Optional[List[str]] = None,
                 max_retries: int = 5,
                 base_delay_seconds: float = 2,
                 llm_temperature: float = 0.7,
                 llm_max_tokens: int = 5000):

        self.cep = cep
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.backward_template_abs_path = backward_template_abs_path
        self.inference_logs_abs_path = inference_logs_abs_path
        self.agent_roles = agent_roles or DEFAULT_AGENT_ROLES
        self.max_retries = max_retries
        self.base_delay = base_delay_seconds
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens

        self.backward_logs_dir = os.path.join(
            os.path.dirname(inference_logs_abs_path), "backward_inference_logs"
        )
        os.makedirs(self.backward_logs_dir, exist_ok=True)

        logger.info(f"Backward inference initialized: agents={self.agent_roles}")

    def _load_prompt_template(self) -> str:
        with open(self.backward_template_abs_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_inference_result(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.inference_logs_abs_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_other_roles(self, agent_role: str) -> Tuple[str, str]:
        others = [r for r in self.agent_roles if r != agent_role]
        if len(others) >= 2:
            return others[0], others[1]
        return others[0] if others else "Other", "Other"

    def _get_strategy_text(self, level: str, strategy_id: Optional[str],
                           agent_role: Optional[str] = None) -> str:
        if not strategy_id:
            return "No strategy used."
        data = self.cep.get_strategy_by_id(level, strategy_id, agent_role or "")
        if data and "item" in data and "strategy" in data["item"]:
            return data["item"]["strategy"]
        return f"Strategy not found: {strategy_id}"

    def _build_successful_peers_block(self, successful_agents: Dict[str, Dict]) -> str:
        """Build XML block for successful peers' traces."""
        blocks = []
        for role, trace in successful_agents.items():
            block = f"""      <Peer>
        <AgentRole>{role}</AgentRole>
        <Belief>{trace.get('belief', 'N/A')}</Belief>
        <Intention>{trace.get('intent', 'N/A')}</Intention>
        <Emotion>{trace.get('emotion', 'N/A')}</Emotion>
      </Peer>"""
            blocks.append(block)
        return "\n".join(blocks) if blocks else "<NoPeersAvailable/>"

    def _save_backward_log(self, timestamp: str, failing_agent: str,
                           predicted_action: str, actual_action: str,
                           inference_filename: str, strategy_updates: Any,
                           analysis_result: str) -> str:
        log_entry = {
            "timestamp": timestamp,
            "failing_agent": failing_agent,
            "prediction_error": {
                "predicted_action": predicted_action,
                "actual_action": actual_action
            },
            "original_inference_file": inference_filename,
            "strategy_updates": strategy_updates,
            "llm_analysis": analysis_result,
            "backward_inference_timestamp": datetime.now().isoformat()
        }
        dt = datetime.now()
        fname = f"backward_{failing_agent.lower()}_{dt.strftime('%Y%m%d_%H%M%S')}.json"
        fpath = os.path.join(self.backward_logs_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved backward log: {fpath}")
        return fpath

    @rate_limit_api_call
    def _call_backward_llm(self, prompt: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"Backward LLM call failed (attempt {attempt+1}): {e}")
                time.sleep(delay)
        return None

    def _rebuild_states_scenario(self, level: str, agent_trace: Dict,
                                 env_state: str) -> Dict[str, str]:
        """Rebuild state scenario for strategy insertion."""
        scenario = {}
        if level == "belief":
            scenario["environmental"] = env_state
        elif level == "intent":
            scenario["belief"] = agent_trace.get('belief', '')
        elif level == "emotion":
            scenario["belief"] = agent_trace.get('belief', '')
            scenario["environmental"] = env_state
        return scenario

    def _process_updates(self, llm_response: str, failing_agent: str,
                         agent_trace: Dict, env_state: str,
                         strategies_used: Dict[str, List[str]]) -> Dict:
        """Process LLM backward response and apply strategy updates."""
        try:
            data = json.loads(llm_response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            first = llm_response.find('{')
            last = llm_response.rfind('}')
            if first != -1 and last > first:
                try:
                    data = json.loads(llm_response[first:last+1])
                except Exception:
                    logger.error("Cannot parse backward inference response")
                    return {}
            else:
                return {}

        # Normalize keys (replace spaces with underscores)
        def norm_keys(obj):
            if isinstance(obj, dict):
                return {k.replace(' ', '_'): norm_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [norm_keys(i) for i in obj]
            return obj

        data = norm_keys(data)

        try:
            response = BackwardInferenceResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"Pydantic validation failed: {e}, trying partial parse")
            response = None

        updates_list = []
        if response:
            updates_list = response.strategy_updates
        elif "strategy_updates" in data and isinstance(data["strategy_updates"], list):
            for item in data["strategy_updates"]:
                try:
                    updates_list.append(StrategyUpdateItem.model_validate(item))
                except Exception:
                    pass

        grouped_updates = {}

        for update in updates_list:
            level = update.target_state_type
            content = update.updated_strategy_content
            decision = update.decision_type
            target_id = update.target_strategy_id

            level_color = {
                "belief": COLOR_BELIEF,
                "intent": COLOR_INTENT,
                "emotion": COLOR_EMOTION
            }.get(level, COLOR_INFO)

            if decision == "MODIFY" and target_id:
                existing = self.cep.get_strategy_by_id(level, target_id, failing_agent)
                if existing:
                    new_id = self.cep.update_strategy(level, target_id, content, failing_agent)
                    if new_id:
                        print(f"{COLOR_MODIFY}  ✓ Modified {level_color}{level}{COLOR_RESET} strategy for {failing_agent}")
                        grouped_updates.setdefault(level, []).append({
                            "type": "MODIFY", "id": new_id, "content": content
                        })
                    continue
                else:
                    logger.info(f"Strategy {target_id} not found, switching to CREATE")
                    decision = "CREATE"

            if decision == "CREATE":
                scenario = self._rebuild_states_scenario(level, agent_trace, env_state)
                new_id = self.cep.insert_strategy(level, scenario, content, failing_agent)
                if new_id:
                    print(f"{COLOR_CREATE}  ✓ Created new {level_color}{level}{COLOR_RESET} strategy for {failing_agent}")
                    grouped_updates.setdefault(level, []).append({
                        "type": "CREATE", "id": new_id, "content": content
                    })

        return grouped_updates

    def perform_backward_inference(self, filename: str,
                                   agent_predictions: Dict[str, Dict] = None,
                                   actual_action: str = "",
                                   # Legacy params
                                   predicted_action: str = None) -> Optional[Dict]:
        """Perform inter-agent backward inference.
        
        Args:
            filename: Inference log filename
            agent_predictions: {agent_role: {predicted_action, p_up, ...}}
            actual_action: Ground truth action (Buy or Sell)
            predicted_action: Legacy param (single prediction)
            
        Returns:
            Dict with analysis results per failing agent, or legacy format
        """
        try:
            print(f"\n{COLOR_TITLE}=== INTER-AGENT BACKWARD INFERENCE ==={COLOR_RESET}")

            log_data = self._load_inference_result(filename)
            agent_results = log_data.get('agent_results', {})
            strategies_used = log_data.get('strategies_used', {})
            env_state = log_data.get('environmental_state', '')

            # Legacy compat: if called with old single-prediction interface
            if agent_predictions is None and predicted_action is not None:
                # Treat all agents as failing
                agent_predictions = {
                    role: {'predicted_action': predicted_action}
                    for role in self.agent_roles
                }
                # If old format, build agent_results from mental_states
                if not agent_results and 'mental_states' in log_data:
                    ms = log_data['mental_states']
                    for role in self.agent_roles:
                        agent_results[role] = {
                            'belief': ms.get('belief', ''),
                            'intent': ms.get('intent', ''),
                            'emotion': ms.get('emotion', '')
                        }

            if not agent_predictions:
                logger.error("No agent predictions provided")
                return None

            # Identify successful and failing agents
            successful_agents = {}
            failing_agents = {}

            for role, pred_info in agent_predictions.items():
                pred_act = pred_info.get('predicted_action', '')
                if pred_act == actual_action:
                    successful_agents[role] = agent_results.get(role, {})
                else:
                    failing_agents[role] = agent_results.get(role, {})

            if not failing_agents:
                print(f"{COLOR_SUCCESS}All agents predicted correctly, no backward inference needed.{COLOR_RESET}")
                return None

            print(f"{COLOR_INFO}Successful: {list(successful_agents.keys())}{COLOR_RESET}")
            print(f"{COLOR_WARNING}Failing: {list(failing_agents.keys())}{COLOR_RESET}")

            all_results = {}

            for failing_role, failing_trace in failing_agents.items():
                print(f"\n{COLOR_PHASE}--- Learning for {failing_role} ---{COLOR_RESET}")

                role2, role3 = self._get_other_roles(failing_role)

                # Get strategies used by failing agent
                agent_strats = strategies_used.get(failing_role, {})
                belief_ids = agent_strats.get('belief', [])
                intent_ids = agent_strats.get('intent', [])
                emotion_ids = agent_strats.get('emotion', [])

                belief_strat = self._get_strategy_text('belief', belief_ids[0] if belief_ids else None, failing_role)
                intent_strat = self._get_strategy_text('intent', intent_ids[0] if intent_ids else None, failing_role)
                emotion_strat = self._get_strategy_text('emotion', emotion_ids[0] if emotion_ids else None, failing_role)

                # Build prompt
                template = self._load_prompt_template()
                peers_block = self._build_successful_peers_block(successful_agents)
                pred_action = agent_predictions[failing_role].get('predicted_action', 'Unknown')

                prompt = template
                prompt = prompt.replace('[AGENT_ROLE]', self.agent_roles[0] if self.agent_roles else failing_role)
                prompt = prompt.replace('[AGENT_ROLE_2]', role2)
                prompt = prompt.replace('[AGENT_ROLE_3]', role3)
                prompt = prompt.replace('[FAILING_AGENT]', failing_role)
                prompt = prompt.replace('[PREDICTED_ACTION]', pred_action)
                prompt = prompt.replace('[ACTUAL_ACTION]', actual_action)
                prompt = prompt.replace('[FAILING_AGENT_BELIEF]', failing_trace.get('belief', 'N/A'))
                prompt = prompt.replace('[FAILING_AGENT_INTENTION]', failing_trace.get('intent', 'N/A'))
                prompt = prompt.replace('[FAILING_AGENT_EMOTION]', failing_trace.get('emotion', 'N/A'))
                prompt = prompt.replace('[SUCCESSFUL_PEERS_BLOCK]', peers_block)
                prompt = prompt.replace('[BELIEF_STRATEGY_ID]', belief_ids[0] if belief_ids else "None")
                prompt = prompt.replace('[BELIEF_STRATEGY_CONTENT]', belief_strat)
                prompt = prompt.replace('[INTENTION_STRATEGY_ID]', intent_ids[0] if intent_ids else "None")
                prompt = prompt.replace('[INTENTION_STRATEGY_CONTENT]', intent_strat)
                prompt = prompt.replace('[EMOTION_STRATEGY_ID]', emotion_ids[0] if emotion_ids else "None")
                prompt = prompt.replace('[EMOTION_STRATEGY_CONTENT]', emotion_strat)

                llm_response = self._call_backward_llm(prompt)
                if not llm_response:
                    print(f"{COLOR_ERROR}✗ Backward inference failed for {failing_role}{COLOR_RESET}")
                    continue

                grouped_updates = self._process_updates(
                    llm_response, failing_role, failing_trace, env_state, agent_strats
                )

                self._save_backward_log(
                    timestamp=log_data.get('timestamp', datetime.now().isoformat()),
                    failing_agent=failing_role,
                    predicted_action=pred_action,
                    actual_action=actual_action,
                    inference_filename=filename,
                    strategy_updates=grouped_updates,
                    analysis_result=llm_response
                )

                all_results[failing_role] = {
                    'analysis': llm_response,
                    'strategy_updates': grouped_updates
                }

                total_updates = sum(len(u) for u in grouped_updates.values())
                print(f"{COLOR_SUCCESS}✓ {failing_role}: {total_updates} strategy updates applied{COLOR_RESET}")

            return all_results

        except Exception as e:
            logger.error(f"Backward inference error: {e}\n{traceback.format_exc()}")
            print(f"{COLOR_ERROR}✗ Backward inference error: {e}{COLOR_RESET}")
            return None
