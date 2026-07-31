# -*- coding: utf-8 -*-
import os
import json
import time
import random
import traceback
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import openai
from pydantic import BaseModel, Field

from core.cep import CognitiveEnhancementPlugin

try:
    import tqdm as tqdm_module

    class NoOpTqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else kwargs.get('iterable', None)
            if self.iterable is None and 'total' in kwargs:
                self.iterable = range(kwargs['total'])
        def __iter__(self):
            return iter(self.iterable)
        def update(self, *a, **kw): pass
        def close(self, *a, **kw): pass
        def set_description(self, *a, **kw): pass

    tqdm_module.tqdm = NoOpTqdm
    if hasattr(tqdm_module, 'notebook'):
        tqdm_module.notebook.tqdm = NoOpTqdm
    tqdm = NoOpTqdm
except ImportError:
    def tqdm(*args, **kwargs):
        return args[0] if args else kwargs.get('iterable', range(kwargs.get('total', 0)))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_inference.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.Inference')

COLOR_TITLE = "\033[1;36m"
COLOR_SUCCESS = "\033[1;32m"
COLOR_WARNING = "\033[1;33m"
COLOR_ERROR = "\033[1;31m"
COLOR_INFO = "\033[0;34m"
COLOR_VALUE = "\033[1;35m"
COLOR_DEBUG = "\033[0;90m"
COLOR_PHASE = "\033[1;94m"
COLOR_RESET = "\033[0m"

DEFAULT_AGENT_ROLES = ["Retail", "Institutional", "Arbitrageur"]

last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 20.
DEFAULT_COOLDOWN = 20.0
MAX_JITTER = 1.0


# ---------- Pydantic response model ----------
class AgentMentalStateResponse(BaseModel):
    agent_role: str = Field(...)
    state_type: str = Field(...)
    description: str = Field(..., description="Inferred mental state description")


# ---------- Rate limiter ----------
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


# ---------- Data logger ----------
class DataLogger:

    def __init__(self, log_dir_abs_path: str):
        self.log_dir = log_dir_abs_path
        os.makedirs(self.log_dir, exist_ok=True)
        logger.info(f"DataLogger: {self.log_dir}")

    def save_inference(self, timestamp: datetime,
                       env_state: str,
                       agent_results: Dict[str, Dict[str, str]],
                       strategies_used: Dict[str, Dict[str, List[str]]],
                       # Legacy compat
                       mental_states: Dict[str, str] = None,
                       run_metadata: Dict[str, object] = None) -> str:
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "environmental_state": env_state,
            "agent_results": agent_results,
            "strategies_used": strategies_used,
        }
        # Records which backbone and configuration produced this trace, so that
        # downstream analyses (e.g. cross-LLM consistency) can verify provenance
        # instead of assuming the config has not changed since the run.
        if run_metadata:
            log_entry["run_metadata"] = run_metadata
        # Keep legacy field for backward compat
        if mental_states:
            log_entry["mental_states"] = mental_states

        filename = f"inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved inference log: {filepath}")
        return filename


# ---------- Agent trace dataclass ----------
@dataclass
class AgentTrace:
    agent_role: str
    belief: str = ""
    intent: str = ""
    emotion: str = ""
    belief_strategy_ids: List[str] = field(default_factory=list)
    intent_strategy_ids: List[str] = field(default_factory=list)
    emotion_strategy_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, str]:
        return {
            "belief": self.belief,
            "intent": self.intent,
            "emotion": self.emotion
        }

    def strategies_dict(self) -> Dict[str, List[str]]:
        return {
            "belief": self.belief_strategy_ids,
            "intent": self.intent_strategy_ids,
            "emotion": self.emotion_strategy_ids
        }


# ---------- Main inference class ----------
class MentalStateInference:

    def __init__(self,
                 cep: CognitiveEnhancementPlugin,
                 logger: DataLogger,
                 llm_client: openai.OpenAI,
                 llm_model: str,
                 forward_template_abs_path: str,
                 cep_default_top_k: int = 1,
                 cep_similarity_threshold: float = 0.1,
                 fwd_inf_max_retries: int = 5,
                 fwd_inf_base_delay: int = 1,
                 emotion_similarity_threshold: float = 0.1,
                 belief_similarity_threshold: float = 0.1,
                 intent_similarity_threshold: float = 0.1,
                 llm_temperature: float = 0.7,
                 agent_roles: List[str] = None,
                 tom_order: int = 2,
                 cep_enabled: bool = True,
                 ccn_dependency_variant: str = "full",
                 llm_extra_body: Optional[dict] = None):

        self.cep = cep
        self.data_logger = logger
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.template_file_abs_path = forward_template_abs_path
        self.default_top_k = cep_default_top_k
        self.similarity_threshold = cep_similarity_threshold
        self.max_retries = fwd_inf_max_retries
        self.base_delay = fwd_inf_base_delay
        self.llm_temperature = llm_temperature
        self.agent_roles = agent_roles or DEFAULT_AGENT_ROLES
        self.tom_order = tom_order
        self.cep_enabled = cep_enabled
        self.ccn_dependency_variant = ccn_dependency_variant
        self.llm_extra_body = llm_extra_body

        self.threshold_map = {
            "belief": belief_similarity_threshold,
            "intent": intent_similarity_threshold,
            "emotion": emotion_similarity_threshold,
        }

    # ----- helpers -----
    def run_metadata(self) -> Dict[str, object]:
        """Configuration provenance stored alongside every inference log."""
        return {
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "agent_roles": list(self.agent_roles),
            "tom_order": self.tom_order,
            "cep_enabled": self.cep_enabled,
            "cep_top_k": self.default_top_k,
            "ccn_dependency_variant": self.ccn_dependency_variant,
            "forward_template": os.path.basename(self.template_file_abs_path),
        }

    def _load_template(self) -> str:
        with open(self.template_file_abs_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_other_roles(self, agent_role: str) -> Tuple[str, str]:
        others = [r for r in self.agent_roles if r != agent_role]
        return (others[0], others[1]) if len(others) >= 2 else (others[0] if others else "Other", "Other")

    def _retrieve_strategies(self, state_type: str, query_scenario: Dict[str, str],
                             agent_role: str) -> Tuple[str, List[str]]:
        if not self.cep_enabled:
            return "No strategy (CEP disabled).", []
        threshold = self.threshold_map.get(state_type, self.similarity_threshold)
        results = self.cep.retrieve_strategies(
            level=state_type,
            query_scenario=query_scenario,
            agent_role=agent_role,
            top_k=self.default_top_k,
            similarity_threshold=threshold
        )
        if results:
            best = results[0]
            strategy_text = best.get("strategy", "No strategy available.")
            strategy_id = best.get("id", "")
            return strategy_text, [strategy_id] if strategy_id else []
        return "No relevant strategy found.", []

    def _build_prompt(self, agent_role: str, state_type: str,
                      env_state: str, belief_state: str,
                      strategy_id: str, strategy_content: str) -> str:
        template = self._load_template()
        role2, role3 = self._get_other_roles(agent_role)

        prompt = template
        prompt = prompt.replace('[AGENT_ROLE]', agent_role)
        prompt = prompt.replace('[AGENT_ROLE_2]', role2)
        prompt = prompt.replace('[AGENT_ROLE_3]', role3)
        prompt = prompt.replace('[STATE_TYPE]', state_type.capitalize())
        prompt = prompt.replace('[ENVIRONMENTAL_STATE]', env_state)
        prompt = prompt.replace('[BELIEF_STATE]', belief_state or "N/A")
        prompt = prompt.replace('[STRATEGY_ID]', strategy_id or "None")
        prompt = prompt.replace('[STRATEGY_CONTENT]', strategy_content or "No strategy.")

        if self.tom_order < 2:
            prompt = prompt.replace(
                "IMPORTANT: You MUST explicitly consider both first-order and second-order ToM about other groups (internally).",
                "IMPORTANT: You MUST explicitly consider first-order ToM about other groups (internally)."
            )
            prompt = prompt.replace(
                f"Based on first- and second-order Theory-of-Mind reasoning regarding {role2} and {role3},",
                f"Based on first-order Theory-of-Mind reasoning regarding {role2} and {role3},"
            )
            prompt = prompt.replace(
                f"2. Consider second-order ToM: what would {role2} and {role3} "
                f"believe about each other's (and your) relevant states?",
                f"2. (Higher-order peer-belief reasoning is not used in this configuration.)"
            )

        return prompt

    @rate_limit_api_call
    def _infer_state(self, prompt: str, state_type: str,
                     agent_role: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                request_kwargs = {
                    "model": self.llm_model,
                    "messages": [{"role": "system", "content": prompt}],
                    "temperature": self.llm_temperature,
                    "response_format": {"type": "json_object"},
                }
                if self.llm_extra_body:
                    request_kwargs["extra_body"] = self.llm_extra_body
                response = self.llm_client.chat.completions.create(**request_kwargs)
                raw = response.choices[0].message.content.strip()
                data = json.loads(raw)
                parsed = AgentMentalStateResponse.model_validate(data)
                return parsed.description
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error ({agent_role}/{state_type}, attempt {attempt+1}): {e}")
            except Exception as e:
                logger.warning(f"LLM error ({agent_role}/{state_type}, attempt {attempt+1}): {e}")
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
        return None

    def infer_agent_trace(self, agent_role: str, env_state: str, belief_env_state: str = None) -> AgentTrace:
        trace = AgentTrace(agent_role=agent_role)
        if belief_env_state is None:
            belief_env_state = env_state

        # --- Belief ---
        print(f"  {COLOR_PHASE}[{agent_role}]{COLOR_RESET} Inferring belief...")
        query_scenario = {"environmental": belief_env_state}
        strat_text, strat_ids = self._retrieve_strategies("belief", query_scenario, agent_role)
        prompt = self._build_prompt(agent_role, "Belief", belief_env_state, "N/A",
                                    strat_ids[0] if strat_ids else "None", strat_text)
        belief = self._infer_state(prompt, "belief", agent_role)
        if belief:
            trace.belief = belief
            trace.belief_strategy_ids = strat_ids
            print(f"  {COLOR_SUCCESS}✓ {agent_role} belief inferred{COLOR_RESET}")
        else:
            print(f"  {COLOR_ERROR}✗ {agent_role} belief failed{COLOR_RESET}")

        # --- Intention ---
        print(f"  {COLOR_PHASE}[{agent_role}]{COLOR_RESET} Inferring intention...")
        intent_parent_belief = "N/A" if self.ccn_dependency_variant == "no_belief_to_intent" else trace.belief
        query_scenario = {"belief": intent_parent_belief}
        strat_text, strat_ids = self._retrieve_strategies("intent", query_scenario, agent_role)
        prompt = self._build_prompt(agent_role, "Intention", env_state, intent_parent_belief,
                                    strat_ids[0] if strat_ids else "None", strat_text)
        intent = self._infer_state(prompt, "intent", agent_role)
        if intent:
            trace.intent = intent
            trace.intent_strategy_ids = strat_ids
            print(f"  {COLOR_SUCCESS}✓ {agent_role} intention inferred{COLOR_RESET}")
        else:
            print(f"  {COLOR_ERROR}✗ {agent_role} intention failed{COLOR_RESET}")

        # --- Emotion ---
        print(f"  {COLOR_PHASE}[{agent_role}]{COLOR_RESET} Inferring emotion...")
        emotion_parent_belief = "N/A" if self.ccn_dependency_variant == "no_belief_to_emotion" else trace.belief
        query_scenario = {"belief": emotion_parent_belief, "environmental": env_state}
        strat_text, strat_ids = self._retrieve_strategies("emotion", query_scenario, agent_role)
        prompt = self._build_prompt(agent_role, "Emotion", env_state, emotion_parent_belief,
                                    strat_ids[0] if strat_ids else "None", strat_text)
        emotion = self._infer_state(prompt, "emotion", agent_role)
        if emotion:
            trace.emotion = emotion
            trace.emotion_strategy_ids = strat_ids
            print(f"  {COLOR_SUCCESS}✓ {agent_role} emotion inferred{COLOR_RESET}")
        else:
            print(f"  {COLOR_ERROR}✗ {agent_role} emotion failed{COLOR_RESET}")

        return trace

    def forward_inference(self, env_state: str, belief_env_state: str = None) -> Tuple[Dict[str, Dict], str]:
        print(f"\n{COLOR_TITLE}=== MULTI-AGENT FORWARD INFERENCE ==={COLOR_RESET}")
        timestamp = datetime.now()

        agent_results: Dict[str, Dict[str, str]] = {}
        strategies_used: Dict[str, Dict[str, List[str]]] = {}

        for role in self.agent_roles:
            print(f"\n{COLOR_INFO}--- Agent: {role} ---{COLOR_RESET}")
            trace = self.infer_agent_trace(role, env_state, belief_env_state)
            agent_results[role] = trace.to_dict()
            strategies_used[role] = trace.strategies_dict()

        # Save log
        filename = self.data_logger.save_inference(
            timestamp=timestamp,
            env_state=env_state,
            agent_results=agent_results,
            strategies_used=strategies_used,
            run_metadata=self.run_metadata()
        )

        print(f"\n{COLOR_SUCCESS}✅ Forward inference complete for {len(self.agent_roles)} agents{COLOR_RESET}")
        return agent_results, filename
