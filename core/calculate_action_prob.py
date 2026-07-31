# -*- coding: utf-8 -*-
import json
import os
import time
import random
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import openai
from pydantic import BaseModel, Field

from core.cep import CognitiveEnhancementPlugin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_action.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.ActionProb')

COLOR_TITLE = "\033[1;36m"
COLOR_SUCCESS = "\033[1;32m"
COLOR_WARNING = "\033[1;33m"
COLOR_ERROR = "\033[1;31m"
COLOR_INFO = "\033[0;34m"
COLOR_VALUE = "\033[1;35m"
COLOR_DEBUG = "\033[0;90m"
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


# ---------- Response models ----------
class AgentActionResponse(BaseModel):
    agent_role: str = Field(...)
    predicted_action: str = Field(..., description="Buy or Sell")


@dataclass
class AgentPrediction:
    agent_role: str
    predicted_action: str  # "Buy" or "Sell"
    p_up: float = 0.5     # 1.0 if Buy, 0.0 if Sell
    log_confidence: float = 0.0  # C_k from logprobs
    weight: float = 0.0


class ProbabilityResult:
    def __init__(self, probability: float,
                 agent_predictions: Dict[str, Dict],
                 weights: Dict[str, float]):
        self.probability = probability
        self.agent_predictions = agent_predictions
        self.weights = weights


# ---------- Main calculator ----------
class ActionProbabilityCalculator:

    def __init__(self,
                 cep: CognitiveEnhancementPlugin = None,
                 llm_client: openai.OpenAI = None,
                 llm_model: str = "gpt-4o",
                 inference_logs_abs_path: str = "",
                 action_template_abs_path: str = "",
                 agent_roles: List[str] = None,
                 alpha: float = 1.0,
                 gamma: float = 1.0,
                 temperature: float = 1.0,
                 ema_decay: float = 0.9,
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 llm_temperature: float = 0.7,
                 ccn_dependency_variant: str = "full",
                 llm_extra_body: dict = None,
                 role_shuffle: bool = False,
                 # Legacy params (accepted but ignored)
                 expert_template_abs_path: str = None,
                 num_probs_to_generate: int = None,
                 action_prob_top_k: int = None,
                 max_retries_list: int = None,
                 base_delay_list_seconds: float = None,
                 kde_bandwidth_rule: str = None,
                 kde_min_bandwidth: float = None,
                 **kwargs):

        self.cep = cep
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.inference_logs_abs_path = inference_logs_abs_path
        self.agent_roles = agent_roles or DEFAULT_AGENT_ROLES
        self.alpha = alpha
        self.gamma = gamma
        self.agg_temperature = temperature
        self.ema_decay = ema_decay
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.llm_temperature = llm_temperature
        self.ccn_dependency_variant = ccn_dependency_variant
        self.llm_extra_body = llm_extra_body
        self.role_shuffle = role_shuffle

        # Template path: prefer new param, fall back to legacy
        self.template_path = action_template_abs_path or expert_template_abs_path or ""

        # EMA accuracy tracker per agent
        self._ema_accuracy: Dict[str, float] = {r: 0.5 for r in self.agent_roles}

    def _load_template(self) -> str:
        with open(self.template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_other_roles(self, agent_role: str) -> Tuple[str, str]:
        others = [r for r in self.agent_roles if r != agent_role]
        return (others[0], others[1]) if len(others) >= 2 else (others[0] if others else "Other", "Other")

    def load_inference_log(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.inference_logs_abs_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @rate_limit_api_call
    def _predict_agent_action(self, agent_role: str,
                              intent: str, emotion: str) -> Optional[AgentPrediction]:
        template = self._load_template()
        role2, role3 = self._get_other_roles(agent_role)

        prompt = template
        prompt = prompt.replace('[AGENT_ROLE]', agent_role)
        prompt = prompt.replace('[AGENT_ROLE_2]', role2)
        prompt = prompt.replace('[AGENT_ROLE_3]', role3)
        prompt = prompt.replace('[INTENTION_STATE]', intent)
        prompt = prompt.replace('[EMOTION_STATE]', emotion)

        for attempt in range(self.max_retries):
            try:
                request_kwargs = {
                    "model": self.llm_model,
                    "messages": [{"role": "system", "content": prompt}],
                    "temperature": self.llm_temperature,
                    "response_format": {"type": "json_object"},
                    "logprobs": True,
                    "top_logprobs": 5,
                }
                if self.llm_extra_body:
                    request_kwargs["extra_body"] = self.llm_extra_body
                try:
                    response = self.llm_client.chat.completions.create(**request_kwargs)
                except Exception as e:
                    if "logprobs" not in str(e).lower() and "top_logprobs" not in str(e).lower():
                        raise
                    request_kwargs.pop("logprobs", None)
                    request_kwargs.pop("top_logprobs", None)
                    response = self.llm_client.chat.completions.create(**request_kwargs)
                raw = response.choices[0].message.content.strip()
                data = json.loads(raw)
                parsed = AgentActionResponse.model_validate(data)

                action = parsed.predicted_action.strip()
                if action not in ("Buy", "Sell"):
                    action = "Buy" if "buy" in action.lower() else "Sell"

                # Extract log-confidence from logprobs
                log_conf = self._extract_action_log_confidence(
                    response.choices[0], action
                )

                pred = AgentPrediction(
                    agent_role=agent_role,
                    predicted_action=action,
                    p_up=1.0 if action == "Buy" else 0.0,
                    log_confidence=log_conf
                )
                return pred

            except json.JSONDecodeError:
                logger.warning(f"JSON parse error ({agent_role} action, attempt {attempt+1})")
            except Exception as e:
                logger.warning(f"Action prediction error ({agent_role}, attempt {attempt+1}): {e}")
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

        return None

    def _extract_action_log_confidence(self, choice, action: str) -> float:
        try:
            if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
                action_lower = action.lower()
                for token_info in choice.logprobs.content:
                    token_str = token_info.token.strip().strip('"').lower()
                    if token_str in (action_lower, "buy", "sell"):
                        return token_info.logprob
                    # Check top_logprobs
                    if hasattr(token_info, 'top_logprobs'):
                        for alt in token_info.top_logprobs:
                            alt_str = alt.token.strip().strip('"').lower()
                            if alt_str == action_lower:
                                return alt.logprob
        except Exception as e:
            logger.warning(f"Could not extract logprob: {e}")
        return math.log(0.5)

    @staticmethod
    def _softmax(values: List[float]) -> List[float]:
        max_v = max(values) if values else 0
        exps = [math.exp(v - max_v) for v in values]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def _dynamic_aggregate(self, predictions: List[AgentPrediction]) -> Tuple[float, Dict[str, float]]:
        if not predictions:
            return 0.5, {}

        scores = []
        for pred in predictions:
            A_k = self._ema_accuracy.get(pred.agent_role, 0.5)
            C_k = pred.log_confidence
            score = (self.alpha * A_k + self.gamma * C_k) / max(self.agg_temperature, 1e-6)
            scores.append(score)

        weights = self._softmax(scores)
        p_up = sum(w * pred.p_up for w, pred in zip(weights, predictions))

        weight_dict = {pred.agent_role: w for w, pred in zip(weights, predictions)}
        for pred, w in zip(predictions, weights):
            pred.weight = w

        return p_up, weight_dict

    def update_ema_accuracy(self, agent_role: str, correct: bool):
        current = self._ema_accuracy.get(agent_role, 0.5)
        self._ema_accuracy[agent_role] = (
            self.ema_decay * current + (1 - self.ema_decay) * (1.0 if correct else 0.0)
        )

    def _shuffle_agent_roles(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        present = [role for role in self.agent_roles if role in agent_results]
        if len(present) < 2:
            return agent_results
        sources = present[:]
        random.shuffle(sources)
        shuffled = dict(agent_results)
        for role, source in zip(present, sources):
            shuffled[role] = agent_results[source]
        return shuffled

    def calculate_probability_from_file(self, filename: str) -> ProbabilityResult:
        log_data = self.load_inference_log(filename)

        agent_results = log_data.get('agent_results', {})

        # Legacy compat: if old format has 'mental_states' but not 'agent_results'
        if not agent_results and 'mental_states' in log_data:
            ms = log_data['mental_states']
            # Treat as a single "Retail" agent for legacy data
            agent_results = {
                "Retail": {
                    "belief": ms.get("belief", ""),
                    "intent": ms.get("intent", ""),
                    "emotion": ms.get("emotion", "")
                }
            }

        if self.role_shuffle:
            agent_results = self._shuffle_agent_roles(agent_results)

        print(f"\n{COLOR_TITLE}=== MULTI-AGENT ACTION PREDICTION ==={COLOR_RESET}")

        predictions: List[AgentPrediction] = []

        for role in self.agent_roles:
            if role not in agent_results:
                continue
            states = agent_results[role]
            intent = states.get('intent', '')
            emotion = states.get('emotion', '')
            if self.ccn_dependency_variant == "no_intent_emotion_to_action":
                intent = "N/A"
                emotion = "N/A"

            if not intent and not emotion:
                logger.warning(f"No intent/emotion for {role}, skipping")
                continue

            print(f"  {COLOR_INFO}[{role}]{COLOR_RESET} Predicting action...")
            pred = self._predict_agent_action(role, intent, emotion)
            if not pred:
                print(f"  {COLOR_ERROR}✗ {role}: action prediction failed{COLOR_RESET}")
                continue

            predictions.append(pred)
            print(f"  {COLOR_SUCCESS}✓ {role}: {pred.predicted_action} "
                  f"(p_up={pred.p_up:.3f}, logconf={pred.log_confidence:.3f}){COLOR_RESET}")

        if not predictions:
            print(f"{COLOR_ERROR}No agent predictions available!{COLOR_RESET}")
            return ProbabilityResult(0.5, {}, {})

        # Dynamic weighted aggregation
        p_up, weight_dict = self._dynamic_aggregate(predictions)

        # Build per-agent prediction dict
        agent_preds = {}
        for pred in predictions:
            agent_preds[pred.agent_role] = {
                'predicted_action': pred.predicted_action,
                'p_up': pred.p_up,
                'log_confidence': pred.log_confidence,
                'weight': pred.weight
            }

        print(f"\n  {COLOR_VALUE}Aggregated P(up) = {p_up:.4f}{COLOR_RESET}")
        for role, w in weight_dict.items():
            print(f"    {role}: weight={w:.3f}")

        return ProbabilityResult(
            probability=p_up,
            agent_predictions=agent_preds,
            weights=weight_dict
        )
