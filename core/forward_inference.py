# -*- coding: utf-8 -*-
import os
import json
import time
import random
import traceback
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
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
        
        def update(self, *args, **kwargs):
            pass
            
        def close(self, *args, **kwargs):
            pass
            
        def set_description(self, *args, **kwargs):
            pass
    
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

COLOR_TITLE = "\033[1;36m"     # Cyan bold (titles)
COLOR_SUCCESS = "\033[1;32m"   # Green bold (success)
COLOR_WARNING = "\033[1;33m"   # Yellow bold (warnings)
COLOR_ERROR = "\033[1;31m"     # Red bold (errors)
COLOR_INFO = "\033[0;34m"      # Blue (info)
COLOR_VALUE = "\033[1;35m"     # Magenta bold (values)
COLOR_DEBUG = "\033[0;90m"     # Gray (debug)
COLOR_PHASE = "\033[1;94m"     # Light blue bold (phases)
COLOR_RESET = "\033[0m"        # Reset color

last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 20.
DEFAULT_COOLDOWN = 20.0 
MAX_JITTER = 1.0


class MentalStateResponse(BaseModel):
    """Pydantic model for validating and parsing LLM responses"""
    mental_state_description: str = Field(..., description="Detailed description of the inferred market mental state")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mental_state_description": "The market is showing optimistic sentiment, with investors believing in positive short-term changes."
                }
            ]
        }
    }


def rate_limit_api_call(func):
    """Decorator: Controls API call frequency to prevent rate limiting."""
    def wrapper(*args, **kwargs):
        global last_api_request_time
        
        now = datetime.now()
        time_since_last_request = (now - last_api_request_time).total_seconds()
        
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last_request + random.uniform(0, MAX_JITTER)
            logger.info(f"Rate limiting: Waiting {wait_time:.2f}s before next API call")
            time.sleep(wait_time)
        
        last_api_request_time = datetime.now()
        
        result = func(*args, **kwargs)
        
        cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
        logger.info(f"API call completed. Cooling down for {cooldown:.2f}s")
        time.sleep(cooldown)
        
        return result
    return wrapper

class DataLogger:
    """Data logger class."""
    def __init__(self, log_dir_abs_path: str):
        """Initialize DataLogger with an absolute path to the log directory."""
        self.log_dir = log_dir_abs_path
        logger.info(f"DataLogger initialized with directory: {self.log_dir}")
        os.makedirs(self.log_dir, exist_ok=True)
        
    def save_inference(self, timestamp: datetime, 
                           env_state: str, 
                           mental_states: Dict[str, str],
                           strategies_used: Dict[str, List[str]]) -> None:
        """Save inference records."""
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "environmental_state": env_state,
            "mental_states": mental_states,
            "strategies_used": strategies_used
        }
        
        filename = f"inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved inference log to: {filepath}")


@dataclass
class EnvironmentalState:
    """Environmental state class."""
    quotes: pd.DataFrame
    texts: List[str]
    timestamp: datetime


class MentalStateInference:
    """Mental state inference class."""
    def __init__(self, 
                    cep: CognitiveEnhancementPlugin,
                    logger: DataLogger,
                    llm_client: openai.OpenAI,
                    llm_model: str,
                    forward_template_abs_path: str,
                    cep_default_top_k: int,
                    cep_similarity_threshold: float,
                    fwd_inf_max_retries: int,
                    fwd_inf_base_delay: int,
                    emotion_similarity_threshold: float = 0.1,
                    belief_similarity_threshold: float = 0.1,
                    intent_similarity_threshold: float = 0.1,
                    llm_temperature: float = 0.7):
        self.cep = cep
        self.logger = logger
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.template_file_abs_path = forward_template_abs_path
        self.default_top_k = cep_default_top_k
        self.similarity_threshold = cep_similarity_threshold
        self.max_retries = fwd_inf_max_retries
        self.base_delay = fwd_inf_base_delay
        self.emotion_similarity_threshold = emotion_similarity_threshold
        self.belief_similarity_threshold = belief_similarity_threshold
        self.intent_similarity_threshold = intent_similarity_threshold
        self.llm_temperature = llm_temperature

    def _retrieve_strategies(self, state_type: str,
                               env_desc: str = None,
                               belief_desc: str = None,
                               top_k: int = None,
                               threshold_override: Optional[float] = None) -> List[Dict]:
        """Retrieve the most relevant strategies for the given state type."""
        if top_k is None:
            top_k = self.default_top_k

        threshold = threshold_override if threshold_override is not None else self.similarity_threshold
        
        query_dict = {}
        if env_desc is not None:
            query_dict['environmental'] = env_desc
        if belief_desc is not None:
            query_dict['belief'] = belief_desc
            
        logger.info(f"Retrieving {state_type} strategies (threshold:{threshold:.2f})")
        retrieved_strategies = self.cep.retrieve_strategies(
            level=state_type,
            scenarios=query_dict,
            top_k=top_k,
            similarity_threshold=threshold
        )
        
        if not retrieved_strategies:
            logger.warning(f"No {state_type} strategies retrieved")
        else:
            ids = [s['item'].get('id', 'unknown') for s in retrieved_strategies]
            logger.info(f"Retrieved {len(retrieved_strategies)} {state_type} strategies: {ids}")
            
        return retrieved_strategies

    def infer_market_belief(self, env_state: str,
                            top_k: int = None) -> Tuple[str, List[str]]:
        """Infer market belief state."""
        retrieved_strategy_objects = self._retrieve_strategies("belief", env_desc=env_state, top_k=top_k)
        
        strategy_text_parts = []
        if retrieved_strategy_objects:
            for i, strat_obj in enumerate(retrieved_strategy_objects):
                try:
                    strategy_content = strat_obj.get('item', {}).get('strategy')
                    if strategy_content:
                        strategy_text_parts.append(f"{i + 1}. {strategy_content}")
                    else:
                        strategy_text_parts.append(f"{i + 1}. [Strategy content not found in expected item.strategy structure]")
                except Exception as e:
                    strategy_text_parts.append(f"{i + 1}. [Error processing strategy object: {str(e)}]")

        if not strategy_text_parts:
            strategies_for_prompt = "Retrieved Strategies:\nNo specific strategies were retrieved or applicable to the current situation."
        else:
            strategies_for_prompt = "Retrieved Strategies:\n" + "\n".join(strategy_text_parts)
            
        user_prompt_text = "Please perform the market belief inference based on the system instructions and the data provided therein. Focus on identifying the most likely belief state."
        
        response = self._get_llm_response(user_prompt_text, "belief", env_state, strategies_for_prompt)
        
        strategy_ids = []
        if retrieved_strategy_objects:
            for s_obj in retrieved_strategy_objects:
                if s_obj and isinstance(s_obj.get('item'), dict) and 'id' in s_obj['item']:
                    strategy_ids.append(s_obj['item']['id'])
        return response, strategy_ids

    def infer_market_intent(self, belief: str,
                            top_k: int = None) -> Tuple[str, List[str]]:
        """Infer market intent state."""
        retrieved_strategy_objects = self._retrieve_strategies("intent", belief_desc=belief, top_k=top_k)
        
        strategy_text_parts = []
        if retrieved_strategy_objects:
            for i, strat_obj in enumerate(retrieved_strategy_objects):
                try:
                    strategy_content = strat_obj.get('item', {}).get('strategy')
                    if strategy_content:
                        strategy_text_parts.append(f"{i + 1}. {strategy_content}")
                    else:
                        strategy_text_parts.append(f"{i + 1}. [Strategy content not found in expected item.strategy structure]")
                except Exception as e:
                    strategy_text_parts.append(f"{i + 1}. [Error processing strategy object: {str(e)}]")

        if not strategy_text_parts:
            strategies_for_prompt = "Retrieved Strategies:\nNo specific strategies were retrieved or applicable to the current situation."
        else:
            strategies_for_prompt = "Retrieved Strategies:\n" + "\n".join(strategy_text_parts)
            
        user_prompt_text = "Please perform the market intent inference based on the system instructions and the data provided therein. Focus on identifying the most likely intent state given the belief."
        
        response = self._get_llm_response(user_prompt_text, "intent", belief, strategies_for_prompt)
        
        strategy_ids = []
        if retrieved_strategy_objects:
            for s_obj in retrieved_strategy_objects:
                if s_obj and isinstance(s_obj.get('item'), dict) and 'id' in s_obj['item']:
                    strategy_ids.append(s_obj['item']['id'])
        return response, strategy_ids

    def infer_market_emotion(self, belief: str,
                             env_state: str,
                             top_k: int = None) -> Tuple[str, List[str]]:
        """Infer market emotion state."""
        emotion_threshold = self.emotion_similarity_threshold  
        retrieved_strategy_objects = self._retrieve_strategies("emotion",
                                               env_desc=env_state,
                                               belief_desc=belief,
                                               top_k=top_k,
                                               threshold_override=emotion_threshold)
        
        strategy_text_parts = []
        if retrieved_strategy_objects:
            for i, strat_obj in enumerate(retrieved_strategy_objects):
                try:
                    strategy_content = strat_obj.get('item', {}).get('strategy')
                    if strategy_content:
                        strategy_text_parts.append(f"{i + 1}. {strategy_content}")
                    else:
                        strategy_text_parts.append(f"{i + 1}. [Strategy content not found in expected item.strategy structure]")
                except Exception as e:
                    strategy_text_parts.append(f"{i + 1}. [Error processing strategy object: {str(e)}]")

        if not strategy_text_parts:
            strategies_for_prompt = "Retrieved Strategies:\nNo specific strategies were retrieved or applicable to the current situation."
        else:
            strategies_for_prompt = "Retrieved Strategies:\n" + "\n".join(strategy_text_parts)

        preceding_data_for_emotion = (
            f"Current Market Belief:\\n{belief}\\n\\n"
            f"Current Environmental State:\\n{env_state}"
        )
        
        user_prompt_text = "Please perform the market emotion inference based on the system instructions and the data provided therein. Consider all relevant preceding states as per the CBN model for emotion."
        
        response = self._get_llm_response(user_prompt_text, "emotion", preceding_data_for_emotion, strategies_for_prompt)

        strategy_ids = []
        if retrieved_strategy_objects:
            for s_obj in retrieved_strategy_objects:
                if s_obj and isinstance(s_obj.get('item'), dict) and 'id' in s_obj['item']:
                    strategy_ids.append(s_obj['item']['id'])
        return response, strategy_ids

    def _get_llm_response(self, user_prompt: str, state_type: str, preceding_state: str, strategies: str) -> str:
        """Get LLM response with Pydantic validation, returns mental state description"""
        try:
            template = self._load_prompt_template()
        except Exception as e:
            logger.error(f"Error loading template: {e}")
            template = "You are an expert market analyst who specializes in [STATE_TYPE] inference. Analyze the data and provide your best inference."

        try:
            system_content = template
            system_content = system_content.replace('[STRATEGIES]', strategies)
            system_content = system_content.replace('[PRECEDING_STATE]', preceding_state)
            system_content = system_content.replace('[STATE_TYPE]', state_type.upper())

        except Exception as e:
            logger.error(f"Error processing template: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.warning(f"Using fallback system prompt for state_type: {state_type}")
            system_content = f"You are a helpful assistant specializing in market {state_type}."

        max_retries = self.max_retries
        base_delay = self.base_delay

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending request to LLM (attempt {attempt+1}/{max_retries})...")
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.llm_temperature,
                    response_format={"type": "json_object"}
                )
                
                llm_content = response.choices[0].message.content.strip()
                logger.info(f"Successfully received LLM response ({len(llm_content)} characters)")
                
                try:
                    response_json = json.loads(llm_content)
                except json.JSONDecodeError as json_err:
                    print(f"{COLOR_ERROR}LLM {state_type} raw response (non-JSON):{COLOR_RESET} {llm_content}")
                    logger.error(f"Could not parse LLM response as JSON: {json_err}")
                    raise
                
                # Validate with Pydantic
                try:
                    validated_response = MentalStateResponse.model_validate(response_json)
                    description = validated_response.mental_state_description
                    print(f"{COLOR_INFO}LLM {state_type} mental state:{COLOR_RESET} {COLOR_VALUE}{description}{COLOR_RESET}")
                    return description
                except Exception as e:
                    logger.warning(f"Pydantic validation failed: {str(e)[:100]}")
                    # Fallback: try both possible key names
                    if "mental_state_description" in response_json:
                        description = response_json["mental_state_description"]
                        print(f"{COLOR_INFO}LLM {state_type} mental state:{COLOR_RESET} {COLOR_VALUE}{description}{COLOR_RESET}")
                        return description
                    elif "mental state description" in response_json:
                        description = response_json["mental state description"]
                        print(f"{COLOR_INFO}LLM {state_type} mental state:{COLOR_RESET} {COLOR_VALUE}{description}{COLOR_RESET}")
                        return description
                    else:
                        print(f"{COLOR_ERROR}LLM {state_type} raw response (missing description field):{COLOR_RESET} {response_json}")
                        logger.error(f"No description field found in response: {list(response_json.keys())}")
                        raise ValueError(f"Invalid response format: missing description field")

            except openai.RateLimitError as e:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)

        raise Exception("LLM call failed after multiple retries")

    def _load_prompt_template(self) -> str:
        """Load forward inference prompt template from absolute path."""
        logger.info(f"Loading forward template from: {self.template_file_abs_path}")
        try:
            with open(self.template_file_abs_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                
                has_json_format = "JSON" in template_content and "mental_state_description" in template_content
                
                if has_json_format:
                    logger.info("Template includes JSON output format guidelines")
                else:
                    logger.warning("Template may be missing JSON output format guidelines")
                
                return template_content
        except FileNotFoundError:
            logger.error(f"Error: Prompt template file not found: {self.template_file_abs_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading prompt template file {self.template_file_abs_path}: {str(e)}")
            raise

    def forward_inference(self, env_state: str) -> Tuple[Dict, str]:
        """Execute the full forward inference process."""
        print(f"\n{COLOR_TITLE}=== ANALYZING Market mental state ===={COLOR_RESET}")
        
        logger.info("Checking strategy database counts")
        for level in ["belief", "intent", "emotion"]:
            strategies = self.cep.get_strategies_by_level(level)
            logger.info(f"{level.capitalize()} strategies count: {len(strategies)}")
        
        # Step 1: Market Belief Analysis
        print(f"\n{COLOR_PHASE}STEP 1: Analyzing market beliefs...{COLOR_RESET}")
        belief_desc, belief_ids = self.infer_market_belief(env_state)
        if belief_ids:
            logger.info(f"Belief strategy IDs: {belief_ids}")
            print(f"{COLOR_SUCCESS}✓ Found relevant market belief patterns{COLOR_RESET}")
        else:
            logger.warning("No belief strategies retrieved")
        
        print(f"{COLOR_VALUE}Market Belief:{COLOR_RESET}")
        print(f"{COLOR_INFO}{belief_desc}{COLOR_RESET}")
        if belief_ids:
            print(f"{COLOR_VALUE}Belief Strategies: {COLOR_RESET}{COLOR_DEBUG}{', '.join(belief_ids)}{COLOR_RESET}")
        print()
        
        # Step 2: Market Intent Analysis  
        print(f"\n{COLOR_PHASE}STEP 2: Analyzing market intentions...{COLOR_RESET}")
        intent_desc, intent_ids = self.infer_market_intent(belief_desc)
        if intent_ids:
            logger.info(f"Intent strategy IDs: {intent_ids}")
            print(f"{COLOR_SUCCESS}✓ Identified market intentions{COLOR_RESET}")
        else:
            logger.warning("No intent strategies retrieved")
        
        print(f"{COLOR_VALUE}Market Intent:{COLOR_RESET}")
        print(f"{COLOR_INFO}{intent_desc}{COLOR_RESET}")
        if intent_ids:
            print(f"{COLOR_VALUE}Intent Strategies: {COLOR_RESET}{COLOR_DEBUG}{', '.join(intent_ids)}{COLOR_RESET}")
        print()
        
        # Step 3: Market Emotion Analysis
        print(f"\n{COLOR_PHASE}STEP 3: Analyzing market emotion...{COLOR_RESET}")
        emotion_desc, emotion_ids = self.infer_market_emotion(belief_desc, env_state)
        if emotion_ids:
            logger.info(f"Emotion strategy IDs: {emotion_ids}")
            print(f"{COLOR_SUCCESS}✓ Determined market emotion{COLOR_RESET}")
        else:
            logger.warning("No emotion strategies retrieved")
        
        print(f"{COLOR_VALUE}Market Emotion:{COLOR_RESET}")
        print(f"{COLOR_INFO}{emotion_desc}{COLOR_RESET}")
        if emotion_ids:
            print(f"{COLOR_VALUE}Emotion Strategies: {COLOR_RESET}{COLOR_DEBUG}{', '.join(emotion_ids)}{COLOR_RESET}")
        print()
        
        mental_states = {
            'belief': belief_desc,
            'intent': intent_desc,
            'emotion': emotion_desc
        }
        
        strategies_used = {
            'belief': belief_ids,
            'intent': intent_ids,
            'emotion': emotion_ids
        }

        timestamp = datetime.now()
        filename = f"inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        self.logger.save_inference(
            timestamp=timestamp,
            env_state=env_state,
            mental_states=mental_states,
            strategies_used=strategies_used
        )
        
        return mental_states, filename
