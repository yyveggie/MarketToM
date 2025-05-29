
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
import openai
import random
from cep import CognitiveEnhancementPlugin
import traceback

last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 10.0
DEFAULT_COOLDOWN = 10.0
MAX_JITTER = 1.0

def rate_limit_api_call(func):
    """Decorator: Controls API call frequency to prevent rate limiting."""
    def wrapper(*args, **kwargs):
        global last_api_request_time
        
        now = datetime.now()
        time_since_last_request = (now - last_api_request_time).total_seconds()
        
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last_request + random.uniform(0, MAX_JITTER)
            print(f"Rate limiting: Waiting {wait_time:.2f}s before next API call...")
            time.sleep(wait_time)
        
        last_api_request_time = datetime.now()
        
        result = func(*args, **kwargs)
        
        cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
        print(f"API call completed. Cooling down for {cooldown:.2f}s...")
        time.sleep(cooldown)
        
        return result
    return wrapper

class DataLogger:
    """Data logger class."""
    def __init__(self, log_dir_abs_path: str):
        """Initialize DataLogger with an absolute path to the log directory."""
        self.log_dir = log_dir_abs_path
        print(f"DataLogger creating/using directory: {self.log_dir}")
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
                    fwd_inf_base_delay: int):
        self.cep = cep
        self.logger = logger
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.template_file_abs_path = forward_template_abs_path
        self.default_top_k = cep_default_top_k
        self.similarity_threshold = cep_similarity_threshold
        self.max_retries = fwd_inf_max_retries
        self.base_delay = fwd_inf_base_delay

    def _retrieve_strategies(self, state_type: str,
                               env_desc: str = None,
                               belief_desc: str = None,
                               top_k: int = None,
                               threshold_override: Optional[float] = None) -> List[Dict]:
        """Retrieve relevant strategies (uses configured top_k and threshold, allows threshold override)."""
        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_threshold = threshold_override if threshold_override is not None else self.similarity_threshold
        
        print(f"  [ForwardInfer DEBUG] Using threshold: {effective_threshold:.4f} for level '{state_type}' retrieval.")

        print(f"  [ForwardInfer DEBUG] Attempting to retrieve strategies for state_type: '{state_type}'")
        print(f"  [ForwardInfer DEBUG] Query context before CEP call: {env_desc if state_type == 'belief' else belief_desc if state_type == 'intent' else {'belief': belief_desc, 'environmental': env_desc} if state_type == 'emotion' else 'N/A'}")
        
        query_context = {}
        if state_type == "belief":
            query_context["environmental"] = env_desc
        elif state_type == "intent":
            query_context["belief"] = belief_desc
        elif state_type == "emotion":
            if belief_desc is None or env_desc is None:
                 print(f"  [ForwardInfer WARNING] Missing belief or environment description for emotion retrieval. Cannot retrieve.")
                 return []
            query_context = {"belief": belief_desc, "environmental": env_desc}

        print(f"  [ForwardInfer DEBUG] Constructed query_context for CEP: {query_context}")
        print(f"  [ForwardInfer DEBUG] Effective top_k: {effective_top_k}, Effective threshold: {effective_threshold:.4f}")

        if not any(query_context.values()):
             print(f"  [ForwardInfer WARNING] No valid context provided for '{state_type}' retrieval. Cannot retrieve.")
             return []

        strategies = self.cep.retrieve_strategies(
            state_type,
            query_context,
            top_k=effective_top_k,
            similarity_threshold=effective_threshold
        )

        print(f"  [ForwardInfer DEBUG] Raw strategies retrieved from CEP for '{state_type}': {strategies}")

        return strategies

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
        emotion_threshold = 0.65 # Specific threshold for emotion retrieval
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


    @rate_limit_api_call
    def _get_llm_response(self, user_prompt: str, state_type: str, preceding_state: str, strategies: str) -> str:
        """Get LLM response with rate limiting and retries."""
        try:
            template = self._load_prompt_template()
            system_content = template.replace('[DESCRIPTION OF THE PRECEDING NODE STATE]', preceding_state)
            system_content = system_content.replace('[STRATEGY RETRIEVED FROM CEP]', strategies)

            print("\n===== Debug: Generated System Prompt =====")
            print(system_content)
            print("===== End of System Prompt =====\n")
            print("\n===== Debug: Generated User Prompt =====")
            print(user_prompt)
            print("===== End of User Prompt =====\n")
        except Exception as e:
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            print(f"Fallback: Using basic fallback system prompt for state_type: {state_type}")
            system_content = f"You are a helpful assistant specializing in market {state_type}."

        max_retries = self.max_retries
        base_delay = self.base_delay

        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                )
                
                llm_content = response.choices[0].message.content.strip()
                
                start_tag = "<InferredMentalStateDescription>"
                end_tag = "</InferredMentalStateDescription>"
                start_index = llm_content.find(start_tag)
                end_index = llm_content.find(end_tag)
                
                if start_index != -1 and end_index != -1:
                    return llm_content[start_index + len(start_tag):end_index].strip()
                
                return llm_content

            except openai.RateLimitError as e:
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit reached. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)

        raise Exception("LLM call failed after multiple retries")

    def _load_prompt_template(self) -> str:
        """Load forward inference prompt template from absolute path."""
        print(f"[ForwardInfer] Loading forward template from: {self.template_file_abs_path}")
        try:
            with open(self.template_file_abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Prompt template file not found: {self.template_file_abs_path}")
            raise
        except Exception as e:
            print(f"Error reading prompt template file {self.template_file_abs_path}: {str(e)}")
            raise

    def forward_inference(self, env_state: str) -> Tuple[Dict, str]:
        """Execute the full forward inference process."""
        print("\nStarting Forward Inference...")
        print("Inferring Belief...")
        belief_desc, belief_ids = self.infer_market_belief(env_state)
        print(f"  -> Strategies used for Belief: {belief_ids}")
        
        print("Inferring Intent...")
        intent_desc, intent_ids = self.infer_market_intent(belief_desc)
        print(f"  -> Strategies used for Intent: {intent_ids}")
        
        print("Inferring Emotion...")
        emotion_desc, emotion_ids = self.infer_market_emotion(belief_desc, env_state)
        print(f"  -> Strategies used for Emotion: {emotion_ids}")
        
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
        print(f"Inference complete. Saving results to {filename}")
        
        self.logger.save_inference(
            timestamp,
            env_state,
            mental_states,
            strategies_used
        )
        
        return {
            'mental_states': mental_states,
            'strategies_used': strategies_used,
            'timestamp': timestamp.isoformat(),
            'environmental_state': env_state
        }, filename


