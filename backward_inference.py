
import json
import os
import time
from typing import Dict, Optional, Any
import traceback
import xml.etree.ElementTree as ET
from openai import OpenAI
import random
from datetime import datetime, timedelta
from cep import CognitiveEnhancementPlugin


last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 20.
DEFAULT_COOLDOWN = 20.0 
MAX_JITTER = 1.0


def rate_limit_api_call(func):
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


class BackwardInference:
    def __init__(self, 
                cep: CognitiveEnhancementPlugin, 
                llm_client: OpenAI,
                llm_model: str,
                backward_template_abs_path: str,
                inference_logs_abs_path: str,
                max_retries: int,
                base_delay_seconds: float,
                llm_temperature: float,
                llm_max_tokens: int):
        
        self.cep = cep
        self.llm_client = llm_client
        self.llm_model = llm_model 
        self.backward_template_abs_path = backward_template_abs_path
        self.inference_logs_abs_path = inference_logs_abs_path 
        self.max_retries = max_retries
        self.base_delay = base_delay_seconds
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        
        print(f"[BackwardInference] Initialized.")
        print(f"  Template Path: {self.backward_template_abs_path}")
        print(f"  Inference Logs Path: {self.inference_logs_abs_path}")

    def _load_inference_result(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.inference_logs_abs_path, filename)
        print(f"[BackwardInference] Loading inference log from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in ['mental_states', 'strategies_used', 'environmental_state']:
                if key not in data:
                    print(f"Warning: Log file {filename} missing expected key '{key}'.")
            return data
        except FileNotFoundError:
            print(f"Error: Inference result file not found at {filepath}")
            raise FileNotFoundError(f"Inference result file not found at {filepath}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in inference result file: {filepath}")
            raise 
        except Exception as e:
            print(f"Error loading inference file {filepath}: {str(e)}")
            raise

    def _load_prompt_template(self) -> str:
        print(f"[BackwardInference] Loading backward template from: {self.backward_template_abs_path}")
        try:
            with open(self.backward_template_abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Prompt template file not found: {self.backward_template_abs_path}")
            raise
        except Exception as e:
            print(f"Error reading prompt template file {self.backward_template_abs_path}: {str(e)}")
            raise

    def _get_strategy_text(self, level: str, strategy_id: Optional[str]) -> str:
        """Helper to get strategy text by ID, returns placeholder if not found."""
        if not strategy_id:
            return f"No {level} strategy used or ID missing"
        
        strategy_data = self.cep.get_strategy_by_id(level, strategy_id)
        if strategy_data and "item" in strategy_data and "strategy" in strategy_data["item"]:
            return strategy_data["item"]["strategy"]
        else:
            print(f"[Backward Inference Warning] Strategy text not found for ID: {strategy_id} in level: {level}")
            return f"Strategy text not found for {level} ID: {strategy_id}"

    def _rebuild_states_scenario(self, level: str, inference_data: Dict) -> Dict[str, str]:
        scenario = {}
        mental_states = inference_data.get('mental_states', {})
        env_state = inference_data.get('environmental_state', "[ENVIRONMENTAL STATE NOT FOUND]")
        # Define dependencies based on the CBN structure described in the prompt
        # Environmental States -> Belief
        # Belief -> Intention
        # Intention + Environmental States -> Emotion # Note: Prompt implies Belief -> Emotion, let's stick to code logic for now
        # Belief + Environmental States -> Emotion  <- Let's correct based on original code comment
        # Intention + Emotion -> Action
        try:
            if level == "belief":
                # Belief depends on Environmental
                scenario["environmental"] = env_state
            elif level == "intention":
                # Intention depends on Belief
                scenario["belief"] = mental_states.get('belief', "[BELIEF NOT FOUND]")
            elif level == "emotion":
                # Emotion depends on Belief and Environmental (as per original code logic)
                scenario["belief"] = mental_states.get('belief', "[BELIEF NOT FOUND]")
                scenario["environmental"] = env_state
            elif level == "action":
                # Action depends on Intention and Emotion
                scenario["intention"] = mental_states.get('intent', "[INTENTION NOT FOUND]") # Note: key in mental_states is 'intent'
                scenario["emotion"] = mental_states.get('emotion', "[EMOTION NOT FOUND]")
            else:
                print(f"Warning: Unknown level '{level}' for rebuilding scenario. Returning empty.")

        except KeyError as e:
             print(f"Warning: Missing key '{e}' in mental_states while rebuilding scenario for level '{level}'.")
             # Add placeholders for missing keys if needed, or just return partial scenario
             if level == "emotion":
                  if "belief" not in scenario: scenario["belief"] = "[BELIEF NOT FOUND]"
                  if "environmental" not in scenario: scenario["environmental"] = "[ENVIRONMENTAL STATE NOT FOUND]" # Added missing check
             if level == "action":
                  if "intention" not in scenario: scenario["intention"] = "[INTENTION NOT FOUND]"
                  if "emotion" not in scenario: scenario["emotion"] = "[EMOTION NOT FOUND]"
        except Exception as e:
            print(f"Error rebuilding scenario for level '{level}': {str(e)}")

        print(f"Rebuilt scenario for level '{level}': {list(scenario.keys())}") # Debug: Show keys
        return scenario

    def _process_llm_backward_response(self, llm_response: str, inference_data: Dict):
        print("\n--- Processing LLM Backward Response ---")
        inserted_count = 0
        updated_count = 0
        try:
            if not llm_response.strip().startswith("<MarketBackwardInference>"):
                llm_response = f"<MarketBackwardInference>{llm_response}</MarketBackwardInference>"

            root = ET.fromstring(llm_response)

            generated_strategies_node = root.find('GeneratedStrategies')
            if generated_strategies_node is not None:
                for strategy_node in generated_strategies_node.findall('Strategy'):
                    level = strategy_node.get('level')
                    content_node = strategy_node.find('GeneratedContent')
                    reasoning_node = strategy_node.find('Reasoning')

                    if level and content_node is not None and content_node.text:
                        content = content_node.text.strip()
                        reasoning = reasoning_node.text.strip() if reasoning_node is not None else ""
                        print(f"Found Generated Strategy for Level: {level}")

                        scenario = self._rebuild_states_scenario(level, inference_data)
                        if not scenario:
                            print(f"Skipping insertion for generated {level} strategy due to empty scenario.")
                            continue

                        try:
                            new_id = self.cep.insert_strategy(
                                level=level,
                                states_scenario=scenario,
                                strategy=content
                            )
                            if new_id:
                                print(f"Successfully Inserted New Strategy for {level}. New ID: {new_id}")
                                inserted_count += 1
                            else:
                                print(f"Insertion reported as failed for level {level} (new_id={new_id}). Check CEP logs or insert_strategy logic.")
                        except Exception as e:
                            print(f"Error calling insert_strategy for level {level}: {str(e)}")
                    else:
                        print("Warning: Skipping malformed generated strategy node in LLM response.")
            else:
                 print("No <GeneratedStrategies> section found in LLM response.")

            revised_strategies_node = root.find('RevisedStrategies')
            if revised_strategies_node is not None:
                 print("Processing <RevisedStrategies>...")
            else:
                 print("No <RevisedStrategies> section found in LLM response.")

        except ET.ParseError as e:
            print(f"Error parsing LLM XML response: {str(e)}")
            print("--- LLM Response Start ---")
            print(llm_response) # Print the raw response for debugging
            print("--- LLM Response End ---")
        except Exception as e:
            print(f"Error processing LLM backward response: {str(e)}")
            traceback.print_exc()

        print(f"--- Processing Complete: {inserted_count} strategies inserted, {updated_count} strategies updated ---")

    def perform_backward_inference(self, filename: str, predicted_action: str, actual_action: str) -> Optional[str]:
        log_data = self._load_inference_result(filename)
        if not log_data:
            return None

        try:
            env_state = log_data.get("environmental_state", "Environmental state missing")
            mental_states = log_data.get("mental_states", {})
            strategies_used = log_data.get("strategies_used", {})

            print(f"Debug: strategies_used content: {strategies_used}")

            belief_state = mental_states.get("belief", "Belief state missing")
            intent_state = mental_states.get("intent", "Intent state missing")
            emotion_state = mental_states.get("emotion", "Emotion state missing")

            belief_ids = strategies_used.get('belief', [])
            intent_ids = strategies_used.get('intent', [])
            emotion_ids = strategies_used.get('emotion', [])

            belief_strategy_text = self._get_strategy_text('belief', belief_ids[0]) if belief_ids else "No belief strategy used"
            intent_strategy_text = self._get_strategy_text('intent', intent_ids[0]) if intent_ids else "No intent strategy used"
            emotion_strategy_text = self._get_strategy_text('emotion', emotion_ids[0]) if emotion_ids else "No emotion strategy used"

            prompt_input = {
                "ENVIRONMENTAL_STATE": env_state,
                "BELIEF_STATE": belief_state,
                "INTENT_STATE": intent_state,
                "EMOTION_STATE": emotion_state,
                "PREDICTED_ACTION": predicted_action,
                "ACTUAL_ACTION": actual_action,
                "BELIEF_STRATEGY": belief_strategy_text,
                "INTENT_STRATEGY": intent_strategy_text,
                "EMOTION_STRATEGY": emotion_strategy_text
            }

            llm_analysis_text = self._call_backward_llm(prompt_input)

            if not llm_analysis_text:
                 print(f"LLM analysis failed for {filename}.")
                 return None

            self._process_llm_backward_response(llm_analysis_text, log_data)

            return llm_analysis_text

        except Exception as e:
             print(f"Error during backward inference for {filename}: {e}")
             import traceback
             traceback.print_exc()
             return None

    @rate_limit_api_call
    def _call_backward_llm(self, prompt_input: Dict[str, str]) -> Optional[str]:
        try:
            template = self._load_prompt_template()
            formatted_prompt = template
            for key, value in prompt_input.items():
                    str_value = str(value) if value is not None else f"[{key} VALUE MISSING]"
                    placeholder = f"[{key}]"
                    formatted_prompt = formatted_prompt.replace(placeholder, str_value)
        except Exception as e:
            print(f"[BackwardInference] Error formatting backward inference prompt: {e}")
            return None

        for attempt in range(self.max_retries):
            try:
                print("\n--- [BackwardInference] Calling LLM for Backward Inference ---")
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model, 
                    messages=[
                        {"role": "system", "content": formatted_prompt }
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                )
                print("--- [BackwardInference] LLM Call Successful ---")
                llm_content = response.choices[0].message.content.strip()
                return llm_content
            except Exception as e:
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5) 
                print(f"[BackwardInference] LLM API error (Attempt {attempt + 1}/{self.max_retries}): {str(e)}. Retrying in {delay:.2f}s...")
                if "400" in str(e) and "Invalid type for 'messages[0].content" in str(e):
                        print("    Hint: Check message content format for OpenAI API.")
                time.sleep(delay)
        print("[BackwardInference] Error: Failed to get LLM response after multiple retries.")
        return None
