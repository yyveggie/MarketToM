# coding=utf-8

from typing import List, Dict, Any, Optional
import json
import os
import numpy as np
from scipy.stats import gaussian_kde
from pydantic import BaseModel, Field, field_validator
import openai # Used for the main API call via module-level functions
import time
from cep import CognitiveEnhancementPlugin
import re
from datetime import datetime, timedelta # timedelta used by rate_limit_api_call
import random # used by rate_limit_api_call
import traceback # Used for detailed error logging

# --- API请求限速控制 ---
# Note: This global state might cause issues if multiple instances are used concurrently.
last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 20.0
DEFAULT_COOLDOWN = 20.0
MAX_JITTER = 2.0

def rate_limit_api_call(func):
    """装饰器：控制API调用频率，防止触发速率限制"""
    def wrapper(*args, **kwargs):
        global last_api_request_time
        now = datetime.now()
        time_since_last_request = (now - last_api_request_time).total_seconds()
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last_request + random.uniform(0, MAX_JITTER)
            print(f"Rate limiting: Waiting {wait_time:.2f}s before next API call...")
            time.sleep(wait_time)
        last_api_request_time = datetime.now()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            # Ensure cooldown happens even if function fails
            cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
            print(f"API call failed or errored. Cooling down for {cooldown:.2f}s...")
            time.sleep(cooldown)
            raise e # Re-raise the original exception
        else:
            # Cooldown always after attempt
            cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
            print(f"API call completed. Cooling down for {cooldown:.2f}s...")
            time.sleep(cooldown)
            return result
    return wrapper

# --- Pydantic Models (No changes needed) ---
class TokenInfo(BaseModel):
    """Token信息模型"""
    token: str
    logit: float
    @field_validator('logit')
    @classmethod
    def validate_logit(cls, v):
        if not np.isfinite(v):
            raise ValueError("logit必须是有限数值")
        return v
    @field_validator('token')
    @classmethod
    def validate_token(cls, v):
        return v # Allow empty strings

class ProbabilityEntry(BaseModel):
    """概率条目模型"""
    raw_text: str
    tokens: List[TokenInfo]
    log_confidence: float
    confidence: float
    normalized_weight: float
    parsed_probability: float
    weighted_probability: float

class DensityEstimation(BaseModel):
    """密度估计结果模型"""
    x: List[float]
    density: List[float]
    bandwidth: float

class ProbabilityResult(BaseModel):
    """最终概率计算结果模型"""
    probability: float
    strategy_ids: List[str]
    probability_entries: List[ProbabilityEntry]
    density_estimation: DensityEstimation
    inference_id: str
    timestamp: str
    environmental_state: str

# --- Main Calculator Class (Refactored) ---
class ActionProbabilityCalculator:
    """市场行为概率计算器 (配置通过 __init__ 传入)"""
    def __init__(self,
                 cep: CognitiveEnhancementPlugin,
                 llm_client: openai.OpenAI,
                 llm_model: str,
                 action_prob_template_abs_path: str, # 模板文件的绝对路径
                 inference_logs_abs_path: str,    # 推理日志目录的绝对路径
                 action_prob_top_k: int,
                 num_probs_to_generate: int,
                 max_retries_list: int,           # LLM调用重试次数(若不用装饰器)
                 base_delay_list_seconds: float,  # LLM调用重试基础延迟(若不用装饰器)
                 kde_bandwidth_rule: str = 'silverman', # KDE带宽规则
                 kde_min_bandwidth: float = 0.01      # KDE最小带宽
                 ):
        self.cep = cep
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.action_prob_template_abs_path = action_prob_template_abs_path
        self.inference_logs_abs_path = inference_logs_abs_path
        self.action_prob_top_k = action_prob_top_k
        self.num_probs_to_generate = num_probs_to_generate
        self.max_retries_list = max_retries_list # 可用于内部重试逻辑，目前主要靠装饰器
        self.base_delay_list = base_delay_list_seconds # 可用于内部重试逻辑
        self.kde_bandwidth_rule = kde_bandwidth_rule
        self.kde_min_bandwidth = kde_min_bandwidth
        print(f"[ActionProbCalc] Initialized.")
        print(f"  Template Path: {self.action_prob_template_abs_path}")
        print(f"  Inference Logs Dir: {self.inference_logs_abs_path}")

    def _load_prompt_template(self) -> str:
        """从绝对路径加载行为概率提示词模板"""
        print(f"[ActionProbCalc] Loading action prob template from: {self.action_prob_template_abs_path}")
        try:
            with open(self.action_prob_template_abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Prompt template file not found: {self.action_prob_template_abs_path}")
            raise
        except Exception as e:
            print(f"Error reading prompt template file {self.action_prob_template_abs_path}: {str(e)}")
            raise

    def load_inference_log(self, filename: str) -> Dict[str, Any]:
        """从已知的绝对路径的 inference_logs 文件夹加载推理日志"""
        filepath = os.path.join(self.inference_logs_abs_path, filename)
        print(f"[ActionProbCalc] Loading inference log from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'mental_states' not in data:
                raise ValueError(f"File {filepath} is missing 'mental_states' field.")
            if 'intent' not in data['mental_states'] or 'emotion' not in data['mental_states']:
                raise ValueError(f"File {filepath} 'mental_states' is missing 'intent' or 'emotion'.")
            return data
        except FileNotFoundError:
             print(f"Error: Inference log file not found at {filepath}")
             raise FileNotFoundError(f"Inference log file not found at {filepath}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in file {filepath}: {str(e)}")
            raise
        except Exception as e:
            print(f"Failed to read inference log file {filepath}: {str(e)}")
            raise

    def calculate_probability_from_file(self, filename: str) -> ProbabilityResult:
        """从推理日志文件计算行为概率"""
        print(f"\n--- [ActionProbCalc] Starting probability calculation for: {filename} ---")
        try:
            data = self.load_inference_log(filename)
            mental_states = data.get('mental_states', {})
            intent_desc = mental_states.get('intent', '')
            emotion_desc = mental_states.get('emotion', '')
            if not intent_desc or not emotion_desc:
                raise ValueError("Missing 'intent' or 'emotion' state in inference log.")

            print(f"  Retrieved Intent: {intent_desc[:100]}...")
            print(f"  Retrieved Emotion: {emotion_desc[:100]}...")

            print(f"  Retrieving action strategies from CEP (top_k={self.action_prob_top_k})...")
            strategy_objects = self.cep.retrieve_strategies(
                "action",
                {"intent": intent_desc, "emotion": emotion_desc},
                top_k=self.action_prob_top_k
            )
            print(f"  Retrieved {len(strategy_objects)} action strategies.")

            preceding_node_state_desc_for_template = (
                f"Current Market Intent:\n{intent_desc}\n\n"
                f"Current Market Emotion:\n{emotion_desc}"
            )

            strategies_str_for_template = "No relevant strategies retrieved."
            if strategy_objects:
                strategy_details = []
                for i, strategy_item_wrapper in enumerate(strategy_objects, 1):
                    strategy_item = strategy_item_wrapper.get('item', {})
                    scenario_data = strategy_item.get('states_scenario', {})
                    strategy_text = strategy_item.get('strategy', 'N/A')
                    outcome = strategy_item.get('outcome', 'Unknown')
                    strategy_details.append(
                        f"Strategy {i}:\n"
                        f"  Scenario: {json.dumps(scenario_data, ensure_ascii=False, indent=2)}\n"
                        f"  Strategy Text: {strategy_text}\n"
                        f"  Historical Outcome: {outcome}" # Assuming outcome exists
                    )
                strategies_str_for_template = "\n".join(strategy_details)
            # print(f"  Formatted Strategies for Template:\n{strategies_str_for_template}") # Optional debug

            retrieved_strategy_ids = [s.get('item', {}).get('id', f'unknown_id_{idx}') for idx, s in enumerate(strategy_objects)]

            print("  Calling core probability calculation logic...")
            result = self.calculate_action_probability(
                preceding_node_state_for_template=preceding_node_state_desc_for_template,
                strategies_for_template=strategies_str_for_template,
                retrieved_strategy_ids=retrieved_strategy_ids
            )

            # Populate remaining fields in the result
            result.inference_id = filename.replace('.json', '')
            result.timestamp = data.get('timestamp', datetime.now().isoformat())
            result.environmental_state = data.get('environmental_state', '[Environmental State Missing]') # Get from log
            print(f"--- [ActionProbCalc] Probability calculation successful for {filename}. Final P(up): {result.probability:.4f} ---")
            return result

        except Exception as e:
            print(f"!! Error calculating probability from file '{filename}': {str(e)}")
            traceback.print_exc() # Print full traceback for better debugging
            raise # Re-raise to allow run.py to handle it

    def parse_probability_text(self, text: str) -> float:
        """解析概率文本为浮点数 (0-1 range)"""
        if not isinstance(text, str): # Handle non-string input gracefully
             print(f"Warning: parse_probability_text received non-string input: {type(text)}. Returning 0.5.")
             return 0.5
        try:
            # Keep only digits and decimal point
            cleaned_text = ''.join(filter(lambda char: char.isdigit() or char == '.', text))
            # Handle cases like "" or "."
            if not cleaned_text or cleaned_text == '.':
                 print(f"Warning: Invalid probability string after cleaning: '{text}' -> '{cleaned_text}'. Returning 0.5.")
                 return 0.5
            value = float(cleaned_text)
            # Handle percentage values
            if value > 1.0 and value <= 100.0:
                print(f"Info: Interpreting value {value} as percentage, converting to {value/100.0}")
                value /= 100.0
            # Clamp between 0 and 1
            return min(max(value, 0.0), 1.0)
        except ValueError:
            print(f"Warning: Could not parse probability string to float: '{text}'. Returning 0.5.")
            return 0.5

    def calculate_action_probability(self,
                                     preceding_node_state_for_template: str,
                                     strategies_for_template: str,
                                     retrieved_strategy_ids: List[str]
                                    ) -> ProbabilityResult:
        """核心计算逻辑: 调用LLM获取带logprobs的概率列表, 处理数据, 进行KDE和期望值计算"""
        print("  Inside calculate_action_probability core logic...")
        try:
            tokens_and_logits_list = self._get_probability_tokens(
                preceding_node_state_for_template,
                strategies_for_template
            )

            if not tokens_and_logits_list:
                 print("Error: _get_probability_tokens returned empty list. Cannot proceed.")
                 raise ValueError("LLM did not return valid probability tokens.")

            probability_entries = []
            print("  Processing LLM response tokens...")
            for idx, single_prob_token_list in enumerate(tokens_and_logits_list):
                if not single_prob_token_list:
                    print(f"Warning: Empty token list for probability entry #{idx+1}, skipping.")
                    continue

                # Calculate log confidence robustly
                valid_logits = [token.logit for token in single_prob_token_list if np.isfinite(token.logit)]
                if not valid_logits:
                     print(f"Warning: No finite logits found for probability entry #{idx+1}, assigning low confidence.")
                     log_confidence = -100.0 # Assign a very low log confidence
                else:
                    # log(1+exp(x)) can overflow for large x. Use stable calculation if needed, but logaddexp(0,x) is generally stable.
                     log_confidence = sum(np.logaddexp(0, logit) for logit in valid_logits)

                confidence = np.exp(log_confidence)
                raw_text = ''.join(token.token for token in single_prob_token_list).strip()
                parsed_prob = self.parse_probability_text(raw_text)
                print(f"    Entry #{idx+1}: Raw='{raw_text}', Parsed(phi)={parsed_prob:.4f}, LogConf={log_confidence:.4f}, Conf={confidence:.4e}")

                entry = ProbabilityEntry(
                    raw_text=raw_text, tokens=single_prob_token_list,
                    log_confidence=log_confidence, confidence=confidence,
                    normalized_weight=0.0, # Calculated next
                    parsed_probability=parsed_prob,
                    weighted_probability=0.0 # Calculated next
                )
                probability_entries.append(entry)

            if not probability_entries:
                print("Error: No valid probability entries could be processed.")
                raise ValueError("No valid probability entries processed.")

            # --- Normalize weights and calculate weighted probabilities ---
            finite_confidences = [entry.confidence for entry in probability_entries if np.isfinite(entry.confidence) and entry.confidence > 0]
            total_finite_confidence = sum(finite_confidences)

            print(f"  Total finite confidence for normalization: {total_finite_confidence:.4e}")

            if total_finite_confidence <= 1e-9:
                print("Warning: Total finite confidence is near zero. Using uniform weights.")
                num_entries = len(probability_entries)
                uniform_weight = 1.0 / num_entries if num_entries > 0 else 0.0
                for entry in probability_entries:
                     entry.normalized_weight = uniform_weight
                     # Use the already parsed probability
                     entry.weighted_probability = entry.parsed_probability * uniform_weight
            else:
                 for entry in probability_entries:
                      if np.isfinite(entry.confidence) and entry.confidence > 0:
                           entry.normalized_weight = entry.confidence / total_finite_confidence
                      else:
                           entry.normalized_weight = 0.0 # Assign zero weight if confidence is non-finite or zero
                      entry.weighted_probability = entry.parsed_probability * entry.normalized_weight

            # --- Prepare data for KDE ---
            updated_entries = []
            parsed_probabilities_for_kde = []
            weights_for_kde = []
            print("\n  Preparing data for KDE...")
            for entry in probability_entries:
                # Only use entries with finite parsed probability and weight for KDE calculation
                if np.isfinite(entry.parsed_probability) and np.isfinite(entry.normalized_weight):
                    print(f"    Adding to KDE: Prob={entry.parsed_probability:.4f}, Weight={entry.normalized_weight:.4f}")
                    parsed_probabilities_for_kde.append(entry.parsed_probability)
                    weights_for_kde.append(entry.normalized_weight)
                else:
                    print(f"    Skipping entry for KDE: Raw='{entry.raw_text}', Parsed={entry.parsed_probability}, Weight={entry.normalized_weight}")
                updated_entries.append(entry) # Keep all processed entries in the result

            # --- Perform KDE and calculate expected value ---
            if not parsed_probabilities_for_kde or len(parsed_probabilities_for_kde) < 2:
                print("Warning: Not enough valid finite data points for KDE. Calculating simple weighted average.")
                # Fallback: Calculate weighted average if possible, otherwise simple average or default
                valid_weights = [w for w in weights_for_kde if np.isfinite(w)]
                valid_probs = [p for p, w in zip(parsed_probabilities_for_kde, weights_for_kde) if np.isfinite(p) and np.isfinite(w)]

                if valid_probs and valid_weights and sum(valid_weights) > 1e-6:
                     final_probability = np.average(valid_probs, weights=valid_weights)
                     print(f"  Calculated weighted average: {final_probability:.4f}")
                elif valid_probs:
                     final_probability = np.mean(valid_probs)
                     print(f"  Calculated simple average: {final_probability:.4f}")
                else:
                     final_probability = 0.5
                     print("  No valid data for average, defaulting to 0.5.")
                final_probability = min(max(final_probability, 0.0), 1.0)
                density_estimation = DensityEstimation(x=[0.0, 1.0], density=[1.0, 1.0], bandwidth=0.1) # Default uniform density
            else:
                print(f"  Performing KDE with {len(parsed_probabilities_for_kde)} points...")
                # Ensure weights sum to 1 for KDE (scipy handles this, but good practice to check)
                weight_sum = sum(weights_for_kde)
                if abs(weight_sum - 1.0) > 1e-6 and weight_sum > 1e-9:
                     print(f"  Normalizing weights for KDE (Sum={weight_sum:.4f}).")
                     weights_for_kde = [w / weight_sum for w in weights_for_kde]
                elif weight_sum <= 1e-9:
                     print("  Weights sum is near zero, performing unweighted KDE.")
                     weights_for_kde = None # Let KDE handle unweighted case

                density_estimation = self._kernel_density_estimation(parsed_probabilities_for_kde, weights=weights_for_kde)
                final_probability = self._calculate_upward_probability(density_estimation)

            print(f"  Final calculated probability P(up): {final_probability:.4f}")
            # Return result (caller fills in metadata)
            return ProbabilityResult(
                probability=final_probability,
                strategy_ids=retrieved_strategy_ids,
                probability_entries=updated_entries,
                density_estimation=density_estimation,
                inference_id="", # Placeholder
                timestamp="",    # Placeholder
                environmental_state="" # Placeholder
            )
        except ValueError as ve:
            print(f"!! ValueError in calculate_action_probability core: {ve}")
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"!! Unexpected error in calculate_action_probability core: {e}")
            traceback.print_exc()
            raise

    def _kernel_density_estimation(self, dataset: List[float], weights: Optional[List[float]] = None) -> DensityEstimation:
        """使用高斯核进行密度估计"""
        print(f"  Inside KDE: {len(dataset)} points.")
        if not dataset or len(dataset) < 2:
            print("  KDE requires at least 2 data points. Returning default uniform density.")
            return DensityEstimation(x=[0.0, 1.0], density=[1.0, 1.0], bandwidth=0.1)
        try:
            dataset_np = np.array(dataset)
            weights_np = np.array(weights) if weights is not None else None

            if weights_np is not None and len(weights_np) != len(dataset_np):
                print(f"  Warning: KDE weights length ({len(weights_np)}) mismatch dataset length ({len(dataset_np)}). Using unweighted.")
                weights_np = None
            # Check for non-finite weights if provided
            if weights_np is not None and not np.all(np.isfinite(weights_np)):
                print("  Warning: Non-finite weights detected. Using unweighted KDE.")
                weights_np = None
             # Check if all weights are zero if provided
            if weights_np is not None and np.sum(weights_np) <= 1e-9:
                 print("  Warning: Sum of weights is near zero. Using unweighted KDE.")
                 weights_np = None

            # Let scipy choose bandwidth initially
            try:
                kde = gaussian_kde(dataset_np, bw_method=self.kde_bandwidth_rule, weights=weights_np)
                bandwidth = kde.factor
            except Exception as kde_init_e: # Catch potential errors during KDE init (e.g., singular matrix)
                print(f"  Error initializing KDE with rule '{self.kde_bandwidth_rule}': {kde_init_e}. Trying default bandwidth rule.")
                try:
                    kde = gaussian_kde(dataset_np, weights=weights_np) # Try default rule
                    bandwidth = kde.factor
                except Exception as kde_fallback_e:
                    print(f"  Error initializing KDE with default rule: {kde_fallback_e}. Returning uniform density.")
                    return DensityEstimation(x=[0.0, 1.0], density=[1.0, 1.0], bandwidth=0.1)


            # Apply minimum bandwidth constraint
            effective_bw = max(bandwidth, self.kde_min_bandwidth)
            if abs(effective_bw - bandwidth) > 1e-6:
                print(f"  Adjusting KDE bandwidth from {bandwidth:.4f} to minimum {self.kde_min_bandwidth:.4f}")
                try:
                    # Re-initialize KDE with the adjusted bandwidth value
                    kde = gaussian_kde(dataset_np, bw_method=effective_bw, weights=weights_np)
                    bandwidth = effective_bw # Use the adjusted value
                except Exception as bw_e:
                      print(f"  Error re-applying adjusted bandwidth {effective_bw}: {bw_e}. Using original KDE object.")
                      # Keep the original kde object

            print(f"  Final KDE Bandwidth: {bandwidth:.4f}")
            x_grid = np.linspace(0, 1, 500) # Grid for density evaluation
            density = kde(x_grid)
            density = np.maximum(density, 0) # Ensure non-negativity
            total_area = np.trapz(density, x_grid)
            if total_area > 1e-6:
                density /= total_area # Normalize density
            else:
                print(f"  Warning: KDE density integrates to near zero ({total_area}). Resulting in uniform density.")
                density = np.ones_like(x_grid) / (x_grid[-1] - x_grid[0]) # Assign uniform
            return DensityEstimation(x=x_grid.tolist(), density=density.tolist(), bandwidth=bandwidth)
        except Exception as e:
            print(f"!! Error during KDE calculation: {e}")
            traceback.print_exc()
            return DensityEstimation(x=[0.0, 1.0], density=[1.0, 1.0], bandwidth=0.1) # Return default on error

    def _calculate_upward_probability(self, density_estimation: DensityEstimation) -> float:
        """根据密度估计计算期望值 E[p]"""
        print("  Calculating expected value from KDE...")
        if not density_estimation.x or not density_estimation.density or len(density_estimation.x) < 2:
            print("  Warning: Insufficient data in density estimation. Returning 0.5.")
            return 0.5
        try:
            x = np.array(density_estimation.x)
            density = np.array(density_estimation.density)
            # Ensure density sums to 1 (or close enough) before calculating expectation
            area = np.trapz(density, x)
            if abs(area - 1.0) > 1e-3: # Allow small tolerance
                 print(f"  Warning: KDE density does not integrate to 1 (Area={area:.4f}). Normalizing for expectation.")
                 if area > 1e-6:
                      density = density / area
                 else:
                      print("  Area is zero, cannot calculate expectation. Returning 0.5.")
                      return 0.5

            integrand = x * density # p * f(p)
            expected_value = np.trapz(integrand, x) # Integral(p * f(p) dp)
            final_probability = min(max(float(expected_value), 0.0), 1.0) # Clamp to [0, 1]
            print(f"  Calculated Expected Value E[p] = P(up): {final_probability:.4f}")
            return final_probability
        except Exception as e:
            print(f"!! Error calculating expected value P(up): {e}")
            traceback.print_exc()
            return 0.5

    @rate_limit_api_call
    def _get_probability_tokens(self,
                                preceding_node_state_for_template: str,
                                strategies_for_template: str
                               ) -> List[List[TokenInfo]]:
        """获取概率值的tokens和对应的logits。系统提示来自XML模板。"""
        try:
            raw_template_xml = self._load_prompt_template()
            system_prompt = raw_template_xml.replace('{num_probabilities}', str(self.num_probs_to_generate))
            system_prompt = system_prompt.replace('[DESCRIPTION OF THE PRECEDING NODE STATE]', preceding_node_state_for_template)
            system_prompt = system_prompt.replace('[STRATEGY RETRIEVED FROM CEP]', strategies_for_template)

            user_prompt_text = (
                f"Please provide {self.num_probs_to_generate} diverse upward trend probability estimates "
                f"strictly following the <Output><UpwardProbabilities> format defined in the system instructions."
            )

            messages_for_llm = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_text}
            ]

            print(f"\n--- [ActionProbCalc] Calling LLM for Action Probability (expecting {self.num_probs_to_generate} values) ---")

            # Assuming openai.api_key is set globally by run.py
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages_for_llm,
                temperature=0.2,
                max_tokens=300, # Increased slightly for safety
                logprobs=True,
                top_logprobs=1
            )
            print("--- [ActionProbCalc] LLM Call Successful ---")

            if not response.choices or not response.choices[0].logprobs or not response.choices[0].logprobs.content:
                 print("Error: LLM response missing necessary logprobs content.")
                 raise ValueError("Failed to get logprobs from LLM response.")

            raw_completion_text = response.choices[0].message.content.strip()
            logprob_content = response.choices[0].logprobs.content

            print(f"Debug - LLM Raw Completion Text:\n{raw_completion_text}")

            # --- 解析概率值字符串 ---
            probability_strings = []
            match = re.search(r"<UpwardProbabilities>(.*?)</UpwardProbabilities>", raw_completion_text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_content = match.group(1).strip()
                if extracted_content.startswith('[') and extracted_content.endswith(']'):
                    extracted_content = extracted_content[1:-1]
                temp_probability_strings = [p.strip() for p in re.split(r'[,\s\n]+', extracted_content) if p.strip()]
                for p_str in temp_probability_strings:
                    if re.fullmatch(r'[0-9]+(\.[0-9]+)?', p_str):
                        try: float(p_str); probability_strings.append(p_str)
                        except ValueError: pass
                print(f"Debug - Extracted valid probability strings via Regex: {probability_strings}")
            else:
                 print("Warning: Could not find <UpwardProbabilities> tag. Attempting fallback.")
                 raw_numbers = re.findall(r"([0-1]\.[0-9]+|[0]\.[0-9]+|1\.0|0(?![0-9])|1(?![0-9]))", raw_completion_text)
                 if raw_numbers:
                     probability_strings = raw_numbers
                     print(f"Warning: Used fallback regex to extract numbers: {probability_strings}")
                 else:
                     print(f"Error: Failed to extract probabilities. Raw text: {raw_completion_text}")
                     raise ValueError("Failed to extract any probability strings.")

            if not probability_strings:
                print("Error: No valid probability strings extracted.")
                raise ValueError("No probability strings extracted.")

            # --- 将 Logprobs 映射到概率字符串 ---
            all_mapped_tokens_for_probs: List[List[TokenInfo]] = []
            logprob_iter = iter(logprob_content)
            current_completion_text_segment = raw_completion_text # For potential future offset logic

            # Find start offset more reliably
            content_start_offset = -1
            if match:
                 content_start_offset = match.start(1) # Start of content inside tag
            elif probability_strings:
                 try: content_start_offset = raw_completion_text.find(probability_strings[0])
                 except ValueError: pass

            # Skip logprobs before the content roughly starts (optional optimization)
            skipped_tokens = 0
            processed_chars = 0
            if content_start_offset > 0:
                 temp_iter = iter(logprob_content) # Use a separate iterator for skipping
                 try:
                     while processed_chars < content_start_offset:
                          token_entry = next(temp_iter)
                          processed_chars += len(token_entry.token) # Rough char count
                          logprob_iter = temp_iter # Advance main iterator if skipping works
                          skipped_tokens += 1
                 except StopIteration:
                      pass # Logprobs ended before offset
                 print(f"Debug: Skipped approx {skipped_tokens} logprobs before content.")


            for prob_str_target in probability_strings:
                tokens_for_current_prob_str: List[TokenInfo] = []
                reconstructed_prob_from_tokens = ""
                found_match_for_this_prob_str = False

                try:
                    while not found_match_for_this_prob_str: # Keep consuming until target found or stream ends
                        token_entry = next(logprob_iter)
                        token_text = token_entry.token
                        token_logit = token_entry.logprob

                        # Try to build target string, ignoring whitespace in token
                        token_text_strip = token_text.strip()
                        if not token_text_strip: continue # Skip whitespace tokens

                        potential_reconstruction = reconstructed_prob_from_tokens + token_text_strip

                        if prob_str_target.startswith(potential_reconstruction):
                            reconstructed_prob_from_tokens = potential_reconstruction
                            try:
                                # Only append if token contributes to the string
                                if token_text_strip:
                                     tokens_for_current_prob_str.append(TokenInfo(token=token_text_strip, logit=token_logit))
                            except ValueError as ve_token:
                                print(f"Warning: Invalid token data ('{token_text}', logit={token_logit}) for target '{prob_str_target}'. Error: {ve_token}. Skipping.")

                            if reconstructed_prob_from_tokens == prob_str_target:
                                all_mapped_tokens_for_probs.append(tokens_for_current_prob_str)
                                found_match_for_this_prob_str = True
                                print(f"  Successfully mapped: '{prob_str_target}' -> Tokens: {[t.token for t in tokens_for_current_prob_str]}")
                                # Don't break inner loop here, let StopIteration handle end
                        elif reconstructed_prob_from_tokens:
                            # Sequence broken after starting
                            print(f"  Sequence broken for '{prob_str_target}' by token '{token_text}' (built: '{reconstructed_prob_from_tokens}').")
                            # This token might start the NEXT probability, so we break the inner loop
                            # but the outer loop will continue with the next target.
                            # The current token is consumed.
                            break # Break while True for this target, move to next target string

                except StopIteration:
                    print("Warning: Logprobs stream ended during mapping.")
                    # Add any partially collected tokens if a match wasn't found
                    if not found_match_for_this_prob_str and tokens_for_current_prob_str:
                         print(f"  Adding partially mapped tokens for '{prob_str_target}'.")
                         all_mapped_tokens_for_probs.append(tokens_for_current_prob_str)
                    break # Break outer for loop

                if not found_match_for_this_prob_str:
                    print(f"Warning: Could not fully map tokens for '{prob_str_target}'.")
                    # Append placeholder if nothing was mapped
                    if not tokens_for_current_prob_str:
                         all_mapped_tokens_for_probs.append([TokenInfo(token=prob_str_target, logit=-100.0)])


            # Final padding/truncating (same as before)
            if len(all_mapped_tokens_for_probs) < len(probability_strings):
                 print(f"Warning: Final mapped lists ({len(all_mapped_tokens_for_probs)}) < extracted strings ({len(probability_strings)}). Padding.")
                 for i in range(len(all_mapped_tokens_for_probs), len(probability_strings)):
                     all_mapped_tokens_for_probs.append([TokenInfo(token=probability_strings[i], logit=-100.0)])
            elif len(all_mapped_tokens_for_probs) > len(probability_strings):
                 print(f"Warning: Final mapped lists ({len(all_mapped_tokens_for_probs)}) > extracted strings ({len(probability_strings)}). Truncating.")
                 all_mapped_tokens_for_probs = all_mapped_tokens_for_probs[:len(probability_strings)]

            print(f"--- [ActionProbCalc] Logprob Mapping Result: Mapped logits for {len(all_mapped_tokens_for_probs)} values ---")
            return all_mapped_tokens_for_probs

        except ValueError as ve:
            print(f"ValueError during _get_probability_tokens: {str(ve)}")
            traceback.print_exc()
            return []
        except Exception as e:
            print(f"获取或映射概率值tokens和logits时发生严重错误: {str(e)}")
            traceback.print_exc()
            return []

    # update_inference_file method is removed as its functionality is typically handled by the main run script
