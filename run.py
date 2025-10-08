# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "true"

import json
import logging
from datetime import datetime

import openai

WELCOME_COLOR_1 = "\033[1;36m"  
WELCOME_COLOR_2 = "\033[1;35m"  
WELCOME_COLOR_3 = "\033[1;33m"  
RESET_COLOR = "\033[0m"

print(f"""
{WELCOME_COLOR_1}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓{RESET_COLOR}
{WELCOME_COLOR_1}┃{RESET_COLOR}                {WELCOME_COLOR_2}Welcome to MarketToM - Market Theory of Mind{RESET_COLOR}                   {WELCOME_COLOR_1}┃{RESET_COLOR}
{WELCOME_COLOR_1}┃{RESET_COLOR}       {WELCOME_COLOR_3}Modeling market mental state and predicting stock trend{RESET_COLOR}                 {WELCOME_COLOR_1}┃{RESET_COLOR}
{WELCOME_COLOR_1}┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛{RESET_COLOR}
""")

from core import (
    CognitiveEnhancementPlugin,
    MentalStateInference, 
    DataLogger,
    ActionProbabilityCalculator,
    BackwardInference
)
from data.data_input import load_stock_data

# --- Setup logging for technical information ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='market_tom_technical.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM')

# --- Logging and Configuration Loading Functions ---
def load_prediction_log(log_path: str) -> dict:
    """
    Load the prediction log file. Returns an empty dictionary if the file doesn't exist or is empty.
    """
    if not os.path.exists(log_path):
        return {"predictions": []}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "predictions" not in data:
                data["predictions"] = []
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {"predictions": []}

def save_prediction_log(log_path: str, log_data: dict):
    """
    Save prediction results to a local JSON file in real-time.
    """
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = 'config.json'
    config_abs_path = os.path.join(script_dir, config_path)
    print(f"Loading config from: {config_abs_path}")
    try:
        with open(config_abs_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_abs_path}")
        raise
    except Exception as e:
        print(f"Error loading config file {config_abs_path}: {str(e)}")
        raise

def main():
    # ============ 1. Load Configuration and Set Global Items ============
    print("\n\033[1;36m=== STEP 1: LOADING CONFIGURATION ===\033[0m")
    config = load_config()
    api_config = config.get('api', {})
    active_provider_name = api_config.get('active_llm_provider', 'openai').lower()
    provider_configs = api_config.get('providers', {})

    llm_client = None
    llm_model_to_use = None

    if active_provider_name == 'openai':
        openai_provider_config = provider_configs.get('openai', {})
        if not openai_provider_config.get('api_key'):
            raise ValueError("OpenAI API key not found in config.json for the active provider.")
            
        base_url = openai_provider_config.get('base_url')
        if base_url and base_url.strip():
            llm_client = openai.OpenAI(
                api_key=openai_provider_config.get('api_key'),
                base_url=base_url
            )
            print(f"\033[32m✅ Connected to OpenAI (custom endpoint)\033[0m")
        else:
            llm_client = openai.OpenAI(
                api_key=openai_provider_config.get('api_key'),
                base_url=base_url
            )
            print(f"\033[32m✅ Connected to OpenAI (official API)\033[0m")
            
        llm_model_to_use = openai_provider_config.get('llm_model_default', 'gpt-4o')
    elif active_provider_name == 'grok':
        grok_provider_config = provider_configs.get('grok', {})
        if not grok_provider_config.get('api_key') or not grok_provider_config.get('base_url'):
            raise ValueError("Grok API key or base_url not found in config.json for the active provider.")
        llm_client = openai.OpenAI(
            api_key=grok_provider_config.get('api_key'),
            base_url=grok_provider_config.get('base_url')
        )
        llm_model_to_use = grok_provider_config.get('llm_model_default', 'grok-3-beta')
        print(f"\033[32m✅ Connected to Grok\033[0m")
    else:
        raise ValueError(f"Unsupported LLM provider: {active_provider_name}. Check config.json.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ============ 2. Parse Paths and Parameters from Config (ensure absolute paths) ============
    print("\n\033[1;36m=== STEP 2: PREPARING SYSTEM ===\033[0m")
    
    directories_config = config.get('directories', {})
    
    inference_logs_rel = directories_config.get('inference_logs', './MarketToM1/inference_logs')
    inference_logs_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, inference_logs_rel)))
    
    strategy_database_rel = directories_config.get('strategy_database', './MarketToM1/strategy_database')
    strategy_database_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, strategy_database_rel)))
    
    data_params_config = config.get('data_params', {})
    dataset_name = data_params_config.get('dataset_name', 'StockNet')
    dataset_split = data_params_config.get('dataset_split', 'Test')
    
    data_base_dir_rel = directories_config.get('data_base_dir', './data')
    data_base_dir_full = os.path.join(data_base_dir_rel, dataset_name, dataset_split)
    data_base_dir_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, data_base_dir_full)))
    
    logger.info(f"Using data path: {data_base_dir_abs}")

    prediction_log_rel = directories_config.get('prediction_log_path', './MarketToM1/prediction_results.json')
    prediction_log_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, prediction_log_rel)))

    templates_config = config.get('templates', {})
    fwd_template_rel = templates_config.get('forward_inference', 'forward_prompt_template.xml')
    fwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, fwd_template_rel)))
    if not os.path.isfile(fwd_template_abs):
        raise FileNotFoundError(f"Forward inference template not found at: {fwd_template_abs}")

    act_prob_template_rel = templates_config.get('action_probability', 'action_prob_prompt_template.xml')
    act_prob_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, act_prob_template_rel)))
    if not os.path.isfile(act_prob_template_abs):
        raise FileNotFoundError(f"Action probability template not found at: {act_prob_template_abs}")

    expert_prob_template_rel = templates_config.get('expert_action_probability', 'expert_action_prob_template.xml')
    expert_prob_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, expert_prob_template_rel)))
    if not os.path.isfile(expert_prob_template_abs):
        print(f"\033[33m⚠️ Expert perspective template not found, using default method instead\033[0m")
        logger.warning(f"Expert perspective template not found: {expert_prob_template_abs}")
        expert_prob_template_abs = None

    bwd_template_rel = templates_config.get('backward_inference', 'backward_prompt_template.xml')
    bwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, bwd_template_rel)))
    if not os.path.isfile(bwd_template_abs):
        raise FileNotFoundError(f"Backward inference template not found at: {bwd_template_abs}")

    # --- Data Parameters ---
    # Note: data_params_config was loaded earlier, just getting other parameters now
    default_window_size = data_params_config.get('default_window_size', 5)
    default_stocks = data_params_config.get('default_stocks', ["AAPL"])
    skip_backward_inference = data_params_config.get('skip_backward_inference', False)

    # --- CEP Retrieval Parameters ---
    cep_retrieval_config = config.get('cep_retrieval', {})
    cep_default_top_k = cep_retrieval_config.get('default_top_k', 1)
    cep_similarity_threshold = cep_retrieval_config.get('similarity_threshold', 0.1)
    emotion_similarity_threshold = cep_retrieval_config.get('emotion_similarity_threshold', 0.1)
    belief_similarity_threshold = cep_retrieval_config.get('belief_similarity_threshold', 0.1)
    intent_similarity_threshold = cep_retrieval_config.get('intent_similarity_threshold', 0.1)
    
    # --- Module-Specific Parameters ---
    fwd_inf_params = config.get('forward_inference_params', {})
    act_prob_params = config.get('action_probability_params', {})
    bwd_inf_params = config.get('backward_inference_params', {})

    # Get probability calculation method configuration
    use_expert_method = act_prob_params.get('use_expert_perspective_method', False)
    prob_method_name = "Expert perspective" if use_expert_method else "Standard"
    print(f"\033[94mAction probability method: {prob_method_name}\033[0m")

    if use_expert_method and expert_prob_template_abs is None:
        print(f"\033[91mWARNING: Expert perspective method selected but template file not found!\033[0m")
        print(f"\033[91mFalling back to standard probability method\033[0m")
        use_expert_method = False
        prob_method_name = "Standard (fallback)"
    elif use_expert_method:
        print(f"\033[94mExpert template path: {expert_prob_template_abs}\033[0m")

    print(f"\033[32m✓ Using {prob_method_name} probability calculation method\033[0m")
    if use_expert_method:
        logger.info("Expert perspective method uses logarithmic confidence weighting")

    os.makedirs(inference_logs_abs, exist_ok=True)
    os.makedirs(os.path.dirname(prediction_log_abs), exist_ok=True)
    print(f"\033[32m✅ Configuration loaded, using model: {llm_model_to_use}\033[0m")

    # ============ 3. Initialize Components (pass resolved configurations) ============
    print("\n\033[1;36m=== STEP 3: INITIALIZING COMPONENTS ===\033[0m")
    print(f"\033[90mInitializing cognitive enhancement plugin...\033[0m")
    cep = CognitiveEnhancementPlugin(storage_path=strategy_database_abs)
    print(f"\033[32m✅ Cognitive enhancement plugin ready\033[0m")
    
    data_logger = DataLogger(log_dir_abs_path=inference_logs_abs)
    
    print(f"\033[90mInitializing forward inference...\033[0m")
    inferencer = MentalStateInference(
        cep=cep,
        logger=data_logger,
        llm_client=llm_client,
        llm_model=llm_model_to_use,
        forward_template_abs_path=fwd_template_abs,
        cep_default_top_k=cep_default_top_k,
        cep_similarity_threshold=cep_similarity_threshold,
        fwd_inf_max_retries=fwd_inf_params.get('max_retries', 5),
        fwd_inf_base_delay=fwd_inf_params.get('base_delay_seconds', 1),
        emotion_similarity_threshold=emotion_similarity_threshold,
        belief_similarity_threshold=belief_similarity_threshold,
        intent_similarity_threshold=intent_similarity_threshold,
        llm_temperature=fwd_inf_params.get('llm_temperature', 0.7)
    )
    print(f"\033[32m✅ Market psychology analyzer ready\033[0m")
    
    print(f"\033[90mInitializing action probability calculator...\033[0m")
    calculator = ActionProbabilityCalculator(
        cep=cep,
        llm_client=llm_client,
        llm_model=llm_model_to_use,
        action_prob_template_abs_path=act_prob_template_abs,
        inference_logs_abs_path=inference_logs_abs, 
        action_prob_top_k=cep_retrieval_config.get('action_prob_top_k', 2), 
        num_probs_to_generate=act_prob_params.get('num_probabilities_to_generate', 10),
        max_retries_list=act_prob_params.get('max_retries_list', 5),
        base_delay_list_seconds=act_prob_params.get('base_delay_list_seconds', 1.0), 
        llm_temperature=act_prob_params.get('llm_temperature', 0.7),
        expert_prob_method=use_expert_method,
        expert_template_abs_path=expert_prob_template_abs if use_expert_method else None,
        kde_bandwidth_rule=act_prob_params.get('kde_bandwidth_rule', 'silverman'), 
        kde_min_bandwidth=act_prob_params.get('kde_min_bandwidth', 0.01) 
    )
    print(f"\033[32m✅ Market prediction system ready\033[0m")
    
    print(f"\033[90mInitializing backward inference...\033[0m")
    backward_inference = BackwardInference(
        cep=cep,
        llm_client=llm_client,
        llm_model=llm_model_to_use,
        backward_template_abs_path=bwd_template_abs,
        inference_logs_abs_path=inference_logs_abs, 
        max_retries=bwd_inf_params.get('max_retries', 5),
        base_delay_seconds=bwd_inf_params.get('base_delay_seconds', 2), 
        llm_temperature=bwd_inf_params.get('llm_temperature', 0.2),
        llm_max_tokens=bwd_inf_params.get('llm_max_tokens', 5000) 
    )
    print(f"\033[32m✅ Error analysis system ready\033[0m")
    
    # ============ 4. Load Data and Existing Prediction Logs ============
    print(f"\n\033[1;36m=== STEP 4: LOADING MARKET DATA ===\033[0m")
    print(f"\033[90mLoading stock data...\033[0m")
    train_text_data, train_price_data, train_labels = load_stock_data(data_base_dir_abs, default_stocks)
    length = train_price_data.shape[0]
    num_digits = 3  # Fixed to 3 digits to match format_day_number in data_input.py
    print(f"\033[32m✅ Loaded data for {len(default_stocks)} stocks with {length} trading days\033[0m")
    
    prediction_data = load_prediction_log(prediction_log_abs)
    done_indices = {item["index"] for item in prediction_data["predictions"]}
    print(f"\033[32m✅ Found {len(done_indices)} previously analyzed trading days\033[0m")

    # ============ 5. Iterate Through Data Samples for Inference and Prediction ============
    print(f"\n\033[1;36m=== STEP 5: STARTING MARKET PREDICTION ===\033[0m")
    for i in range(default_window_size, length + 1):
        if i in done_indices:
            logger.info(f"Skipping sample {i} (already processed)")
            continue

        print(f"\n\033[1;35mAnalyzing trading day {i}/{length}...\033[0m")

        window_texts = []
        window_prices = []
        for j in range(i - default_window_size, i):
            day_str = default_stocks[0] + "day" + f"{j:0{num_digits}d}"
            tweets = train_text_data.get(day_str, {})
            if tweets:
                window_texts.extend([tweet['content'] for tweet in tweets.values()])
            else:
                logger.warning(f"No tweets found for {day_str}")
            price_row = train_price_data[j]
            window_prices.append(price_row)

        label = train_labels[i - 1]
        prices_str = "".join([
            f"Day {idx + 1}: Open={p[0]}, High={p[1]}, Low={p[2]}, Close={p[3]}, Volume={p[4]}"
            for idx, p in enumerate(window_prices)
        ])
        env_state = f"""Market State Description:
            1. Price Conditions:
            {prices_str}

            2. Social Media Tweets (past {default_window_size} days):
            - {", ".join(window_texts)}
        """

        print("\n\033[1;34m▶ Analyzing market psychology...\033[0m")
        try:
            inference_result, generated_filename = inferencer.forward_inference(env_state)
            print(f"\033[32m✅ Market psychology analysis complete\033[0m")
            logger.info(f"Forward inference completed, log file: {generated_filename}")
            
            full_generated_filepath = os.path.join(inference_logs_abs, generated_filename)
            file_accessible = False
            if os.path.isfile(full_generated_filepath):
                file_accessible = True
            else:
                print(f"\033[31m❌ Error accessing analysis results\033[0m")
                logger.error(f"Log file not accessible: {generated_filename}")
                if os.path.exists(inference_logs_abs):
                    try: 
                        dir_contents = os.listdir(inference_logs_abs)[:5]
                        logger.error(f"Directory contents: {dir_contents}")
                    except Exception as list_e: 
                        logger.error(f"Cannot list directory contents: {list_e}")
                else: 
                    logger.error(f"Directory does not exist: {inference_logs_abs}")

            if file_accessible:
                prob_method_text = "expert" if use_expert_method else "standard"
                print(f"\n\033[1;34m▶ Calculating market movement probability...\033[0m")
                try:
                    probability_result = calculator.calculate_probability_from_file(generated_filename) 
                    print(f"\033[33m💡 Probability: {probability_result.probability:.4f}\033[0m")

                    predicted_up = probability_result.probability > 0.5
                    is_correct = (predicted_up == bool(label))
                    
                    prediction_str = "\033[32mCORRECT PREDICTION ✓\033[0m" if is_correct else "\033[31mINCORRECT PREDICTION ✗\033[0m"
                    print(f"\033[1mResult: {prediction_str} [Predicted: {'UP 📈' if predicted_up else 'DOWN 📉'} | Actual: {'UP 📈' if label == 1 else 'DOWN 📉'}]\033[0m")

                    if not is_correct and not skip_backward_inference:
                        print("\n\033[1;34m▶ Analyzing prediction error...\033[0m")
                        try:
                            predicted_action_str = 'Buy' if predicted_up else 'Sell'
                            actual_action_str = 'Buy' if label == 1 else 'Sell'
                            
                            backward_result = backward_inference.perform_backward_inference(
                                filename=generated_filename, 
                                predicted_action=predicted_action_str,
                                actual_action=actual_action_str
                            )
                            if backward_result:
                                # New format: dict {'analysis': str, 'strategy_updates': dict}
                                analysis_text = backward_result.get('analysis', '')
                                strategy_updates = backward_result.get('strategy_updates', {})
                                
                                # Display analysis text
                                if analysis_text:
                                    analysis_parts = analysis_text.split("</ErrorAnalysis>", 1)
                                    if len(analysis_parts) > 1:
                                        error_analysis = analysis_parts[0].split("<ErrorAnalysis>")[-1].strip()
                                        print(f"\033[33m🔍 Error analysis: {error_analysis[:200]}...\033[0m")
                                
                                # Display strategy update summary
                                if strategy_updates:
                                    total_updates = sum(len(updates) for updates in strategy_updates.values())
                                    print(f"\033[32m✅ Updated {total_updates} strategies\033[0m")
                            else:
                                print("\033[31m❌ Error analysis failed\033[0m")
                        except Exception as bk_e:
                            logger.error(f"Backward inference error: {str(bk_e)}")
                            print("\033[31m❌ Error during analysis\033[0m")

                    prediction_data["predictions"].append({
                        "index": i,
                        "probability": probability_result.probability,
                        "predicted_up": predicted_up,
                        "label": int(label),
                        "correct": is_correct,
                        "method": "expert" if use_expert_method else "default",
                        "timestamp": datetime.now().isoformat()
                    })
                    save_prediction_log(prediction_log_abs, prediction_data)
                
                except FileNotFoundError as e:
                    logger.error(f"File not found: {generated_filename}")
                    print(f"\033[31m❌ Data file not found\033[0m")
                    continue
                except Exception as e:
                    logger.error(f"Probability calculation error: {str(e)}")
                    print(f"\033[31m❌ Error calculating market probability\033[0m")
                    continue
            else:
                logger.error(f"Skipping probability calculation: file not accessible")
                print(f"\033[31m❌ Skipping analysis: data not accessible\033[0m")
                continue

        except Exception as e:
            logger.error(f"Forward inference failed: {str(e)}")
            print(f"\033[31m❌ Market psychology analysis failed\033[0m")
            continue
            
    print("\n\033[1;32m=== ANALYSIS COMPLETE ===\033[0m")

if __name__ == "__main__":
    main()
