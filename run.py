# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime

import openai

from data_input import load_stock_data
from forward_inference import MentalStateInference, DataLogger
from calculate_action_prob import ActionProbabilityCalculator
from backward_inference import BackwardInference
from cep import CognitiveEnhancementPlugin

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
    config_path = 'config.json' # Assumes config.json is in the same directory as run.py
    config_abs_path = os.path.join(script_dir, config_path)
    print(f"Loading config for run.py from: {config_abs_path}")
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
        llm_client = openai.OpenAI(
            api_key=openai_provider_config.get('api_key')
        )
        llm_model_to_use = openai_provider_config.get('llm_model_default', 'gpt-4o')
        print(f"Using OpenAI provider. Model: {llm_model_to_use}")
    elif active_provider_name == 'grok':
        grok_provider_config = provider_configs.get('grok', {})
        if not grok_provider_config.get('api_key') or not grok_provider_config.get('base_url'):
            raise ValueError("Grok API key or base_url not found in config.json for the active provider.")
        llm_client = openai.OpenAI(
            api_key=grok_provider_config.get('api_key'),
            base_url=grok_provider_config.get('base_url')
        )
        llm_model_to_use = grok_provider_config.get('llm_model_default', 'grok-3-beta')
        print(f"Using Grok provider. Model: {llm_model_to_use}. Base URL: {grok_provider_config.get('base_url')}")
    else:
        raise ValueError(f"Unsupported LLM provider: {active_provider_name}. Check config.json.")

    # script_dir is the directory where run.py is located, used as a base for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ============ 2. Parse Paths and Parameters from Config (ensure absolute paths) ============
    
    # --- Directory Paths ---
    directories_config = config.get('directories', {})
    
    inference_logs_rel = directories_config.get('inference_logs', './MarketToM1/inference_logs')
    inference_logs_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, inference_logs_rel)))
    
    strategy_database_rel = directories_config.get('strategy_database', './MarketToM1/strategy_database')
    strategy_database_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, strategy_database_rel)))
    
    data_base_dir_rel = directories_config.get('data_base_dir', './Data/Structured_Data/StockNet/Test')
    data_base_dir_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, data_base_dir_rel)))

    # --- File Paths ---
    prediction_log_rel = directories_config.get('prediction_log_path', './MarketToM1/prediction_results.json')
    prediction_log_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, prediction_log_rel)))

    # --- Template File Paths ---
    templates_config = config.get('templates', {})
    fwd_template_rel = templates_config.get('forward_inference', 'forward_prompt_template.xml')
    fwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, fwd_template_rel)))
    if not os.path.isfile(fwd_template_abs):
        raise FileNotFoundError(f"Forward inference template not found at: {fwd_template_abs}")

    act_prob_template_rel = templates_config.get('action_probability', 'action_prob_prompt_template.xml')
    act_prob_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, act_prob_template_rel)))
    if not os.path.isfile(act_prob_template_abs):
        raise FileNotFoundError(f"Action probability template not found at: {act_prob_template_abs}")

    bwd_template_rel = templates_config.get('backward_inference', 'backward_prompt_template.xml')
    bwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, bwd_template_rel)))
    if not os.path.isfile(bwd_template_abs):
        raise FileNotFoundError(f"Backward inference template not found at: {bwd_template_abs}")

    # --- Data Parameters ---
    data_params_config = config.get('data_params', {})
    default_window_size = data_params_config.get('default_window_size', 5)
    default_stocks = data_params_config.get('default_stocks', ["AAPL"])
    skip_backward_inference = data_params_config.get('skip_backward_inference', False)

    # --- CEP Retrieval Parameters ---
    cep_retrieval_config = config.get('cep_retrieval', {})
    cep_default_top_k = cep_retrieval_config.get('default_top_k', 1)
    cep_similarity_threshold = cep_retrieval_config.get('similarity_threshold', 0.3)
    
    # --- Module-Specific Parameters ---
    fwd_inf_params = config.get('forward_inference_params', {})
    act_prob_params = config.get('action_probability_params', {})
    bwd_inf_params = config.get('backward_inference_params', {})

    # Create necessary directories
    os.makedirs(inference_logs_abs, exist_ok=True)
    os.makedirs(os.path.dirname(prediction_log_abs), exist_ok=True) # Ensure prediction log directory exists
    # CEP's _ensure_storage_exists will handle creation of strategy_database_abs

    print(f"--- Resolved Configurations ---")
    print(f"  LLM Model: {llm_model_to_use}")
    print(f"  Inference Logs Dir: {inference_logs_abs}")
    print(f"  Strategy Database Dir: {strategy_database_abs}")
    print(f"  Data Base Dir: {data_base_dir_abs}")
    print(f"  Prediction Log File: {prediction_log_abs}")
    print(f"  Forward Template: {fwd_template_abs}")
    print(f"  Action Prob Template: {act_prob_template_abs}")
    print(f"  Backward Template: {bwd_template_abs}")
    print(f"  Default Stocks: {default_stocks}")
    print(f"  Window Size: {default_window_size}")
    print(f"-----------------------------")

    # ============ 3. Initialize Components (pass resolved configurations) ============
    print(f"Initializing CognitiveEnhancementPlugin with storage path: '{strategy_database_abs}'")
    cep = CognitiveEnhancementPlugin(storage_path=strategy_database_abs)
    print(f"CEP initialized. Internal storage path: '{cep.storage_path}'")
    
    logger = DataLogger(log_dir_abs_path=inference_logs_abs)
    
    inferencer = MentalStateInference(
        cep=cep,
        logger=logger,
        llm_client=llm_client,
        llm_model=llm_model_to_use,
        forward_template_abs_path=fwd_template_abs,
        cep_default_top_k=cep_default_top_k,
        cep_similarity_threshold=cep_similarity_threshold,
        fwd_inf_max_retries=fwd_inf_params.get('max_retries', 5),
        fwd_inf_base_delay=fwd_inf_params.get('base_delay_seconds', 1)
    )

    # --- Debugging section for ActionProbabilityCalculator initialization ---
    # print("" + "="*10 + " DEBUGGING ActionProbabilityCalculator CALL " + "="*10)
    # print(f"DEBUG: Importing ActionProbabilityCalculator from module: {ActionProbabilityCalculator.__module__}")
    # try:
    #     init_file = ActionProbabilityCalculator.__init__.__code__.co_filename
    #     print(f"DEBUG: __init__ definition expected from file: {init_file}")
    #     print(f"DEBUG: Parameters expected by __init__: {ActionProbabilityCalculator.__init__.__code__.co_varnames}")
    # except AttributeError:
    #     print("DEBUG: Could not inspect __init__ details (maybe it's a built-in or C extension?).")

    # calculator_args_to_pass = {
    # 'cep': cep,
    # 'llm_client': llm_client,
    # 'llm_model': llm_model_to_use,
    # 'action_prob_template_abs_path': act_prob_template_abs,
    # 'inference_logs_abs_path': inference_logs_abs,
    # 'action_prob_top_k': cep_retrieval_config.get('action_prob_top_k', 2),
    # 'num_probs_to_generate': act_prob_params.get('num_probabilities_to_generate', 10),
    # 'max_retries_list': act_prob_params.get('max_retries_list', 5),
    # 'base_delay_list_seconds': act_prob_params.get('base_delay_list_seconds', 1.0),
    # 'kde_bandwidth_rule': act_prob_params.get('kde_bandwidth_rule', 'silverman'),
    # 'kde_min_bandwidth': act_prob_params.get('kde_min_bandwidth', 0.01)
    # }
    # print("DEBUG: Arguments ACTUALLY being passed to ActionProbabilityCalculator:")
    # for key, value in calculator_args_to_pass.items():
    # print(f"  - {key}: {value} (Type: {type(value)})")
    # print("="*10 + " END DEBUGGING " + "="*10 + "")
    # --- End of debugging section ---

    
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
        kde_bandwidth_rule=act_prob_params.get('kde_bandwidth_rule', 'silverman'), 
        kde_min_bandwidth=act_prob_params.get('kde_min_bandwidth', 0.01) 
    )
    
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
    
    # ============ 4. Load Data and Existing Prediction Logs ============
    print(f"Loading stock data from: {data_base_dir_abs}")
    train_text_data, train_price_data, train_labels = load_stock_data(data_base_dir_abs, default_stocks)
    length = train_price_data.shape[0]
    num_digits = len(str(length))
    
    prediction_data = load_prediction_log(prediction_log_abs)
    done_indices = {item["index"] for item in prediction_data["predictions"]}

    # ============ 5. Iterate Through Data Samples for Inference and Prediction ============
    for i in range(default_window_size, length + 1):
        if i in done_indices:
            print(f"Sample {i} has already been predicted. Skip.")
            continue

        print(f"Processing sample {i}/{length}...")

        # --- (1) Prepare Sliding Window Data ---
        window_texts = []
        window_prices = []
        for j in range(i - default_window_size, i):
            day_str = default_stocks[0] + "day" + f"{j:0{num_digits}d}" # Assuming processing only the first stock
            tweets = train_text_data.get(day_str, {})
            if tweets:
                window_texts.extend([tweet['content'] for tweet in tweets.values()])
            else:
                print(f"Warning: No tweets found for {day_str}")
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

        # --- (4) Forward Inference ---
        print("Performing forward inference...")
        try:
            inference_result, generated_filename = inferencer.forward_inference(env_state)
            print(f"Forward inference log file generated: {generated_filename}")
            
            # --- Check file accessibility ---
            full_generated_filepath = os.path.join(inference_logs_abs, generated_filename)
            file_accessible = False
            if os.path.isfile(full_generated_filepath):
                 print(f"File {full_generated_filepath} confirmed to exist.")
                 file_accessible = True
            else:
                print(f"CRITICAL: File {full_generated_filepath} NOT FOUND after forward inference.")
                if os.path.exists(inference_logs_abs):
                    try: print(f"Contents of {inference_logs_abs}: {os.listdir(inference_logs_abs)}")
                    except Exception as list_e: print(f"Could not list directory {inference_logs_abs}: {list_e}")
                else: print(f"Directory {inference_logs_abs} does not exist.")


            # --- (5) Calculate Action Probability ---
            if file_accessible:
                print("Calculating market behavior probability...")
                try:
                    # calculator.calculate_probability_from_file expects the filename relative to inference_logs_abs
                    probability_result = calculator.calculate_probability_from_file(generated_filename) 
                    print(f"Probability calculation result: {probability_result.model_dump_json(indent=2)}")

                    predicted_up = probability_result.probability > 0.5
                    is_correct = (predicted_up == bool(label))

                    if not is_correct and not skip_backward_inference:
                        print("Prediction is incorrect! Activating backward inference...")
                        try:
                            predicted_action_str = 'Buy' if predicted_up else 'Sell'
                            actual_action_str = 'Buy' if label == 1 else 'Sell'
                            print(f"Performing backward inference for {generated_filename} (Predicted: {predicted_action_str}, Actual: {actual_action_str})")
                            
                            backward_result_text = backward_inference.perform_backward_inference(
                                filename=generated_filename, 
                                predicted_action=predicted_action_str,
                                actual_action=actual_action_str
                            )
                            if backward_result_text:
                                 print("=== Backward Inference LLM Analysis (Raw) ===")
                                 print(backward_result_text)
                            else:
                                 print("Backward inference call failed or returned no result.")
                        except Exception as bk_e:
                             print(f"Error during backward inference execution: {bk_e}")
                             import traceback
                             traceback.print_exc()

                    prediction_data["predictions"].append({
                        "index": i,
                        "probability": probability_result.probability,
                        "predicted_up": predicted_up,
                        "label": int(label),
                        "correct": is_correct,
                        "timestamp": datetime.now().isoformat()
                    })
                    save_prediction_log(prediction_log_abs, prediction_data)

                except FileNotFoundError as e:
                     print(f"Probability calculation failed (FileNotFound): {e}")
                     print(f"  Attempted to use file (relative to inference_logs_abs): {generated_filename}")
                     continue 
                except Exception as e:
                     print(f"Probability calculation failed (other error): {str(e)}")
                     import traceback
                     traceback.print_exc()
                     continue
            else:
                print(f"Skipping probability calculation because file {generated_filename} was not accessible in {inference_logs_abs}.")
                continue

        except Exception as e:
            print(f"Forward Inference failed for sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
            
    print("--- Processing Complete ---")

if __name__ == "__main__":
    main()