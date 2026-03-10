# -*- coding: utf-8 -*-
"""
MarketToM — Multi-Agent Market Theory of Mind.

Main workflow (Algorithm 2):
1. Initialize per-agent CEP.
2. For each sample:
   a. Multi-agent forward inference (3 agents × CCN).
   b. Per-agent action prediction + dynamic weighted aggregation.
   c. Compare with label; if wrong → inter-agent backward inference.
   d. Update EMA accuracy per agent.

Ablation support:
  python run.py                           # Use config.json defaults
  python run.py --preset MarketToM-1st    # Load preset from ablation_presets.json
  python run.py --preset LLM-only         # Raw LLM baseline
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "true"

import sys
import argparse
import json
import logging
from datetime import datetime
import traceback

import openai

WELCOME_COLOR_1 = "\033[1;36m"
WELCOME_COLOR_2 = "\033[1;35m"
WELCOME_COLOR_3 = "\033[1;33m"
RESET_COLOR = "\033[0m"

print(f"""
{WELCOME_COLOR_1}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓{RESET_COLOR}
{WELCOME_COLOR_1}┃{RESET_COLOR}                {WELCOME_COLOR_2}Welcome to MarketToM - Market Theory of Mind{RESET_COLOR}                   {WELCOME_COLOR_1}┃{RESET_COLOR}
{WELCOME_COLOR_1}┃{RESET_COLOR}       {WELCOME_COLOR_3}Multi-Agent Heterogeneous Framework with Second-Order ToM{RESET_COLOR}            {WELCOME_COLOR_1}┃{RESET_COLOR}
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

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='market_tom_technical.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM')


# --- CLI argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='MarketToM — Multi-Agent Market Theory of Mind')
    parser.add_argument('--preset', type=str, default=None,
                        help='Ablation preset name (e.g. LLM-only, MarketToM-1st, MarketToM-T0.7-n10-k1)')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all available ablation presets and exit')
    return parser.parse_args()


def load_ablation_presets(config: dict) -> dict:
    """Extract ablation presets from the 'ablation.presets' section of config."""
    return config.get('ablation', {}).get('presets', {})


def apply_preset(config: dict, preset_name: str, presets: dict) -> dict:
    """Deep-merge a preset's overrides into the live config.
    
    Preset ablation fields are merged into config['ablation'] (top-level keys only);
    other sections (forward_inference_params, cep_retrieval, etc.) are shallow-merged.
    The experiment_name is auto-set to the preset name.
    """
    if preset_name not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    preset = presets[preset_name]
    for section_key, section_val in preset.items():
        if section_key.startswith('_'):
            continue  # skip metadata
        if section_key == 'ablation':
            # Merge ablation overrides into top-level ablation (not into presets sub-dict)
            for k, v in section_val.items():
                config['ablation'][k] = v
        elif isinstance(section_val, dict) and section_key in config:
            config[section_key].update(section_val)
        else:
            config[section_key] = section_val
    # Auto-set experiment_name to preset name
    config['ablation']['experiment_name'] = preset_name
    print(f"\033[1;33m⚡ Ablation preset applied: {preset_name}\033[0m")
    desc = preset.get('_description', '')
    if desc:
        print(f"   {desc}")
    return config


# --- Utility functions ---
def load_prediction_log(log_path: str) -> dict:
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
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = 'config.json'
    config_abs_path = os.path.join(script_dir, config_path)
    print(f"Loading config from: {config_abs_path}")
    try:
        with open(config_abs_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_abs_path}")
        raise
    except Exception as e:
        print(f"Error loading config file {config_abs_path}: {str(e)}")
        raise


def main():
    args = parse_args()

    # --- List presets and exit ---
    if args.list_presets:
        config = load_config()
        presets = load_ablation_presets(config)
        top_ablation = config.get('ablation', {})
        if not presets:
            print("No presets found in config.json → ablation.presets.")
        else:
            print(f"\n{'='*60}")
            print(f"  Available ablation presets ({len(presets)})")
            print(f"{'='*60}")
            for name, p in presets.items():
                desc = p.get('_description', '')
                ab = p.get('ablation', {})
                # Fall back to top-level ablation defaults for display
                mode = ab.get('mode', top_ablation.get('mode', 'full'))
                tom = ab.get('tom_order', top_ablation.get('tom_order', 2))
                n = ab.get('num_action_samples', top_ablation.get('num_action_samples', 10))
                T = p.get('forward_inference_params', {}).get('llm_temperature', '—')
                k = p.get('cep_retrieval', {}).get('default_top_k', '—')
                print(f"  {name:30s}  mode={mode}, tom={tom}, T={T}, n={n}, k={k}")
                if desc:
                    print(f"    {'':30s}  {desc}")
            print()
        return

    # ============ 1. Load Configuration ============
    print("\n\033[1;36m=== STEP 1: LOADING CONFIGURATION ===\033[0m")
    config = load_config()

    # Apply ablation preset if requested
    if args.preset:
        presets = load_ablation_presets(config)
        config = apply_preset(config, args.preset, presets)
    api_config = config.get('api', {})
    active_provider_name = api_config.get('active_llm_provider', 'openai').lower()
    provider_configs = api_config.get('providers', {})

    llm_client = None
    llm_model_to_use = None

    if active_provider_name == 'openai':
        openai_provider_config = provider_configs.get('openai', {})
        if not openai_provider_config.get('api_key'):
            raise ValueError("OpenAI API key not found in config.json")
        base_url = openai_provider_config.get('base_url')
        if base_url and base_url.strip():
            llm_client = openai.OpenAI(
                api_key=openai_provider_config.get('api_key'),
                base_url=base_url
            )
            print(f"\033[32m✅ Connected to OpenAI (custom endpoint)\033[0m")
        else:
            llm_client = openai.OpenAI(
                api_key=openai_provider_config.get('api_key')
            )
            print(f"\033[32m✅ Connected to OpenAI (official API)\033[0m")
        llm_model_to_use = openai_provider_config.get('llm_model_default', 'gpt-4o')
    elif active_provider_name == 'grok':
        grok_config = provider_configs.get('grok', {})
        if not grok_config.get('api_key') or not grok_config.get('base_url'):
            raise ValueError("Grok API key or base_url not found in config.json")
        llm_client = openai.OpenAI(
            api_key=grok_config.get('api_key'),
            base_url=grok_config.get('base_url')
        )
        llm_model_to_use = grok_config.get('llm_model_default', 'grok-3-beta')
        print(f"\033[32m✅ Connected to Grok\033[0m")
    else:
        raise ValueError(f"Unsupported LLM provider: {active_provider_name}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ============ 2. Parse Paths and Parameters ============
    print("\n\033[1;36m=== STEP 2: PREPARING SYSTEM ===\033[0m")

    directories_config = config.get('directories', {})

    inference_logs_rel = directories_config.get('inference_logs', './storage/inference_logs')
    inference_logs_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, inference_logs_rel)))

    strategy_database_rel = directories_config.get('strategy_database', './storage/strategy_database')
    strategy_database_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, strategy_database_rel)))

    data_params_config = config.get('data_params', {})
    dataset_name = data_params_config.get('dataset_name', 'StockNet')
    dataset_split = data_params_config.get('dataset_split', 'Test')
    data_base_dir_rel = directories_config.get('data_base_dir', './data')
    data_base_dir_full = os.path.join(data_base_dir_rel, dataset_name, dataset_split)
    data_base_dir_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, data_base_dir_full)))

    prediction_log_rel = directories_config.get('prediction_log_path', './prediction_results.json')
    prediction_log_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, prediction_log_rel)))

    templates_config = config.get('templates', {})
    fwd_template_rel = templates_config.get('forward_inference', './templates/forward_prompt_template.xml')
    fwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, fwd_template_rel)))

    action_template_rel = templates_config.get('expert_action_probability', './templates/expert_action_prob_template.xml')
    action_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, action_template_rel)))

    bwd_template_rel = templates_config.get('backward_inference', './templates/backward_prompt_template.xml')
    bwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(script_dir, bwd_template_rel)))

    for tpl_path, tpl_name in [(fwd_template_abs, "Forward"), (action_template_abs, "Action"), (bwd_template_abs, "Backward")]:
        if not os.path.isfile(tpl_path):
            raise FileNotFoundError(f"{tpl_name} template not found at: {tpl_path}")

    # --- Data Parameters ---
    default_window_size = data_params_config.get('default_window_size', 5)
    default_stocks = data_params_config.get('default_stocks', ["AAPL"])
    skip_backward_inference = data_params_config.get('skip_backward_inference', False)

    # --- Multi-Agent parameters ---
    agent_config = config.get('agent_params', {})
    agent_roles = agent_config.get('agent_roles', ["Retail", "Institutional", "Arbitrageur"])
    alpha = agent_config.get('alpha', 1.0)
    gamma = agent_config.get('gamma', 1.0)
    agg_temperature = agent_config.get('temperature', agent_config.get('aggregation_temperature', 1.0))
    ema_decay = agent_config.get('ema_decay', 0.9)

    # --- Ablation experiment configuration ---
    ablation_config = config.get('ablation', {})
    ablation_mode = ablation_config.get('mode', 'full')          # full | llm_only | no_cep
    ablation_tom_order = ablation_config.get('tom_order', 2)     # 1 | 2
    ablation_cep_enabled = ablation_config.get('cep_enabled', True)
    ablation_backward_enabled = ablation_config.get('backward_enabled', True)
    ablation_num_samples = ablation_config.get('num_action_samples', 10)
    experiment_name = ablation_config.get('experiment_name', 'MarketToM-2nd')

    # Override derived flags from ablation mode
    if ablation_mode == 'llm_only':
        ablation_cep_enabled = False
        ablation_backward_enabled = False
    elif ablation_mode == 'no_cep':
        ablation_cep_enabled = False
        ablation_backward_enabled = False

    # skip_backward_inference from data_params OR ablation config
    effective_skip_backward = skip_backward_inference or (not ablation_backward_enabled)

    print(f"\033[1;33m📋 Experiment: {experiment_name}  "
          f"[mode={ablation_mode}, tom={ablation_tom_order}, "
          f"cep={ablation_cep_enabled}, n={ablation_num_samples}]\033[0m")

    # --- CEP Retrieval Parameters ---
    cep_retrieval_config = config.get('cep_retrieval', {})
    cep_default_top_k = cep_retrieval_config.get('default_top_k', 1)
    cep_similarity_threshold = cep_retrieval_config.get('similarity_threshold', 0.1)
    emotion_similarity_threshold = cep_retrieval_config.get('emotion_similarity_threshold', 0.1)
    belief_similarity_threshold = cep_retrieval_config.get('belief_similarity_threshold', 0.1)
    intent_similarity_threshold = cep_retrieval_config.get('intent_similarity_threshold', 0.1)

    # --- Module-specific params ---
    fwd_inf_params = config.get('forward_inference_params', {})
    act_prob_params = config.get('action_probability_params', {})
    bwd_inf_params = config.get('backward_inference_params', {})

    os.makedirs(inference_logs_abs, exist_ok=True)
    os.makedirs(os.path.dirname(prediction_log_abs), exist_ok=True)
    print(f"\033[32m✅ Configuration loaded — model: {llm_model_to_use}, agents: {agent_roles}\033[0m")

    # ============ 3. Initialize Components ============
    print("\n\033[1;36m=== STEP 3: INITIALIZING COMPONENTS ===\033[0m")

    print(f"\033[90mInitializing per-agent CEP...\033[0m")
    cep = CognitiveEnhancementPlugin(
        storage_path=strategy_database_abs,
        agent_roles=agent_roles
    )
    print(f"\033[32m✅ Per-agent CEP ready\033[0m")

    data_logger = DataLogger(log_dir_abs_path=inference_logs_abs)

    print(f"\033[90mInitializing multi-agent forward inference...\033[0m")
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
        llm_temperature=fwd_inf_params.get('llm_temperature', 0.7),
        agent_roles=agent_roles,
        tom_order=ablation_tom_order,
        cep_enabled=ablation_cep_enabled
    )
    print(f"\033[32m✅ Multi-agent inference ready (tom_order={ablation_tom_order}, cep={ablation_cep_enabled})\033[0m")

    print(f"\033[90mInitializing action probability calculator...\033[0m")
    calculator = ActionProbabilityCalculator(
        cep=cep,
        llm_client=llm_client,
        llm_model=llm_model_to_use,
        inference_logs_abs_path=inference_logs_abs,
        action_template_abs_path=action_template_abs,
        agent_roles=agent_roles,
        alpha=alpha,
        gamma=gamma,
        temperature=agg_temperature,
        ema_decay=ema_decay,
        max_retries=act_prob_params.get('max_retries_list', 5),
        base_delay=act_prob_params.get('base_delay_list_seconds', 1.0),
        llm_temperature=act_prob_params.get('llm_temperature', 0.7),
        num_action_samples=ablation_num_samples
    )
    print(f"\033[32m✅ Dynamic weighted aggregation ready (n={ablation_num_samples})\033[0m")

    print(f"\033[90mInitializing backward inference...\033[0m")
    backward_inference = BackwardInference(
        cep=cep,
        llm_client=llm_client,
        llm_model=llm_model_to_use,
        backward_template_abs_path=bwd_template_abs,
        inference_logs_abs_path=inference_logs_abs,
        agent_roles=agent_roles,
        max_retries=bwd_inf_params.get('max_retries', 5),
        base_delay_seconds=bwd_inf_params.get('base_delay_seconds', 2),
        llm_temperature=bwd_inf_params.get('llm_temperature', 0.2),
        llm_max_tokens=bwd_inf_params.get('llm_max_tokens', 5000)
    )
    print(f"\033[32m✅ Inter-agent learning ready\033[0m")

    # ============ 4. Load Data ============
    print(f"\n\033[1;36m=== STEP 4: LOADING MARKET DATA ===\033[0m")
    train_text_data, train_price_data, train_labels = load_stock_data(data_base_dir_abs, default_stocks)
    length = train_price_data.shape[0]
    num_digits = 3
    print(f"\033[32m✅ Loaded data for {len(default_stocks)} stocks, {length} trading days\033[0m")

    prediction_data = load_prediction_log(prediction_log_abs)
    done_indices = {item["index"] for item in prediction_data["predictions"]}
    print(f"\033[32m✅ Found {len(done_indices)} previously analyzed days\033[0m")

    # ============ 5. Main Prediction Loop (Algorithm 2) ============
    print(f"\n\033[1;36m=== STEP 5: STARTING MULTI-AGENT MARKET PREDICTION ===\033[0m")
    for i in range(default_window_size, length + 1):
        if i in done_indices:
            logger.info(f"Skipping sample {i} (already processed)")
            continue

        print(f"\n\033[1;35m{'='*60}\033[0m")
        print(f"\033[1;35m  Trading day {i}/{length}\033[0m")
        print(f"\033[1;35m{'='*60}\033[0m")

        # Build environmental state
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

        # ----- Step A: Multi-Agent Forward Inference -----
        if ablation_mode == 'llm_only':
            # LLM-only baseline: direct prediction without CCN/ToM framework
            print("\n\033[1;34m▶ LLM-only direct prediction (no CCN/ToM)...\033[0m")
            try:
                llm_only_prompt = (
                    f"You are a financial analyst. Based on the following market data, "
                    f"predict whether the stock price will go UP or DOWN tomorrow.\n\n"
                    f"{env_state}\n\n"
                    f"Respond with exactly one word: Buy or Sell."
                )
                llm_resp = llm_client.chat.completions.create(
                    model=llm_model_to_use,
                    messages=[{"role": "user", "content": llm_only_prompt}],
                    temperature=fwd_inf_params.get('llm_temperature', 0.7),
                    max_tokens=10
                )
                raw_answer = llm_resp.choices[0].message.content.strip().lower()
                predicted_up = 'buy' in raw_answer
                p_up = 1.0 if predicted_up else 0.0
                is_correct = (predicted_up == bool(label))

                prediction_str = "\033[32mCORRECT ✓\033[0m" if is_correct else "\033[31mINCORRECT ✗\033[0m"
                print(f"\033[1mResult: {prediction_str} "
                      f"[Predicted: {'UP 📈' if predicted_up else 'DOWN 📉'} | "
                      f"Actual: {'UP 📈' if label == 1 else 'DOWN 📉'}]\033[0m")

                prediction_data["predictions"].append({
                    "index": i,
                    "probability": p_up,
                    "predicted_up": predicted_up,
                    "label": int(label),
                    "correct": is_correct,
                    "method": "llm_only",
                    "experiment": experiment_name,
                    "timestamp": datetime.now().isoformat()
                })
                save_prediction_log(prediction_log_abs, prediction_data)
            except Exception as e:
                logger.error(f"LLM-only prediction failed: {e}")
                print(f"\033[31m❌ LLM-only prediction failed: {e}\033[0m")
            continue  # skip forward/backward, go to next sample

        print("\n\033[1;34m▶ Multi-agent forward inference...\033[0m")
        try:
            agent_results, generated_filename = inferencer.forward_inference(env_state)
            print(f"\033[32m✅ Forward inference complete\033[0m")

            full_path = os.path.join(inference_logs_abs, generated_filename)
            if not os.path.isfile(full_path):
                print(f"\033[31m❌ Error: log file not accessible\033[0m")
                continue

            # ----- Step B: Per-Agent Action Prediction + Aggregation -----
            print(f"\n\033[1;34m▶ Dynamic weighted aggregation...\033[0m")
            try:
                probability_result = calculator.calculate_probability_from_file(generated_filename)
                p_up = probability_result.probability
                print(f"\033[33m💡 Aggregated P(up) = {p_up:.4f}\033[0m")

                predicted_up = p_up > 0.5
                is_correct = (predicted_up == bool(label))

                prediction_str = "\033[32mCORRECT ✓\033[0m" if is_correct else "\033[31mINCORRECT ✗\033[0m"
                print(f"\033[1mResult: {prediction_str} "
                      f"[Predicted: {'UP 📈' if predicted_up else 'DOWN 📉'} | "
                      f"Actual: {'UP 📈' if label == 1 else 'DOWN 📉'}]\033[0m")

                # ----- Step C: Update EMA accuracy per agent -----
                actual_action_str = 'Buy' if label == 1 else 'Sell'
                for role, pred_info in probability_result.agent_predictions.items():
                    agent_correct = pred_info.get('predicted_action') == actual_action_str
                    calculator.update_ema_accuracy(role, agent_correct)

                # ----- Step D: Backward inference if wrong -----
                if not is_correct and not effective_skip_backward:
                    print("\n\033[1;34m▶ Inter-agent backward learning...\033[0m")
                    try:
                        backward_result = backward_inference.perform_backward_inference(
                            filename=generated_filename,
                            agent_predictions=probability_result.agent_predictions,
                            actual_action=actual_action_str
                        )
                        if backward_result:
                            total = sum(
                                sum(len(u) for u in agent_res.get('strategy_updates', {}).values())
                                for agent_res in backward_result.values()
                            )
                            print(f"\033[32m✅ Backward: {total} total strategy updates\033[0m")
                        else:
                            print("\033[33m⚠ No failing agents to learn from\033[0m")
                    except Exception as bk_e:
                        logger.error(f"Backward inference error: {bk_e}")
                        print(f"\033[31m❌ Error during backward learning\033[0m")

                # ----- Save prediction -----
                prediction_data["predictions"].append({
                    "index": i,
                    "probability": p_up,
                    "predicted_up": predicted_up,
                    "label": int(label),
                    "correct": is_correct,
                    "method": "multi_agent_aggregation",
                    "experiment": experiment_name,
                    "agent_weights": probability_result.weights,
                    "timestamp": datetime.now().isoformat()
                })
                save_prediction_log(prediction_log_abs, prediction_data)

            except FileNotFoundError as e:
                logger.error(f"File not found: {generated_filename}")
                print(f"\033[31m❌ Data file not found\033[0m")
                continue
            except Exception as e:
                logger.error(f"Probability calculation error: {e}")
                print(f"\033[31m❌ Error calculating market probability\033[0m")
                continue

        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Forward inference failed: {e}")
            logger.error(error_trace)
            print(f"\033[31m❌ Forward inference failed: {e}\033[0m")
            print(error_trace)
            raise

    print("\n\033[1;32m=== ANALYSIS COMPLETE ===\033[0m")


if __name__ == "__main__":
    main()
