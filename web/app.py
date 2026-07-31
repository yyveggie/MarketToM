import os
import sys
import json
import threading
import traceback
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import openai

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import core modules
import core.forward_inference as forward_inference_module
import core.calculate_action_prob as action_probability_module
import core.backward_inference as backward_inference_module
from core import (
    CognitiveEnhancementPlugin,
    MentalStateInference,
    DataLogger,
    ActionProbabilityCalculator,
    BackwardInference
)
from core.config_utils import get_active_provider_config, get_model_params, resolve_api_key
from visualization import MentalStateVisualizer

app = Flask(__name__)
CORS(app)

# Disable static file caching in development mode
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Add response headers to disable caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Load configuration (from project root)
config_path = os.path.join(project_root, 'config.json')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Global variables to store component instances
global_components = {
    'initialized': False,
    'cep': None,
    'data_logger': None,
    'inferencer': None,
    'calculator': None,
    'backward_engine': None,
    'llm_client': None,
    'llm_model': None,
    'llm_extra_body': None,
    'visualizer': None,
    'agent_roles': [],
    'ablation': {},
    'inference_logs_abs': None,
    'visualization_dir_abs': None
}

# Global variables to store current inference state
current_inference_state = {
    'status': 'idle',
    'progress': 0,
    'current_step': '',
    'results': None,
    'error': None,
    'intermediate_results': {
        'belief': None,
        'intent': None,
        'emotion': None,
        'belief_strategies': [],
        'intent_strategies': [],
        'emotion_strategies': [],
        'predicted_action': None,
        'confidence': None
    }
}


def resolve_project_path(path_value):
    if os.path.isabs(path_value):
        return os.path.normpath(path_value)
    return os.path.normpath(os.path.abspath(os.path.join(project_root, path_value)))


def apply_rate_limit_config():
    rate_config = config.get('api_rate_limit', {})
    for module in (forward_inference_module, action_probability_module, backward_inference_module):
        if 'min_request_interval' in rate_config:
            module.MIN_REQUEST_INTERVAL = float(rate_config['min_request_interval'])
        if 'default_cooldown' in rate_config:
            module.DEFAULT_COOLDOWN = float(rate_config['default_cooldown'])
        if 'max_jitter' in rate_config:
            module.MAX_JITTER = float(rate_config['max_jitter'])


def ablation_settings():
    ablation_config = config.get('ablation', {})
    mode = ablation_config.get('mode', 'full')
    cep_enabled = ablation_config.get('cep_enabled', True)
    backward_enabled = ablation_config.get('backward_enabled', True)
    if mode in {'llm_only', 'no_cep'}:
        cep_enabled = False
        backward_enabled = False
    return {
        'mode': mode,
        'tom_order': ablation_config.get('tom_order', 2),
        'cep_enabled': cep_enabled,
        'backward_enabled': backward_enabled,
        'ccn_dependency_variant': ablation_config.get('ccn_dependency_variant', 'full'),
        'tom_role_shuffle': ablation_config.get('tom_role_shuffle', False),
        'experiment_name': ablation_config.get('experiment_name', 'MarketToM-2nd')
    }


def core_action_to_display(action):
    action_lower = str(action or '').strip().lower()
    if action_lower == 'buy':
        return 'Up'
    if action_lower == 'sell':
        return 'Down'
    return str(action or 'Unknown')


def display_action_to_core(action):
    action_lower = str(action or '').strip().lower()
    if action_lower in {'up', 'buy', '1', 'true'}:
        return 'Buy'
    if action_lower in {'down', 'sell', '0', 'false'}:
        return 'Sell'
    return 'Unknown'


def label_to_display(label_value):
    if isinstance(label_value, bool):
        return 'Up' if label_value else 'Down'
    if isinstance(label_value, (int, float)):
        return 'Up' if int(label_value) == 1 else 'Down'
    label_lower = str(label_value or '').strip().lower()
    if label_lower in {'1', 'up', 'buy', 'true'}:
        return 'Up'
    if label_lower in {'0', 'down', 'sell', 'false'}:
        return 'Down'
    return str(label_value or 'Unknown')


def format_agent_dimension(agent_results, dimension):
    lines = []
    for role in global_components.get('agent_roles') or agent_results.keys():
        states = agent_results.get(role, {})
        value = extract_mental_state_description(str(states.get(dimension, '')))
        lines.append(f"{role}: {value or 'N/A'}")
    return "\n\n".join(lines)


def flatten_strategies(strategies_used, dimension):
    items = []
    for role, role_strategies in strategies_used.items():
        for strategy_id in role_strategies.get(dimension, []):
            items.append({'content': f"{role}: {strategy_id}"})
    return items


def build_expert_samples(probability_result):
    samples = []
    for index, (role, pred) in enumerate(probability_result.agent_predictions.items(), start=1):
        weight = probability_result.weights.get(role, pred.get('weight', 0.0))
        samples.append({
            'index': index,
            'role': role,
            'reasoning': f"Predicted action: {core_action_to_display(pred.get('predicted_action'))}",
            'probability': float(pred.get('p_up', 0.5)),
            'log_confidence': float(pred.get('log_confidence', 0.0)),
            'normalized_weight': float(weight or 0.0)
        })
    return samples


def cep_updates_allowed_for_split(split):
    data_params = config.get('data_params', {})
    cep_update_mode = data_params.get('cep_update_mode', 'benchmark').lower()
    return (
        cep_update_mode in {'online', 'realtime', 'real_time', 'live'}
        or str(split or '').lower() in {'train', 'training'}
    )


def count_backward_updates(backward_result):
    if not backward_result:
        return 0
    total = 0
    for agent_result in backward_result.values():
        updates = agent_result.get('strategy_updates', {}) if isinstance(agent_result, dict) else {}
        total += sum(len(items) for items in updates.values() if isinstance(items, list))
    return total


def load_price_data(price_data_raw):
    price_data = {}
    if isinstance(price_data_raw, dict) and 'price_data' in price_data_raw:
        for item in price_data_raw['price_data']:
            if 'day' in item:
                price_data[item['day']] = {k: v for k, v in item.items() if k != 'day'}
    elif isinstance(price_data_raw, dict):
        price_data = price_data_raw
    return price_data


def load_labels(labels_raw):
    labels = {}
    if isinstance(labels_raw, dict) and 'labels' in labels_raw:
        for item in labels_raw['labels']:
            if 'day' in item and 'label' in item:
                labels[item['day']] = label_to_display(item['label'])
    elif isinstance(labels_raw, dict):
        for day, label_value in labels_raw.items():
            labels[day] = label_to_display(label_value)
    return labels


def initialize_components():
    if global_components['initialized']:
        return
    
    print("Initializing MarketToM components...")

    apply_rate_limit_config()

    # Initialize LLM client
    active_provider_name, active_provider_config = get_active_provider_config(config)
    if not active_provider_config:
        raise ValueError(f"LLM provider config not found: {active_provider_name}")
    api_key = resolve_api_key(active_provider_config)
    if not api_key:
        raise ValueError(f"API key not found for provider '{active_provider_name}'")
    client_kwargs = {'api_key': api_key}
    base_url = active_provider_config.get('base_url')
    if base_url and base_url.strip():
        client_kwargs['base_url'] = base_url
    provider_timeout = active_provider_config.get('timeout')
    if provider_timeout:
        client_kwargs['timeout'] = float(provider_timeout)
    global_components['llm_client'] = openai.OpenAI(**client_kwargs)
    global_components['llm_model'] = active_provider_config.get('llm_model_default', 'gpt-4o')
    global_components['llm_extra_body'] = active_provider_config.get('extra_body')
    
    # Parse directory configuration (all paths relative to project root)
    directories_config = config.get('directories', {})
    
    strategy_database_rel = directories_config.get('strategy_database', './storage/strategy_database')
    strategy_database_abs = resolve_project_path(strategy_database_rel)
    
    inference_logs_rel = directories_config.get('inference_logs', './storage/inference_logs')
    inference_logs_abs = resolve_project_path(inference_logs_rel)
    global_components['inference_logs_abs'] = inference_logs_abs
    
    # Parse template paths
    templates_config = config.get('templates', {})
    fwd_template_rel = templates_config.get('forward_inference', './templates/forward_prompt_template.xml')
    fwd_template_abs = resolve_project_path(fwd_template_rel)
    
    expert_prob_template_rel = templates_config.get('expert_action_probability', './templates/expert_action_prob_template.xml')
    expert_prob_template_abs = resolve_project_path(expert_prob_template_rel)
    if not os.path.isfile(expert_prob_template_abs):
        raise FileNotFoundError(f"Expert perspective template not found at: {expert_prob_template_abs}")
    
    bwd_template_rel = templates_config.get('backward_inference', './templates/backward_prompt_template.xml')
    bwd_template_abs = resolve_project_path(bwd_template_rel)
    
    # Parse parameters
    cep_retrieval_config = config.get('cep_retrieval', {})
    fwd_inf_params = get_model_params(config, 'forward', provider_config=active_provider_config)
    act_prob_params = get_model_params(config, 'action_probability', provider_config=active_provider_config)
    bwd_inf_params = get_model_params(config, 'backward', provider_config=active_provider_config)
    agent_config = config.get('agent_params', {})
    agent_roles = agent_config.get('agent_roles', ["Retail", "Institutional", "Arbitrageur"])
    global_components['agent_roles'] = agent_roles
    global_components['ablation'] = ablation_settings()
    
    # Create necessary directories
    os.makedirs(strategy_database_abs, exist_ok=True)
    os.makedirs(inference_logs_abs, exist_ok=True)
    
    # Initialize components
    global_components['cep'] = CognitiveEnhancementPlugin(
        storage_path=strategy_database_abs,
        agent_roles=agent_roles
    )
    global_components['data_logger'] = DataLogger(log_dir_abs_path=inference_logs_abs)
    
    global_components['inferencer'] = MentalStateInference(
        cep=global_components['cep'],
        logger=global_components['data_logger'],
        llm_client=global_components['llm_client'],
        llm_model=global_components['llm_model'],
        forward_template_abs_path=fwd_template_abs,
        cep_default_top_k=cep_retrieval_config.get('default_top_k', 1),
        cep_similarity_threshold=cep_retrieval_config.get('similarity_threshold', 0.1),
        fwd_inf_max_retries=fwd_inf_params.get('max_retries', 5),
        fwd_inf_base_delay=fwd_inf_params.get('base_delay_seconds', 1),
        emotion_similarity_threshold=cep_retrieval_config.get('emotion_similarity_threshold', 0.1),
        belief_similarity_threshold=cep_retrieval_config.get('belief_similarity_threshold', 0.1),
        intent_similarity_threshold=cep_retrieval_config.get('intent_similarity_threshold', 0.1),
        llm_temperature=fwd_inf_params.get('llm_temperature', 0.7),
        agent_roles=agent_roles,
        tom_order=global_components['ablation']['tom_order'],
        cep_enabled=global_components['ablation']['cep_enabled'],
        ccn_dependency_variant=global_components['ablation']['ccn_dependency_variant'],
        llm_extra_body=global_components['llm_extra_body']
    )
    
    global_components['calculator'] = ActionProbabilityCalculator(
        cep=global_components['cep'],
        llm_client=global_components['llm_client'],
        llm_model=global_components['llm_model'],
        inference_logs_abs_path=inference_logs_abs,
        action_template_abs_path=expert_prob_template_abs,
        agent_roles=agent_roles,
        alpha=agent_config.get('alpha', 1.0),
        gamma=agent_config.get('gamma', 1.0),
        temperature=agent_config.get('temperature', agent_config.get('aggregation_temperature', 1.0)),
        ema_decay=agent_config.get('ema_decay', 0.9),
        max_retries=act_prob_params.get('max_retries', 5),
        base_delay=act_prob_params.get('base_delay_seconds', 1.0),
        llm_temperature=act_prob_params.get('llm_temperature', 0.7),
        ccn_dependency_variant=global_components['ablation']['ccn_dependency_variant'],
        llm_extra_body=global_components['llm_extra_body'],
        role_shuffle=global_components['ablation']['tom_role_shuffle']
    )
    
    print("ActionProbabilityCalculator initialized")
    
    global_components['backward_engine'] = BackwardInference(
        cep=global_components['cep'],
        llm_client=global_components['llm_client'],
        llm_model=global_components['llm_model'],
        backward_template_abs_path=bwd_template_abs,
        inference_logs_abs_path=inference_logs_abs,
        max_retries=bwd_inf_params.get('max_retries', 5),
        base_delay_seconds=bwd_inf_params.get('base_delay_seconds', 2),
        llm_temperature=bwd_inf_params.get('llm_temperature', 0.7),
        llm_max_tokens=bwd_inf_params.get('llm_max_tokens', 5000),
        agent_roles=agent_roles,
        llm_extra_body=global_components['llm_extra_body']
    )
    
    # Initialize visualizer - use the same directories as DataLogger and CEP
    # Extract the parent directory of inference_logs to get the storage directory
    storage_dir_abs = os.path.dirname(inference_logs_abs)
    print(f"📁 Visualizer storage directory: {storage_dir_abs}")
    print(f"📁 Visualizer inference logs: {inference_logs_abs}")
    global_components['visualizer'] = MentalStateVisualizer(storage_dir=storage_dir_abs)
    global_components['visualization_dir_abs'] = os.path.join(storage_dir_abs, 'visualizations')
    print("Visualizer initialized")
    
    global_components['initialized'] = True
    print(f"Components initialized with provider={active_provider_name}, model={global_components['llm_model']}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    try:
        data_base_dir = config['directories']['data_base_dir']
        data_base_dir_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, data_base_dir)))
        datasets = []
        
        # Scan data directory
        if os.path.exists(data_base_dir_abs):
            for dataset_name in os.listdir(data_base_dir_abs):
                dataset_path = os.path.join(data_base_dir_abs, dataset_name)
                if os.path.isdir(dataset_path) and not dataset_name.startswith('.') and not dataset_name.startswith('__'):
                    # Check if Train/Test/Validation splits exist
                    splits = []
                    for split in ['Train', 'Test', 'Validation']:
                        split_path = os.path.join(dataset_path, split)
                        if os.path.exists(split_path) and os.path.isdir(split_path):
                            splits.append(split)
                    
                    if splits:
                        datasets.append({
                            'name': dataset_name,
                            'splits': splits
                        })
        
        return jsonify({'datasets': datasets})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stocks/<dataset>/<split>', methods=['GET'])
def get_stocks(dataset, split):
    try:
        data_base_dir = config['directories']['data_base_dir']
        data_base_dir_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, data_base_dir)))
        split_path = os.path.join(data_base_dir_abs, dataset, split)
        
        if not os.path.exists(split_path):
            return jsonify({'error': 'Dataset split not found'}), 404
        
        stocks = []
        for stock_name in os.listdir(split_path):
            stock_path = os.path.join(split_path, stock_name)
            if os.path.isdir(stock_path):
                # Check if required data files exist
                required_files = ['text_data.json', 'price_data.json', 'labels.json']
                has_all_files = all(os.path.exists(os.path.join(stock_path, f)) for f in required_files)
                
                if has_all_files:
                    stocks.append(stock_name)
        
        stocks.sort()
        return jsonify({'stocks': stocks})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def extract_mental_state_description(mental_state_json: str) -> str:
    try:
        # Try to parse as JSON
        if mental_state_json.strip().startswith('{'):
            data = json.loads(mental_state_json)
            if isinstance(data, dict) and 'mental state description' in data:
                return data['mental state description']
        return mental_state_json
    except:
        return mental_state_json

def run_inference_background(dataset, split, stock, day_index, window_size):
    global current_inference_state

    try:
        # Load data
        data_base_dir = config['directories']['data_base_dir']
        data_base_dir_abs = resolve_project_path(data_base_dir)
        stock_path = os.path.join(data_base_dir_abs, dataset, split, stock)

        text_data_path = os.path.join(stock_path, 'text_data.json')
        price_data_path = os.path.join(stock_path, 'price_data.json')
        labels_path = os.path.join(stock_path, 'labels.json')

        with open(text_data_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        with open(price_data_path, 'r', encoding='utf-8') as f:
            price_data_raw = json.load(f)

        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_raw = json.load(f)

        price_data = load_price_data(price_data_raw)
        print(f"[INFERENCE] Price data conversion completed, total {len(price_data)} days")

        labels = load_labels(labels_raw)
        print(f"[INFERENCE] Label data conversion completed, total {len(labels)} days")

        # Get data for specified days
        day_keys = sorted(text_data.keys())
        if day_index >= len(day_keys):
            current_inference_state['status'] = 'error'
            current_inference_state['error'] = f'Day index {day_index} out of range (max: {len(day_keys)-1})'
            return

        target_day = day_keys[day_index]

        # Get price information
        price_info = price_data.get(target_day, {})
        print(f"[INFERENCE] Target date: {target_day}, price info: {price_info}")

        # Update progress
        current_inference_state['progress'] = 30
        current_inference_state['current_step'] = 'Preparing environment data...'
        print(f"[INFERENCE] Progress 30%: Preparing environment data...")

        current_inference_state['intermediate_results']['environment'] = {
            'stock': stock,
            'day': target_day,
            'num_texts': 0,
            'sample_texts': [],
            'price_data': price_info
        }

        # Prepare environment data (build text string)
        start_idx = max(0, day_index - window_size + 1)
        window_days = day_keys[start_idx:day_index + 1]

        text_list = []
        for day in window_days:
            day_tweets = text_data.get(day, {})
            if not day_tweets:
                print(f"[INFERENCE] ⚠️ Warning: No tweets found for day {day}")
            for tweet_key in sorted(day_tweets.keys()):
                tweet_data = day_tweets[tweet_key]
                if isinstance(tweet_data, dict) and 'content' in tweet_data:
                    text_list.append(tweet_data['content'])
                elif isinstance(tweet_data, str):
                    text_list.append(tweet_data)

        print(f"[INFERENCE] 📝 Collected {len(text_list)} texts from {len(window_days)} days (day range: {window_days[0] if window_days else 'N/A'} to {window_days[-1] if window_days else 'N/A'})")

        # Build environment state string (consistent format with run.py)
        prices_str = ""
        if price_info:
            for key, value in price_info.items():
                prices_str += f"{key}={value}, "
            prices_str = prices_str.rstrip(", ")

        env_state_text = f"""Market State Description:
            1. Price Conditions:
            {prices_str}

            2. Social Media Tweets (past {len(window_days)} days):
            - {", ".join(text_list) if text_list else ""}
        """

        current_inference_state['intermediate_results']['environment'] = {
            'stock': stock,
            'day': target_day,
            'num_texts': len(text_list),
            'sample_texts': text_list[:5],
            'price_data': price_info
        }
        current_inference_state['progress'] = 38
        current_inference_state['current_step'] = 'Environment data preparation completed'
        print(f"[INFERENCE] ✅ Environment data preparation completed: {len(text_list)}texts")

        # Current core performs belief, intent, and emotion inference per agent.
        inferencer = global_components['inferencer']
        current_inference_state['progress'] = 40
        current_inference_state['current_step'] = 'Running multi-agent forward inference...'
        print("[INFERENCE] Progress 40%: starting multi-agent forward inference...")

        agent_results, generated_filename = inferencer.forward_inference(env_state_text)
        log_data = inferencer.data_logger.load_inference(generated_filename) if hasattr(inferencer.data_logger, 'load_inference') else None
        strategies_used = {}
        if log_data:
            strategies_used = log_data.get('strategies_used', {})
        else:
            log_path = os.path.join(global_components['inference_logs_abs'], generated_filename)
            with open(log_path, 'r', encoding='utf-8') as f:
                strategies_used = json.load(f).get('strategies_used', {})

        state_time = datetime.now().strftime('%H:%M:%S')
        belief = format_agent_dimension(agent_results, 'belief')
        intent = format_agent_dimension(agent_results, 'intent')
        emotion = format_agent_dimension(agent_results, 'emotion')

        current_inference_state['intermediate_results']['belief'] = belief
        current_inference_state['intermediate_results']['intent'] = intent
        current_inference_state['intermediate_results']['emotion'] = emotion
        current_inference_state['intermediate_results']['belief_time'] = state_time
        current_inference_state['intermediate_results']['intent_time'] = state_time
        current_inference_state['intermediate_results']['emotion_time'] = state_time
        current_inference_state['intermediate_results']['agent_results'] = agent_results
        current_inference_state['progress'] = 65
        current_inference_state['current_step'] = f'Multi-agent mental-state inference completed ({state_time})'
        print(f"[INFERENCE] Forward inference completed, log: {generated_filename}")

        # Update progress
        current_inference_state['progress'] = 70
        current_inference_state['current_step'] = 'Calculating action probability...'
        print(f"[INFERENCE] Progress 70%: calculate action probability...")

        calculator = global_components['calculator']
        print(f"[INFERENCE] Calling calculate_probability_from_file...")
        probability_result = calculator.calculate_probability_from_file(generated_filename)
        print(f"[INFERENCE] Probability calculation completed: {probability_result.probability:.4f}")

        # Parse prediction result
        predicted_action_core = "Buy" if probability_result.probability > 0.5 else "Sell"
        predicted_action = core_action_to_display(predicted_action_core)
        confidence = probability_result.probability if probability_result.probability > 0.5 else (1 - probability_result.probability)
        method_used = "Log-Confidence Weighting"  # Fixed method name
        expert_samples = build_expert_samples(probability_result)

        current_inference_state['intermediate_results']['predicted_action'] = predicted_action
        current_inference_state['intermediate_results']['confidence'] = confidence
        current_inference_state['intermediate_results']['expert_samples'] = expert_samples
        current_inference_state['intermediate_results']['agent_predictions'] = probability_result.agent_predictions
        current_inference_state['intermediate_results']['agent_weights'] = probability_result.weights
        current_inference_state['progress'] = 80
        current_inference_state['current_step'] = f'Prediction completed: {predicted_action} (confidence: {confidence:.2%})'
        print(f"[INFERENCE] prediction: {predicted_action}, confidence: {confidence:.2%}")
        print(f"[INFERENCE] Contains {len(expert_samples)} expert judgments")

        # Get true label
        actual_label = labels.get(target_day, 'Unknown')
        actual_action_str = display_action_to_core(actual_label)
        is_correct = predicted_action_core == actual_action_str

        if actual_action_str != 'Unknown':
            for role, pred_info in probability_result.agent_predictions.items():
                agent_correct = pred_info.get('predicted_action') == actual_action_str
                calculator.update_ema_accuracy(role, agent_correct)

        current_inference_state['intermediate_results']['actual_label'] = actual_label
        current_inference_state['intermediate_results']['is_correct'] = is_correct
        print(f"[INFERENCE] Actual label: {actual_label}, prediction correct: {is_correct}")

        # Update progress
        current_inference_state['progress'] = 85
        current_inference_state['current_step'] = 'Checking if backward inference is needed...'

        # backward inference
        data_params = config.get('data_params', {})
        cep_updates_allowed = cep_updates_allowed_for_split(split)
        backward_result = None
        effective_skip_backward = (
            data_params.get('skip_backward_inference', False)
            or not global_components['ablation'].get('backward_enabled', True)
            or not cep_updates_allowed
        )
        if (
            not is_correct
            and actual_action_str != 'Unknown'
            and not effective_skip_backward
        ):
            current_inference_state['progress'] = 90
            current_inference_state['current_step'] = 'Prediction error, executing backward inference...'
            print(f"[INFERENCE] ⚠️ Prediction error detected! Starting backward inference...")

            backward_engine = global_components['backward_engine']
            backward_result = backward_engine.perform_backward_inference(
                filename=generated_filename,
                agent_predictions=probability_result.agent_predictions,
                actual_action=actual_action_str
            )
            if backward_result:
                current_inference_state['intermediate_results']['backward_result'] = backward_result
                current_inference_state['current_step'] = 'Backward inference completed, strategy database updated'
                print(f"[INFERENCE] Backward inference completed: {count_backward_updates(backward_result)} total updates")
        else:
            if is_correct:
                print(f"[INFERENCE] ✅ Prediction correct! No strategy updates needed")
            elif not cep_updates_allowed:
                print(f"[INFERENCE] ℹ️  CEP updates frozen for this benchmark split")
            else:
                print(f"[INFERENCE] ℹ️  Backward inference disabled in configuration")

        # Update progress
        current_inference_state['progress'] = 95
        current_inference_state['current_step'] = 'Generating visualization...'
        print(f"[INFERENCE] ✅ Inference completed! Prediction: {predicted_action}, actual: {actual_label}, correct: {is_correct}")
        
        # Generate visualization
        try:
            visualizer = global_components['visualizer']
            if visualizer:
                print(f"[INFERENCE] 🎨 Generating inference flow visualization...")
                viz_path = visualizer.create_latest_complete_inference_graph()
                if viz_path:
                    # Store visualization filename in state
                    viz_filename = os.path.basename(viz_path)
                    current_inference_state['intermediate_results']['visualization'] = viz_filename
                    print(f"[INFERENCE] ✅ Visualization generated: {viz_filename}")
                else:
                    print(f"[INFERENCE] ⚠️ Visualization generation returned None")
        except Exception as viz_error:
            print(f"[INFERENCE] ⚠️ Visualization generation failed: {str(viz_error)}")
            traceback.print_exc()
        
        current_inference_state['progress'] = 100
        current_inference_state['current_step'] = 'Completed!'
        current_inference_state['status'] = 'completed'

        # Build result
        results = {
            'dataset': dataset,
            'split': split,
            'stock': stock,
            'day': target_day,
            'day_index': day_index,
            'environment_state': {
                'num_texts': len(text_list),
                'sample_texts': text_list[:5],
                'price_data': price_info
            },
            'mental_states': {
                'belief': belief,
                'intent': intent,
                'emotion': emotion
            },
            'agent_results': agent_results,
            'agent_predictions': probability_result.agent_predictions,
            'agent_weights': probability_result.weights,
            'action_prediction': {
                'predicted_action': predicted_action,
                'confidence': confidence,
                'method': method_used
            },
            'actual_label': actual_label,
            'is_correct': is_correct,
            'backward_inference': backward_result,
            'retrieved_strategies': {
                'belief': flatten_strategies(strategies_used, 'belief'),
                'intent': flatten_strategies(strategies_used, 'intent'),
                'emotion': flatten_strategies(strategies_used, 'emotion')
            },
            'timestamp': datetime.now().isoformat()
        }

        current_inference_state['results'] = results

    except Exception as e:
        current_inference_state['status'] = 'error'
        current_inference_state['error'] = str(e)
        current_inference_state['traceback'] = traceback.format_exc()
        
        print(f"Error during inference: {str(e)}")
        print(traceback.format_exc())


@app.route('/api/run_inference', methods=['POST'])
def run_inference():
    global current_inference_state
    
    try:
        # Ensure components are initialized
        if not global_components['initialized']:
            initialize_components()
        
        data = request.json
        dataset = data.get('dataset')
        split = data.get('split')
        stock = data.get('stock')
        day_index = data.get('day_index', 0)
        window_size = data.get('window_size', config['data_params']['default_window_size'])
        
        # Reset state
        current_inference_state = {
            'status': 'running',
            'progress': 10,
            'current_step': 'Load data...',
            'results': None,
            'error': None,
            'intermediate_results': {
                'belief': None,
                'intent': None,
                'emotion': None,
                'belief_strategies': [],
                'intent_strategies': [],
                'emotion_strategies': [],
                'predicted_action': None,
                'confidence': None
            }
        }
        
        print(f"\n{'='*60}")
        print(f"[API] Starting new inference task: {dataset}/{split}/{stock}, day={day_index}, window={window_size}")
        print(f"{'='*60}\n")
        
        # Run inference in background thread
        thread = threading.Thread(
            target=run_inference_background,
            args=(dataset, split, stock, day_index, window_size)
        )
        thread.daemon = True
        thread.start()
        
        print(f"[API] Background thread started")
        
        # Return immediately, do not wait for inference completion
        return jsonify({'success': True, 'message': 'Inference task started'})
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference_status', methods=['GET'])
def get_inference_status():
    print(f"[STATUS] Returning status: {current_inference_state.get('status')}, Progress: {current_inference_state.get('progress')}%, steps: {current_inference_state.get('current_step')}")
    return jsonify(current_inference_state)


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    try:
        strategies = {'belief': [], 'intent': [], 'emotion': []}
        cep = global_components.get('cep')
        if cep:
            roles = global_components.get('agent_roles') or []
            for role in roles:
                for strategy_type in strategies:
                    for item in cep.get_strategies_by_level(strategy_type, role):
                        item_with_role = dict(item)
                        item_with_role.setdefault('agent_role', role)
                        strategies[strategy_type].append(item_with_role)
        else:
            strategy_db_dir = config['directories']['strategy_database']
            strategy_db_dir_abs = resolve_project_path(strategy_db_dir)
            roles = config.get('agent_params', {}).get('agent_roles', ["Retail", "Institutional", "Arbitrageur"])
            for role in roles:
                role_dir = os.path.join(strategy_db_dir_abs, role.lower())
                for strategy_type in strategies:
                    strategy_file = os.path.join(role_dir, f'{strategy_type}_strategies.json')
                    if os.path.exists(strategy_file):
                        with open(strategy_file, 'r', encoding='utf-8') as f:
                            for item in json.load(f):
                                item_with_role = dict(item)
                                item_with_role.setdefault('agent_role', role)
                                strategies[strategy_type].append(item_with_role)
        
        return jsonify({'strategies': strategies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inference_logs', methods=['GET'])
def get_inference_logs():
    try:
        logs_dir = config['directories']['inference_logs']
        logs_dir_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, logs_dir)))
        
        logs = []
        if os.path.exists(logs_dir_abs):
            for log_file in os.listdir(logs_dir_abs):
                if log_file.endswith('.json'):
                    log_path = os.path.join(logs_dir_abs, log_file)
                    mtime = os.path.getmtime(log_path)
                    logs.append({
                        'filename': log_file,
                        'timestamp': datetime.fromtimestamp(mtime).isoformat(),
                        'size': os.path.getsize(log_path)
                    })
        
        logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({'logs': logs[:50]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        safe_config = json.loads(json.dumps(config))  # Deep copy
        # Hide API key
        if 'api' in safe_config:
            for provider in safe_config['api'].get('providers', {}).values():
                if 'api_key' in provider:
                    provider['api_key'] = '***HIDDEN***'
        
        return jsonify({'config': safe_config})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate_visualization', methods=['POST'])
def generate_visualization():
    try:
        if not global_components['initialized']:
            initialize_components()
        
        visualizer = global_components['visualizer']
        if not visualizer:
            return jsonify({
                'success': False,
                'error': 'Visualizer not initialized'
            }), 500
        
        print("[API] Generating visualization...")
        viz_path = visualizer.create_latest_complete_inference_graph()
        
        if viz_path:
            viz_filename = os.path.basename(viz_path)
            print(f"[API] ✅ Visualization generated: {viz_filename}")
            return jsonify({
                'success': True,
                'filename': viz_filename,
                'path': f'/visualizations/{viz_filename}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate visualization'
            }), 500
            
    except Exception as e:
        print(f"[API] ❌ Visualization generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    try:
        from flask import send_from_directory
        viz_dir = global_components.get('visualization_dir_abs')
        if not viz_dir:
            inference_logs_rel = config.get('directories', {}).get('inference_logs', './storage/inference_logs')
            inference_logs_abs = resolve_project_path(inference_logs_rel)
            viz_dir = os.path.join(os.path.dirname(inference_logs_abs), 'visualizations')
        return send_from_directory(viz_dir, filename)
    except Exception as e:
        print(f"[API] ❌ Failed to serve visualization: {str(e)}")
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    print("🚀 Starting MarketToM Web Application...")
    print("📊 Visit http://localhost:8080 to view interface")
    print("💡 Tip: If port 8080 is occupied, modify the port parameter in the last line of app.py")
    app.run(debug=True, host='0.0.0.0', port=8080)
