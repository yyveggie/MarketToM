"""
MarketToM Web Application
Flask backend service for real-time visualization interface
"""

import os
import sys
import json
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import openai

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import core modules
from core import (
    CognitiveEnhancementPlugin,
    MentalStateInference,
    DataLogger,
    ActionProbabilityCalculator,
    BackwardInference
)
from visualization import MentalStateVisualizer

app = Flask(__name__)
CORS(app)

# Disable static file caching in development mode
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Add response headers to disable caching
@app.after_request
def add_header(response):
    """Add response headers to prevent browser caching"""
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
    'visualizer': None
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


def initialize_components():
    """Initialize all components"""
    if global_components['initialized']:
        return
    
    print("Initializing MarketToM components...")
    
    # Parse API configuration
    api_config = config.get('api', {})
    active_provider_name = api_config.get('active_llm_provider', 'openai').lower()
    provider_configs = api_config.get('providers', {})
    
    # Initialize LLM client
    if active_provider_name == 'openai':
        openai_config = provider_configs.get('openai', {})
        base_url = openai_config.get('base_url')
        if base_url and base_url.strip():
            global_components['llm_client'] = openai.OpenAI(
                api_key=openai_config.get('api_key'),
                base_url=base_url
            )
        else:
            global_components['llm_client'] = openai.OpenAI(
                api_key=openai_config.get('api_key')
            )
        global_components['llm_model'] = openai_config.get('llm_model_default', 'gpt-4o')
    
    # Parse directory configuration (all paths relative to project root)
    directories_config = config.get('directories', {})
    
    strategy_database_rel = directories_config.get('strategy_database', './storage/strategy_database')
    strategy_database_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, strategy_database_rel)))
    
    inference_logs_rel = directories_config.get('inference_logs', './storage/inference_logs')
    inference_logs_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, inference_logs_rel)))
    
    # Parse template paths
    templates_config = config.get('templates', {})
    fwd_template_rel = templates_config.get('forward_inference', './templates/forward_prompt_template.xml')
    fwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, fwd_template_rel)))
    
    act_prob_template_rel = templates_config.get('action_probability', './templates/action_prob_prompt_template.xml')
    act_prob_template_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, act_prob_template_rel)))
    
    expert_prob_template_rel = templates_config.get('expert_action_probability', './templates/expert_action_prob_template.xml')
    expert_prob_template_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, expert_prob_template_rel)))
    
    bwd_template_rel = templates_config.get('backward_inference', './templates/backward_prompt_template.xml')
    bwd_template_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, bwd_template_rel)))
    
    # Parse parameters
    cep_retrieval_config = config.get('cep_retrieval', {})
    fwd_inf_params = config.get('forward_inference_params', {})
    act_prob_params = config.get('action_probability_params', {})
    bwd_inf_params = config.get('backward_inference_params', {})
    
    # Create necessary directories
    os.makedirs(strategy_database_abs, exist_ok=True)
    os.makedirs(inference_logs_abs, exist_ok=True)
    
    # Initialize components
    global_components['cep'] = CognitiveEnhancementPlugin(storage_path=strategy_database_abs)
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
        llm_temperature=fwd_inf_params.get('llm_temperature', 0.7)
    )
    
    use_expert_method = act_prob_params.get('use_expert_perspective_method', False)
    
    global_components['calculator'] = ActionProbabilityCalculator(
        cep=global_components['cep'],
        llm_client=global_components['llm_client'],
        llm_model=global_components['llm_model'],
        action_prob_template_abs_path=act_prob_template_abs,
        inference_logs_abs_path=inference_logs_abs,
        action_prob_top_k=cep_retrieval_config.get('action_prob_top_k', 2),
        num_probs_to_generate=act_prob_params.get('num_probabilities_to_generate', 10),
        max_retries_list=act_prob_params.get('max_retries_list', 5),
        max_retries_logprobs=act_prob_params.get('max_retries_logprobs', 5),
        base_delay_list_seconds=act_prob_params.get('base_delay_list_seconds', 1),
        base_delay_logprobs_seconds=act_prob_params.get('base_delay_logprobs_seconds', 1),
        expert_prob_method=use_expert_method,
        expert_template_abs_path=expert_prob_template_abs if use_expert_method else None,
        llm_temperature=act_prob_params.get('llm_temperature', 0.7)
    )
    
    print(f"✅ ActionProbabilityCalculator initialized, expert mode: {'enabled' if use_expert_method else 'disabled'}")
    
    global_components['backward_engine'] = BackwardInference(
        cep=global_components['cep'],
        llm_client=global_components['llm_client'],
        llm_model=global_components['llm_model'],
        backward_template_abs_path=bwd_template_abs,
        inference_logs_abs_path=inference_logs_abs,
        max_retries=bwd_inf_params.get('max_retries', 5),
        base_delay_seconds=bwd_inf_params.get('base_delay_seconds', 2),
        llm_temperature=bwd_inf_params.get('llm_temperature', 0.7),
        llm_max_tokens=bwd_inf_params.get('llm_max_tokens', 5000)
    )
    
    # Initialize visualizer - use the same directories as DataLogger and CEP
    # Extract the parent directory of inference_logs to get the storage directory
    storage_dir_abs = os.path.dirname(inference_logs_abs)
    print(f"📁 Visualizer storage directory: {storage_dir_abs}")
    print(f"📁 Visualizer inference logs: {inference_logs_abs}")
    global_components['visualizer'] = MentalStateVisualizer(storage_dir=storage_dir_abs)
    print("✅ Visualizer initialized")
    
    global_components['initialized'] = True
    print("✅ Components initialization completed")


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get available dataset list"""
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
    """Get stock list for specified dataset and split"""
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
    """
    Extract 'mental state description' value from JSON string
    Returns the description text or the original string if parsing fails
    """
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
    """Run inference in background thread"""
    global current_inference_state
    
    try:
        
        # Load data
        data_base_dir = config['directories']['data_base_dir']
        data_base_dir_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, data_base_dir)))
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
        
        # 🔧 Convert price data array to dictionary (day -> price_info)
        price_data = {}
        if isinstance(price_data_raw, dict) and 'price_data' in price_data_raw:
            # Format: {"price_data": [{"day": "day1", ...}, ...]}
            for item in price_data_raw['price_data']:
                if 'day' in item:
                    price_data[item['day']] = {k: v for k, v in item.items() if k != 'day'}
        elif isinstance(price_data_raw, dict):
            # Format: {"day1": {...}, "day2": {...}}
            price_data = price_data_raw
        
        print(f"[INFERENCE] Price data conversion completed, total {len(price_data)} days")
        
        # 🔧 Convert label data array to dictionary (day -> "Up"/"Down")
        labels = {}
        if isinstance(labels_raw, dict) and 'labels' in labels_raw:
            # Format: {"labels": [{"day": "day1", "label": 1}, ...]}
            for item in labels_raw['labels']:
                if 'day' in item and 'label' in item:
                    # label: 1 -> "Up", 0 -> "Down"
                    labels[item['day']] = "Up" if item['label'] == 1 else "Down"
        elif isinstance(labels_raw, dict):
            # Format: {"day1": 1, "day2": 0} or {"day1": "Up", "day2": "Down"}
            for day, label_value in labels_raw.items():
                if isinstance(label_value, int):
                    labels[day] = "Up" if label_value == 1 else "Down"
                else:
                    labels[day] = label_value
        
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
        
        # 🆕 Display environment data immediately
        current_inference_state['intermediate_results']['environment'] = {
            'stock': stock,
            'day': target_day,
            'num_texts': 0,  # Update later
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
        # Build price conditions string
        prices_str = ""
        if price_info:
            for key, value in price_info.items():
                prices_str += f"{key}={value}, "
            prices_str = prices_str.rstrip(", ")
        
        # Build environment state (matching run.py format)
        env_state_text = f"""Market State Description:
            1. Price Conditions:
            {prices_str}

            2. Social Media Tweets (past {len(window_days)} days):
            - {", ".join(text_list) if text_list else ""}
        """
        
        # 🆕 Update complete environment data
        current_inference_state['intermediate_results']['environment'] = {
            'stock': stock,
            'day': target_day,
            'num_texts': len(text_list),
            'sample_texts': text_list[:5],
            'price_data': price_info
        }
        current_inference_state['progress'] = 38
        current_inference_state['current_step'] = '✓ Environment data preparation completed'
        print(f"[INFERENCE] ✅ Environment data preparation completed: {len(text_list)}texts")
        
        # 🆕 Execute inference step by step, display immediately upon completion
        inferencer = global_components['inferencer']
        
        # Step 1: infer belief
        current_inference_state['progress'] = 40
        current_inference_state['current_step'] = 'Inferring market belief...'
        print(f"[INFERENCE] Progress 40%: starting belief inference...")
        
        belief, belief_strategies = inferencer.infer_market_belief(env_state_text)
        
        # ✨ Update and display belief immediately (extract description only)
        belief_time = datetime.now().strftime('%H:%M:%S')
        current_inference_state['intermediate_results']['belief'] = extract_mental_state_description(belief)
        current_inference_state['intermediate_results']['belief_strategies'] = belief_strategies
        current_inference_state['intermediate_results']['belief_time'] = belief_time
        current_inference_state['progress'] = 50
        current_inference_state['current_step'] = f'✓ Belief inference completed ({belief_time})'
        print(f"[{belief_time}] [INFERENCE] ✅ Belief completed: {belief[:80]}...")
        print(f"[{belief_time}] [INFERENCE] 🔄 Frontend polls every 0.5s, visible with max 0.5s delay!")
        
        # Step 2: infer intent
        current_inference_state['progress'] = 55
        current_inference_state['current_step'] = 'Inferring market intent...'
        print(f"[INFERENCE] Progress 55%: starting intent inference...")
        
        intent, intent_strategies = inferencer.infer_market_intent(belief)
        
        # ✨ Update and display intent immediately (extract description only)
        intent_time = datetime.now().strftime('%H:%M:%S')
        current_inference_state['intermediate_results']['intent'] = extract_mental_state_description(intent)
        current_inference_state['intermediate_results']['intent_strategies'] = intent_strategies
        current_inference_state['intermediate_results']['intent_time'] = intent_time
        current_inference_state['progress'] = 60
        current_inference_state['current_step'] = f'✓ Intent inference completed ({intent_time})'
        print(f"[{intent_time}] [INFERENCE] ✅ Intent completed: {intent[:80]}...")
        print(f"[{intent_time}] [INFERENCE] 🔄 Frontend will display within 0.5s!")
        
        # Step 3: infer emotion
        current_inference_state['progress'] = 62
        current_inference_state['current_step'] = 'Inferring market emotion...'
        print(f"[INFERENCE] Progress 62%: starting emotion inference...")
        
        emotion, emotion_strategies = inferencer.infer_market_emotion(belief, env_state_text)
        
        # ✨ Update and display emotion immediately (extract description only)
        emotion_time = datetime.now().strftime('%H:%M:%S')
        current_inference_state['intermediate_results']['emotion'] = extract_mental_state_description(emotion)
        current_inference_state['intermediate_results']['emotion_strategies'] = emotion_strategies
        current_inference_state['intermediate_results']['emotion_time'] = emotion_time
        current_inference_state['progress'] = 65
        current_inference_state['current_step'] = f'✓ All mental state analysis completed ({emotion_time})'
        print(f"[{emotion_time}] [INFERENCE] ✅ Emotion completed: {emotion[:80]}...")
        print(f"[{emotion_time}] [INFERENCE] 🔄 Frontend will show all mental states within 0.5s!")
        
        # Save inference log
        data_logger = global_components['data_logger']
        timestamp = datetime.now()
        data_logger.save_inference(
            timestamp=timestamp,
            env_state=env_state_text,  # 保存完整的环境状态，不要截断！
            mental_states={'belief': belief, 'intent': intent, 'emotion': emotion},
            strategies_used={'belief': belief_strategies, 'intent': intent_strategies, 'emotion': emotion_strategies}
        )
        generated_filename = f"inference_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        print(f"[INFERENCE] 💾 Saved inference log: {generated_filename} (env_state length: {len(env_state_text)} chars, texts: {len(text_list)})")
        
        # Update progress
        current_inference_state['progress'] = 70
        current_inference_state['current_step'] = 'calculate action probability...'
        print(f"[INFERENCE] Progress 70%: calculate action probability...")
        
        # Using generated filecalculate action probability
        calculator = global_components['calculator']
        print(f"[INFERENCE] Calling calculate_probability_from_file...")
        probability_result = calculator.calculate_probability_from_file(generated_filename)
        print(f"[INFERENCE] Probability calculation completed: {probability_result.probability:.4f}")
        
        # Parse prediction result
        predicted_action = "Up" if probability_result.probability > 0.5 else "Down"
        confidence = probability_result.probability if probability_result.probability > 0.5 else (1 - probability_result.probability)
        method_used = "Log-Confidence Weighting"  # Fixed method name
        
        # 🆕 Real-time updatepredictionresult（Containsexpert judgments）
        # Preferentially useexpert_details（Containsroleandreasoning），otherwise fallback tosamples
        if hasattr(probability_result, 'expert_details') and probability_result.expert_details:
            expert_samples = [
                {
                    'index': i + 1,
                    'role': detail['role'],
                    'reasoning': detail['reasoning'],
                    'probability': detail['probability'],
                    'log_confidence': detail['log_confidence'],
                    'normalized_weight': detail['normalized_weight']
                }
                for i, detail in enumerate(probability_result.expert_details)
            ]
            print(f"[INFERENCE] ✅ Using complete expert details (including role and reasoning)")
        else:
            expert_samples = [
                {
                    'index': i + 1,
                    'role': 'Unknown expert',
                    'reasoning': 'No reasoning info',
                    'probability': sample.value,
                    'log_confidence': sample.log_confidence,
                    'normalized_weight': sample.normalized_weight
                }
                for i, sample in enumerate(probability_result.samples)
            ]
            print(f"[INFERENCE] ⚠️ Using simplified expert data (no role and reasoning info)")
        
        current_inference_state['intermediate_results']['predicted_action'] = predicted_action
        current_inference_state['intermediate_results']['confidence'] = confidence
        current_inference_state['intermediate_results']['expert_samples'] = expert_samples
        current_inference_state['progress'] = 80
        current_inference_state['current_step'] = f'✓ Prediction completed: {predicted_action} (confidence: {confidence:.2%})'
        print(f"[INFERENCE] prediction: {predicted_action}, confidence: {confidence:.2%}")
        print(f"[INFERENCE] Contains {len(expert_samples)} expert judgments")
        
        # Get true label
        actual_label = labels.get(target_day, 'Unknown')
        is_correct = (predicted_action.lower() == actual_label.lower())
        
        # 🆕 Immediately updateActual labeland accuracy
        current_inference_state['intermediate_results']['actual_label'] = actual_label
        current_inference_state['intermediate_results']['is_correct'] = is_correct
        print(f"[INFERENCE] Actual label: {actual_label}, prediction correct: {is_correct}")
        
        # Update progress
        current_inference_state['progress'] = 85
        current_inference_state['current_step'] = 'Checking if backward inference is needed...'
        
        # backward inference
        backward_updates = None
        if not is_correct and not config['data_params'].get('skip_backward_inference', False):
            current_inference_state['progress'] = 90
            current_inference_state['current_step'] = '⚠️ Prediction error, executing backward inference...'
            print(f"[INFERENCE] ⚠️ Prediction error detected! Starting backward inference...")
            
            backward_engine = global_components['backward_engine']
            backward_result = backward_engine.perform_backward_inference(
                filename=generated_filename,
                predicted_action=predicted_action,
                actual_action=actual_label
            )
            if backward_result:
                # 新格式：字典 {'analysis': str, 'strategy_updates': dict}
                analysis_text = backward_result.get('analysis', '')
                strategy_updates = backward_result.get('strategy_updates', {})
                
                # 🆕 Real-time update backward inference result (frontend expects 'backward_result')
                current_inference_state['intermediate_results']['backward_result'] = {
                    'analysis': analysis_text,
                    'strategy_updates': strategy_updates
                }
                current_inference_state['intermediate_results']['backward_analysis'] = analysis_text
                current_inference_state['intermediate_results']['strategy_updates'] = strategy_updates
                
                # 打印策略更新详情
                print(f"\n{'='*60}")
                print(f"[INFERENCE] ✅ Backward inference completed")
                
                if strategy_updates:
                    total_updates = sum(len(updates) for updates in strategy_updates.values())
                    print(f"[INFERENCE] 📊 Updated {total_updates} strategies:")
                    
                    for level, updates in strategy_updates.items():
                        if updates:
                            print(f"\n[INFERENCE]   🔹 {level.upper()} ({len(updates)} updates):")
                            for update in updates:
                                update_type = update.get('type', 'N/A')
                                update_id = update.get('id', 'N/A')
                                content_preview = update.get('content', 'N/A')[:100]
                                print(f"[INFERENCE]     - {update_type} [{update_id}]")
                                print(f"[INFERENCE]       {content_preview}...")
                else:
                    print(f"[INFERENCE] ℹ️  No strategy updates needed")
                
                print(f"[INFERENCE] 📝 Analysis preview: {str(analysis_text)[:200]}...")
                print(f"{'='*60}\n")
                
                current_inference_state['current_step'] = '✓ Backward inference completed, Strategy database updated'
        else:
            # Prediction was correct or backward inference disabled
            if is_correct:
                print(f"[INFERENCE] ✅ Prediction correct! No strategy updates needed")
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
            import traceback
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
            'action_prediction': {
                'predicted_action': predicted_action,
                'confidence': confidence,
                'method': method_used
            },
            'actual_label': actual_label,
            'is_correct': is_correct,
            'backward_inference': backward_updates,
            'retrieved_strategies': {
                'belief': [{'content': s} for s in belief_strategies] if belief_strategies else [],
                'intent': [{'content': s} for s in intent_strategies] if intent_strategies else [],
                'emotion': [{'content': s} for s in emotion_strategies] if emotion_strategies else []
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
    """Start inference task"""
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
    """Get current inference status"""
    print(f"[STATUS] Returning status: {current_inference_state.get('status')}, Progress: {current_inference_state.get('progress')}%, steps: {current_inference_state.get('current_step')}")
    return jsonify(current_inference_state)


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get current strategy database"""
    try:
        strategy_db_dir = config['directories']['strategy_database']
        strategy_db_dir_abs = os.path.normpath(os.path.abspath(os.path.join(project_root, strategy_db_dir)))
        
        strategies = {}
        for strategy_type in ['belief', 'intent', 'emotion']:
            strategy_file = os.path.join(strategy_db_dir_abs, f'{strategy_type}_strategies.json')
            if os.path.exists(strategy_file):
                with open(strategy_file, 'r', encoding='utf-8') as f:
                    strategies[strategy_type] = json.load(f)
            else:
                strategies[strategy_type] = []
        
        return jsonify({'strategies': strategies})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/inference_logs', methods=['GET'])
def get_inference_logs():
    """Get inference logs list"""
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
    """Get current configuration (hide sensitive info)"""
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
    """Generate visualization for the latest inference"""
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
    """Serve visualization images"""
    try:
        viz_dir = os.path.join(project_root, 'storage', 'visualizations')
        from flask import send_from_directory
        return send_from_directory(viz_dir, filename)
    except Exception as e:
        print(f"[API] ❌ Failed to serve visualization: {str(e)}")
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    print("🚀 Starting MarketToM Web Application...")
    print("📊 Visit http://localhost:8080 to view interface")
    print("💡 Tip: If port 8080 is occupied, modify the port parameter in the last line of app.py")
    app.run(debug=True, host='0.0.0.0', port=8080)

