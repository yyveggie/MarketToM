# -*- coding: utf-8 -*-
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "true"

import sys
import argparse
import json
import logging
import random
from datetime import datetime
import traceback

from core.config_utils import (
    LEGACY_MODEL_PARAM_SECTIONS,
    deep_update,
    get_active_provider_config,
    get_model_params,
    normalize_model_params,
    resolve_api_key,
)

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
    parser.add_argument('--experiment', choices=['single', 'batch', 'analyze'], default='single',
                        help='Experiment mode: single run, batch runner, or analysis')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--preset', type=str, default=None,
                        help='Ablation preset name (e.g. LLM-only, MarketToM-1st, MarketToM-T0)')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all available ablation presets and exit')
    parser.add_argument('--task', default='main',
                        help='Batch/analyze task name, e.g. main')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Batch mode datasets, supports ACL18, StockNet, CMIN-US, CMIN_US, CMIN-CN, CMIN_CN')
    parser.add_argument('--splits', nargs='+', default=None,
                        help='Batch mode dataset splits')
    parser.add_argument('--presets', nargs='+', default=None,
                        help='Batch mode presets')
    parser.add_argument('--models', nargs='*', default=[],
                        help='Batch mode LLM model names')
    parser.add_argument('--ccn-dependency-variants', nargs='+', default=None,
                        help='Batch mode CCN dependency variants')
    parser.add_argument('--prompt-formats', nargs='+', default=None,
                        choices=['xml', 'markdown', 'plain'],
                        help='Batch mode prompt formats')
    parser.add_argument('--prepare-robustness-data', action='store_true',
                        help='Batch mode: create derived robustness datasets if needed')
    parser.add_argument('--provider', default=None,
                        help='Batch mode active LLM provider')
    parser.add_argument('--python', default=sys.executable,
                        help='Batch mode Python executable used for child runs')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0],
                        help='Batch mode random seeds')
    parser.add_argument('--stocks-per-run', type=int, default=5,
                        help='Batch mode number of sampled stocks per seed')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Batch/single generated config sample limit for smoke tests')
    parser.add_argument('--stock-pools-file', default=None,
                        help='Batch mode optional JSON file defining eligible stock pools')
    parser.add_argument('--train-first', action='store_true',
                        help='Batch mode: run Train before evaluation splits with the same strategy database')
    parser.add_argument('--skip-train-first', action='store_true',
                        help='Batch mode: skip task default Train warm-up stage')
    parser.add_argument('--combine-stocks-in-run', action='store_true',
                        help='Batch mode: pass all sampled stocks to a single child run')
    parser.add_argument('--quiet', action='store_true',
                        help='Batch mode: write child-run stdout/stderr to run.log')
    parser.add_argument('--dry-run', action='store_true',
                        help='Batch mode: generate configs and commands without executing child runs')
    parser.add_argument('--force', action='store_true',
                        help='Batch mode: remove existing prediction log before each child run')
    parser.add_argument('--batch-name', default=None,
                        help='Batch mode output directory name')
    parser.add_argument('--predictions', default='outputs/metrics/predictions.csv',
                        help='Analyze mode prediction CSV path')
    parser.add_argument('--output-dir', default='outputs/metrics/stat_tests',
                        help='Analyze mode output directory')
    parser.add_argument('--tests', nargs='+', choices=['pt', 'dm'], default=['pt', 'dm'],
                        help='Analyze mode statistical tests')
    parser.add_argument('--reference-models', nargs='*', default=[],
                        help='Analyze mode reference models for DM tests')
    parser.add_argument('--summary', default='outputs/metrics/batch_experiments/summary.json',
                        help='Analyze mode batch summary JSON for masking consistency / perturbation delta')
    parser.add_argument('--include-splits', nargs='*', default=None,
                        help='Analyze mode split filter for masking consistency / perturbation delta / trading')
    parser.add_argument('--round-trip-bps', type=float, default=10.0,
                        help='Analyze mode round-trip transaction cost in bps for trading simulation')
    parser.add_argument('--logs-dir', default='runtime_storage/inference_logs',
                        help='Analyze mode inference logs dir for external consistency')
    parser.add_argument('--max-instances', type=int, default=None,
                        help='Analyze mode max instances for external consistency (deprecated alias of --sample-size)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Analyze mode sample size for external consistency')
    parser.add_argument('--sampling', choices=['random', 'first', 'all'], default=None,
                        help='Analyze mode sampling rule for external consistency')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Analyze mode sampling seed for external consistency')
    parser.add_argument('--evaluator-provider', default=None,
                        help='Analyze mode evaluator provider for external consistency; must differ from the inference backbone')
    parser.add_argument('--evaluator-model', default=None,
                        help='Analyze mode explicit evaluator model for external consistency')
    parser.add_argument('--allow-same-model', action='store_true',
                        help='Analyze mode: permit an evaluator identical to the inference backbone (not cross-LLM)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Analyze mode: resolve logs, sampling and evaluator without issuing API calls')
    parser.add_argument('--input', default='data/human_eval/ratings.json',
                        help='Analyze mode input data file for human-eval statistics')
    return parser.parse_args()


def load_ablation_presets(config: dict) -> dict:
    return config.get('ablation', {}).get('presets', {})


def apply_preset(config: dict, preset_name: str, presets: dict) -> dict:
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
        elif section_key in LEGACY_MODEL_PARAM_SECTIONS.values():
            model_section = next(
                key for key, legacy_key in LEGACY_MODEL_PARAM_SECTIONS.items()
                if legacy_key == section_key
            )
            config.setdefault('model_params', {}).setdefault(model_section, {})
            deep_update(
                config['model_params'][model_section],
                normalize_model_params(model_section, section_val)
            )
        elif isinstance(section_val, dict) and section_key in config:
            deep_update(config[section_key], section_val)
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


def load_config(config_path: str = 'config.json'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_abs_path = config_path
    if not os.path.isabs(config_abs_path):
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

    if args.experiment == 'analyze':
        if args.task == 'stat_tests':
            from experiments.stat_tests import main as run_stat_tests
            analyze_argv = [
                '--predictions', args.predictions,
                '--output-dir', args.output_dir,
                '--tests', *args.tests,
            ]
            if args.reference_models:
                analyze_argv.extend(['--reference-models', *args.reference_models])
            run_stat_tests(analyze_argv)
            return
        if args.task == 'masking_consistency':
            from experiments.robustness.masking_consistency import main as run_masking_consistency
            analyze_argv = ['--summary', args.summary]
            if args.include_splits:
                analyze_argv.extend(['--include-splits', *args.include_splits])
            if args.output_dir != 'outputs/metrics/stat_tests':
                analyze_argv.extend(['--output-dir', args.output_dir])
            run_masking_consistency(analyze_argv)
            return
        if args.task == 'perturbation_delta':
            from experiments.robustness.perturbation_delta import main as run_perturbation_delta
            analyze_argv = ['--summary', args.summary]
            if args.include_splits:
                analyze_argv.extend(['--include-splits', *args.include_splits])
            if args.output_dir != 'outputs/metrics/stat_tests':
                analyze_argv.extend(['--output-dir', args.output_dir])
            run_perturbation_delta(analyze_argv)
            return
        if args.task == 'target_definition_robustness':
            from experiments.robustness.target_definition_robustness import main as run_target_definition
            analyze_argv = ['--summary', args.summary]
            if args.include_splits:
                analyze_argv.extend(['--include-splits', *args.include_splits])
            if args.output_dir != 'outputs/metrics/stat_tests':
                analyze_argv.extend(['--output-dir', args.output_dir])
            run_target_definition(analyze_argv)
            return
        if args.task == 'trading':
            from experiments.metrics.trading_simulation import main as run_trading
            analyze_argv = ['--summary', args.summary, '--round-trip-bps', str(args.round_trip_bps)]
            if args.include_splits:
                analyze_argv.extend(['--include-splits', *args.include_splits])
            if args.output_dir != 'outputs/metrics/stat_tests':
                analyze_argv.extend(['--output-dir', args.output_dir])
            run_trading(analyze_argv)
            return
        if args.task == 'external_consistency':
            from experiments.metrics.external_consistency import main as run_external_consistency
            analyze_argv = ['--logs-dir', args.logs_dir, '--config', args.config]
            if args.max_instances is not None:
                analyze_argv.extend(['--max-instances', str(args.max_instances)])
            if args.sample_size is not None:
                analyze_argv.extend(['--sample-size', str(args.sample_size)])
            if args.sampling is not None:
                analyze_argv.extend(['--sampling', args.sampling])
            if args.random_seed is not None:
                analyze_argv.extend(['--random-seed', str(args.random_seed)])
            if args.evaluator_provider is not None:
                analyze_argv.extend(['--evaluator-provider', args.evaluator_provider])
            if args.evaluator_model is not None:
                analyze_argv.extend(['--evaluator-model', args.evaluator_model])
            if args.allow_same_model:
                analyze_argv.append('--allow-same-model')
            if args.dry_run:
                analyze_argv.append('--dry-run')
            if args.output_dir != 'outputs/metrics/stat_tests':
                analyze_argv.extend(['--output-dir', args.output_dir])
            run_external_consistency(analyze_argv)
            return
        if args.task == 'human_eval':
            from experiments.metrics.human_eval_stats import main as run_human_eval
            analyze_argv = ['--input', args.input]
            if args.output_dir != 'outputs/metrics/stat_tests':
                analyze_argv.extend(['--output-dir', args.output_dir])
            run_human_eval(analyze_argv)
            return
        if args.task == 'case_study':
            from experiments.case_study import main as run_case_study
            analyze_argv = ['--config', args.config]
            if args.output_dir != 'outputs/metrics/stat_tests':
                analyze_argv.extend(['--output-dir', args.output_dir])
            run_case_study(analyze_argv)
            return
        raise ValueError(f"Unknown analyze task '{args.task}'. Available: stat_tests, masking_consistency, perturbation_delta, target_definition_robustness, trading, external_consistency, human_eval, case_study")

    if args.experiment == 'batch':
        from experiments.markettom_batch import main as run_batch_experiment
        batch_argv = [
            '--task', args.task,
            '--config', args.config,
            '--python', args.python,
            '--seeds', *[str(seed) for seed in args.seeds],
            '--stocks-per-run', str(args.stocks_per_run),
        ]
        if args.datasets:
            batch_argv.extend(['--datasets', *args.datasets])
        if args.splits:
            batch_argv.extend(['--splits', *args.splits])
        if args.presets:
            batch_argv.extend(['--presets', *args.presets])
        if args.models:
            batch_argv.extend(['--models', *args.models])
        if args.ccn_dependency_variants:
            batch_argv.extend(['--ccn-dependency-variants', *args.ccn_dependency_variants])
        if args.prompt_formats:
            batch_argv.extend(['--prompt-formats', *args.prompt_formats])
        if args.prepare_robustness_data:
            batch_argv.append('--prepare-robustness-data')
        if args.provider:
            batch_argv.extend(['--provider', args.provider])
        if args.max_samples is not None:
            batch_argv.extend(['--max-samples', str(args.max_samples)])
        if args.stock_pools_file:
            batch_argv.extend(['--stock-pools-file', args.stock_pools_file])
        if args.train_first:
            batch_argv.append('--train-first')
        if args.skip_train_first:
            batch_argv.append('--skip-train-first')
        if args.combine_stocks_in_run:
            batch_argv.append('--combine-stocks-in-run')
        if args.quiet:
            batch_argv.append('--quiet')
        if args.dry_run:
            batch_argv.append('--dry-run')
        if args.force:
            batch_argv.append('--force')
        if args.batch_name:
            batch_argv.extend(['--batch-name', args.batch_name])
        run_batch_experiment(batch_argv)
        return

    import openai
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
    from data.data_input import load_stock_data

    # --- List presets and exit ---
    if args.list_presets:
        config = load_config(args.config)
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
                T = get_model_params(p, 'forward').get('llm_temperature', '—')
                print(f"  {name:30s}  mode={mode}, tom={tom}, T={T}")
                if desc:
                    print(f"    {'':30s}  {desc}")
            print()
        return

    # ============ 1. Load Configuration ============
    print("\n\033[1;36m=== STEP 1: LOADING CONFIGURATION ===\033[0m")
    config = load_config(args.config)
    api_rate_limit_config = config.get('api_rate_limit', {})
    if api_rate_limit_config:
        for module in (forward_inference_module, action_probability_module, backward_inference_module):
            if 'min_request_interval' in api_rate_limit_config:
                module.MIN_REQUEST_INTERVAL = float(api_rate_limit_config['min_request_interval'])
            if 'default_cooldown' in api_rate_limit_config:
                module.DEFAULT_COOLDOWN = float(api_rate_limit_config['default_cooldown'])
            if 'max_jitter' in api_rate_limit_config:
                module.MAX_JITTER = float(api_rate_limit_config['max_jitter'])

    # Apply ablation preset if requested
    if args.preset:
        presets = load_ablation_presets(config)
        config = apply_preset(config, args.preset, presets)
    llm_client = None
    llm_model_to_use = None

    active_provider_name, active_provider_config = get_active_provider_config(config)
    if not active_provider_config:
        raise ValueError(f"LLM provider config not found: {active_provider_name}")
    api_key = resolve_api_key(active_provider_config)
    if not api_key:
        raise ValueError(f"API key not found for provider '{active_provider_name}'")
    base_url = active_provider_config.get('base_url')
    client_kwargs = {'api_key': api_key}
    if base_url and base_url.strip():
        client_kwargs['base_url'] = base_url
    provider_timeout = active_provider_config.get('timeout')
    if provider_timeout:
        client_kwargs['timeout'] = float(provider_timeout)
    llm_extra_body = active_provider_config.get('extra_body')
    llm_client = openai.OpenAI(**client_kwargs)
    llm_model_to_use = active_provider_config.get('llm_model_default', 'gpt-4o')
    endpoint_label = "custom endpoint" if base_url and base_url.strip() else "official endpoint"
    print(f"\033[32m✅ Connected to {active_provider_name} ({endpoint_label})\033[0m")

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
    max_samples = data_params_config.get('max_samples')
    cep_update_mode = data_params_config.get('cep_update_mode', 'benchmark').lower()
    benchmark_split = dataset_split.lower()
    cep_updates_allowed = (
        cep_update_mode in {'online', 'realtime', 'real_time', 'live'}
        or benchmark_split in {'train', 'training'}
    )

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
    ccn_dependency_variant = ablation_config.get('ccn_dependency_variant', 'full')
    ablation_role_shuffle = ablation_config.get('tom_role_shuffle', False)
    experiment_name = ablation_config.get('experiment_name', 'MarketToM-2nd')

    # Override derived flags from ablation mode
    if ablation_mode == 'llm_only':
        ablation_cep_enabled = False
        ablation_backward_enabled = False
    elif ablation_mode == 'no_cep':
        ablation_cep_enabled = False
        ablation_backward_enabled = False

    # skip_backward_inference from data_params OR ablation config
    effective_skip_backward = skip_backward_inference or (not ablation_backward_enabled) or (not cep_updates_allowed)

    print(f"\033[1;33m📋 Experiment: {experiment_name}  "
          f"[mode={ablation_mode}, tom={ablation_tom_order}, "
          f"cep={ablation_cep_enabled}, "
          f"cep_update_mode={cep_update_mode}, cep_updates_allowed={cep_updates_allowed}, "
          f"ccn_dependency_variant={ccn_dependency_variant}, role_shuffle={ablation_role_shuffle}]\033[0m")

    # --- CEP Retrieval Parameters ---
    cep_retrieval_config = config.get('cep_retrieval', {})
    cep_default_top_k = cep_retrieval_config.get('default_top_k', 1)
    cep_similarity_threshold = cep_retrieval_config.get('similarity_threshold', 0.1)
    emotion_similarity_threshold = cep_retrieval_config.get('emotion_similarity_threshold', 0.1)
    belief_similarity_threshold = cep_retrieval_config.get('belief_similarity_threshold', 0.1)
    intent_similarity_threshold = cep_retrieval_config.get('intent_similarity_threshold', 0.1)

    # --- Module-specific params ---
    fwd_inf_params = get_model_params(config, 'forward')
    act_prob_params = get_model_params(config, 'action_probability')
    bwd_inf_params = get_model_params(config, 'backward')

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
        cep_enabled=ablation_cep_enabled,
        ccn_dependency_variant=ccn_dependency_variant,
        llm_extra_body=llm_extra_body
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
        max_retries=act_prob_params.get('max_retries', 5),
        base_delay=act_prob_params.get('base_delay_seconds', 1.0),
        llm_temperature=act_prob_params.get('llm_temperature', 0.7),
        ccn_dependency_variant=ccn_dependency_variant,
        llm_extra_body=llm_extra_body,
        role_shuffle=ablation_role_shuffle
    )
    print(f"\033[32m✅ Dynamic weighted aggregation ready\033[0m")

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
        llm_max_tokens=bwd_inf_params.get('llm_max_tokens', 5000),
        llm_extra_body=llm_extra_body
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
    available_samples = max(0, length - default_window_size + 1)
    if max_samples is not None:
        available_samples = min(available_samples, max(0, int(max_samples)))
        print(f"\033[33m⚠ Sample limit active: {available_samples} samples\033[0m")
    loop_stop = default_window_size + available_samples

    def build_env_state(sample_index):
        window_texts = []
        window_prices = []
        for j in range(sample_index - default_window_size, sample_index):
            day_str = default_stocks[0] + "day" + f"{j:0{num_digits}d}"
            tweets = train_text_data.get(day_str, {})
            if tweets:
                window_texts.extend([tweet['content'] for tweet in tweets.values()])
            window_prices.append(train_price_data[j])
        prices_str = "".join([
            f"Day {idx + 1}: Open={p[0]}, High={p[1]}, Low={p[2]}, Close={p[3]}, Volume={p[4]}"
            for idx, p in enumerate(window_prices)
        ])
        return f"""Market State Description:
            1. Price Conditions:
            {prices_str}

            2. Social Media Tweets (past {default_window_size} days):
            - {", ".join(window_texts)}
        """

    belief_env_map = None
    if ccn_dependency_variant == "shuffled_belief_parents":
        all_indices = list(range(default_window_size, loop_stop))
        env_by_index = {idx: build_env_state(idx) for idx in all_indices}
        permuted = all_indices[:]
        random.Random(42).shuffle(permuted)
        belief_env_map = {idx: env_by_index[perm] for idx, perm in zip(all_indices, permuted)}

    for i in range(default_window_size, loop_stop):
        if i in done_indices:
            logger.info(f"Skipping sample {i} (already processed)")
            continue

        print(f"\n\033[1;35m{'='*60}\033[0m")
        print(f"\033[1;35m  Trading day {i}/{length}\033[0m")
        print(f"\033[1;35m{'='*60}\033[0m")

        env_state = build_env_state(i)
        label = train_labels[i - 1]
        belief_env_state = belief_env_map[i] if belief_env_map else None

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
                llm_only_kwargs = {
                    "model": llm_model_to_use,
                    "messages": [{"role": "user", "content": llm_only_prompt}],
                    "temperature": fwd_inf_params.get('llm_temperature', 0.7),
                    "max_tokens": 10,
                }
                if llm_extra_body:
                    llm_only_kwargs["extra_body"] = llm_extra_body
                llm_resp = llm_client.chat.completions.create(**llm_only_kwargs)
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
            agent_results, generated_filename = inferencer.forward_inference(env_state, belief_env_state)
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
                elif not is_correct and not cep_updates_allowed:
                    print("\033[33m⚠ CEP updates frozen for this benchmark split\033[0m")

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
