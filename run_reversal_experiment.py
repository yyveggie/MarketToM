# -*- coding: utf-8 -*-
"""
Run Reversal Experiment: Main program for reversal rate experiment
Integrates stance classification and reversal analysis for E.2 experiment
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import openai
from core.stance_classifier import StanceClassifier, add_stances_to_inference_log
from analysis.reversal_analyzer import ReversalAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ReversalExperiment')

COLOR_TITLE = "\033[1;36m"
COLOR_SUCCESS = "\033[1;32m"
COLOR_WARNING = "\033[1;33m"
COLOR_ERROR = "\033[1;31m"
COLOR_INFO = "\033[0;34m"
COLOR_VALUE = "\033[1;35m"
COLOR_DEBUG = "\033[0;90m"
COLOR_RESET = "\033[0m"


def load_config(config_path: str = None) -> dict:
    """Load configuration file"""
    if config_path is None:
        config_path = os.path.join(project_root, 'config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def initialize_llm_client(config: dict):
    """Initialize LLM client"""
    api_config = config.get('api', {})
    active_provider = api_config.get('active_llm_provider', 'openai').lower()
    providers = api_config.get('providers', {})
    
    if active_provider == 'openai':
        openai_config = providers.get('openai', {})
        base_url = openai_config.get('base_url')
        
        if base_url and base_url.strip():
            client = openai.OpenAI(
                api_key=openai_config.get('api_key'),
                base_url=base_url
            )
        else:
            client = openai.OpenAI(
                api_key=openai_config.get('api_key')
            )
        
        model = openai_config.get('llm_model_default', 'gpt-4o')
        return client, model
    
    else:
        raise ValueError(f"Unsupported provider: {active_provider}")


def classify_existing_logs(inference_logs_dir: str, classifier: StanceClassifier, 
                          force_reclassify: bool = False, verbose: bool = True) -> int:
    """
    Incremental stance classification for inference logs with real-time saving
    
    Args:
        inference_logs_dir: Directory containing inference logs
        classifier: Stance classifier instance
        force_reclassify: Force re-classification even if stances exist
        verbose: Show detailed output for each classification
    
    Returns:
        Number of newly classified logs
    """
    print(f"\n{COLOR_TITLE}=== STEP 1: INCREMENTAL STANCE CLASSIFICATION ==={COLOR_RESET}")
    
    log_files = sorted([f for f in os.listdir(inference_logs_dir) 
                        if f.startswith('inference_') and f.endswith('.json')])
    
    total_files = len(log_files)
    print(f"{COLOR_INFO}Found {total_files} inference logs{COLOR_RESET}")
    
    already_classified = []
    need_classification = []
    
    print(f"{COLOR_DEBUG}Checking existing classifications...{COLOR_RESET}", end='')
    for log_file in log_files:
        log_path = os.path.join(inference_logs_dir, log_file)
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            if 'mental_state_stances' in log_data and not force_reclassify:
                already_classified.append(log_file)
            else:
                need_classification.append(log_file)
        except Exception as e:
            logger.warning(f"Error reading {log_file}: {str(e)}")
            need_classification.append(log_file)
    
    print(f"\r{COLOR_SUCCESS}✓ Classification status:{COLOR_RESET}")
    print(f"  {COLOR_VALUE}Already classified:{COLOR_RESET} {len(already_classified)}/{total_files}")
    print(f"  {COLOR_WARNING}Need classification:{COLOR_RESET} {len(need_classification)}/{total_files}")
    
    if len(need_classification) == 0:
        print(f"\n{COLOR_SUCCESS}✓ All logs already classified!{COLOR_RESET}")
        return 0
    
    classified_count = 0
    
    print(f"\n{COLOR_TITLE}Starting incremental classification (Ctrl+C to pause)...{COLOR_RESET}")
    
    for i, log_file in enumerate(need_classification, 1):
        log_path = os.path.join(inference_logs_dir, log_file)
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            mental_states = log_data.get('mental_states', {})
            if not mental_states:
                logger.warning(f"No mental states in {log_file}")
                continue
            
            sample_id = log_data.get('sample_id', 'unknown')
            timestamp = log_data.get('timestamp', 'unknown')
            
            print(f"\n{COLOR_TITLE}{'─'*70}{COLOR_RESET}")
            print(f"{COLOR_INFO}[{i}/{len(need_classification)}] Sample ID: {sample_id}{COLOR_RESET}")
            print(f"{COLOR_DEBUG}├─ File: {log_file}{COLOR_RESET}")
            print(f"{COLOR_DEBUG}├─ Timestamp: {timestamp}{COLOR_RESET}")
            print(f"{COLOR_DEBUG}└─ Progress: {len(already_classified) + classified_count + 1}/{total_files} total ({((len(already_classified) + classified_count + 1)/total_files*100):.1f}%){COLOR_RESET}")
            
            stances = classifier.classify_all_states(mental_states, verbose=verbose)
            
            add_stances_to_inference_log(log_path, stances)
            classified_count += 1
            
            print(f"\n{COLOR_SUCCESS}✓ Classification saved to file:{COLOR_RESET}")
            for state_name, stance_info in stances.items():
                stance = stance_info['stance']
                confidence = stance_info['confidence']
                
                if stance == 'UP':
                    stance_color = "\033[1;32m"
                elif stance == 'DOWN':
                    stance_color = "\033[1;31m"
                else:
                    stance_color = "\033[1;33m"
                
                print(f"  {state_name:10s}: {stance_color}{stance:>6s}{COLOR_RESET} "
                      f"(confidence: {COLOR_VALUE}{confidence:.2f}{COLOR_RESET})")
            
            logger.info(f"Classified and saved: {log_file} ({classified_count}/{len(need_classification)})")
            
        except KeyboardInterrupt:
            print(f"\n\n{COLOR_WARNING}⚠ Classification paused by user{COLOR_RESET}")
            print(f"{COLOR_INFO}Progress saved: {classified_count}/{len(need_classification)} files classified{COLOR_RESET}")
            print(f"{COLOR_INFO}Total: {len(already_classified) + classified_count}/{total_files} files have classifications{COLOR_RESET}")
            print(f"\n{COLOR_DEBUG}You can resume by running the same command again.{COLOR_RESET}")
            return classified_count
            
        except Exception as e:
            logger.error(f"Error processing {log_file}: {str(e)}")
            print(f"{COLOR_ERROR}✗ Failed to classify {log_file}: {str(e)}{COLOR_RESET}")
            continue
    
    print(f"\n{COLOR_TITLE}{'='*70}{COLOR_RESET}")
    print(f"{COLOR_SUCCESS}✓ Incremental classification complete!{COLOR_RESET}")
    print(f"  {COLOR_VALUE}Newly classified:{COLOR_RESET} {classified_count} files")
    print(f"  {COLOR_VALUE}Total classified:{COLOR_RESET} {len(already_classified) + classified_count}/{total_files} files")
    
    return classified_count


def run_reversal_analysis(inference_logs_dir: str, predictions_log: str, 
                          output_dir: str) -> dict:
    """
    Run reversal analysis
    
    Args:
        inference_logs_dir: Directory containing inference logs
        predictions_log: Path to predictions log file
        output_dir: Output directory for reports
    
    Returns:
        Analysis report dictionary
    """
    print(f"\n{COLOR_TITLE}=== STEP 2: ANALYZING REVERSALS ==={COLOR_RESET}")
    
    analyzer = ReversalAnalyzer(inference_logs_dir)
    analyzer.analyze_directory(predictions_log)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'reversal_analysis_{timestamp}.json')
    
    analyzer.generate_report(report_path)
    
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_with_baseline(report: dict) -> None:
    """
    Compare with baseline model
    
    Args:
        report: Reversal analysis report
    """
    print(f"\n{COLOR_TITLE}=== STEP 3: BASELINE COMPARISON ==={COLOR_RESET}")
    
    coherence_analysis = report.get('coherence_analysis', {})
    
    print(f"\n{COLOR_INFO}MarketToM Performance:{COLOR_RESET}")
    print(f"{COLOR_SUCCESS}A组 (Coherent):{COLOR_RESET}")
    if coherence_analysis['coherent']['with_predictions'] > 0:
        print(f"  MCC: {COLOR_VALUE}{coherence_analysis['coherent']['mcc']:.4f}{COLOR_RESET}")
        print(f"  Accuracy: {COLOR_VALUE}{coherence_analysis['coherent']['accuracy']:.2%}{COLOR_RESET}")
    
    print(f"\n{COLOR_WARNING}B组 (Dissonant):{COLOR_RESET}")
    if coherence_analysis['dissonant']['with_predictions'] > 0:
        print(f"  MCC: {COLOR_VALUE}{coherence_analysis['dissonant']['mcc']:.4f}{COLOR_RESET}")
        print(f"  Accuracy: {COLOR_VALUE}{coherence_analysis['dissonant']['accuracy']:.2%}{COLOR_RESET}")
    
    if (coherence_analysis['coherent']['with_predictions'] > 0 and 
        coherence_analysis['dissonant']['with_predictions'] > 0):
        
        mcc_diff = coherence_analysis['coherent']['mcc'] - coherence_analysis['dissonant']['mcc']
        
        print(f"\n{COLOR_TITLE}Key Findings:{COLOR_RESET}")
        print(f"  MCC difference (A - B): {COLOR_VALUE}{mcc_diff:.4f}{COLOR_RESET}")
        
        if abs(mcc_diff) < 0.05:
            print(f"  {COLOR_SUCCESS}✓ MarketToM maintains robustness across both scenarios!{COLOR_RESET}")
            print(f"  {COLOR_INFO}This suggests the model can handle cognitive dissonance effectively.{COLOR_RESET}")
        else:
            print(f"  {COLOR_WARNING}! Performance gap detected between coherent and dissonant scenarios{COLOR_RESET}")


def analyze_reversal_effectiveness(report: dict) -> None:
    """
    Analyze reversal effectiveness (E.2-Prime validation)
    
    Args:
        report: Reversal analysis report
    """
    print(f"\n{COLOR_TITLE}=== STEP 4: REVERSAL EFFECTIVENESS ANALYSIS (E.2-Prime) ==={COLOR_RESET}")
    
    reversal_patterns = report.get('reversal_patterns', {})
    
    if not reversal_patterns:
        print(f"{COLOR_WARNING}No reversal patterns found for analysis{COLOR_RESET}")
        return
    
    print(f"\n{COLOR_INFO}Analyzing specific reversal patterns:{COLOR_RESET}")
    
    key_patterns = {k: v for k, v in reversal_patterns.items() 
                   if 'UP->DOWN' in k or 'DOWN->UP' in k}
    
    if key_patterns:
        sorted_patterns = sorted(key_patterns.items(), 
                               key=lambda x: x[1]['total'], 
                               reverse=True)[:3]
        
        print(f"\n{COLOR_TITLE}Top reversal patterns:{COLOR_RESET}")
        for pattern, stats in sorted_patterns:
            accuracy = stats['accuracy']
            color = COLOR_SUCCESS if accuracy > 0.6 else COLOR_WARNING
            
            print(f"\n{COLOR_VALUE}{pattern}{COLOR_RESET}")
            print(f"  Accuracy: {color}{accuracy:.2%}{COLOR_RESET} ({stats['correct']}/{stats['total']})")
            
            if accuracy > 0.65:
                print(f"  {COLOR_SUCCESS}✓ This reversal pattern is highly predictive!{COLOR_RESET}")
                print(f"  {COLOR_INFO}The model correctly uses cognitive dissonance to improve predictions.{COLOR_RESET}")
    
    dissonant_stats = report['coherence_analysis']['dissonant']
    if dissonant_stats['with_predictions'] > 0:
        print(f"\n{COLOR_TITLE}Overall reversal effectiveness:{COLOR_RESET}")
        print(f"  Dissonant scenarios accuracy: {COLOR_VALUE}{dissonant_stats['accuracy']:.2%}{COLOR_RESET}")
        print(f"  Total dissonant samples: {COLOR_VALUE}{dissonant_stats['count']}{COLOR_RESET}")
        
        if dissonant_stats['accuracy'] > 0.5:
            print(f"\n{COLOR_SUCCESS}✓ Reversals are NOT noise - they improve predictions!{COLOR_RESET}")
            print(f"{COLOR_INFO}E.2-Prime validated: Cognitive dissonance is a valuable signal.{COLOR_RESET}")


def main():
    """Main program"""
    parser = argparse.ArgumentParser(description='Run Reversal Experiment')
    parser.add_argument('--config', type=str, help='Path to config.json')
    parser.add_argument('--inference-logs', type=str, 
                       help='Path to inference logs directory')
    parser.add_argument('--predictions', type=str,
                       help='Path to predictions log file')
    parser.add_argument('--output', type=str, default='./reversal_reports',
                       help='Output directory for reports')
    parser.add_argument('--force-reclassify', action='store_true',
                       help='Force reclassify existing stances')
    parser.add_argument('--skip-classification', action='store_true',
                       help='Skip stance classification (use existing)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Show detailed output for each classification (default: True)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed classification output (sets verbose=False)')
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    config = load_config(args.config)
    
    if args.inference_logs:
        inference_logs_dir = args.inference_logs
    else:
        directories_config = config.get('directories', {})
        inference_logs_rel = directories_config.get('inference_logs', './storage/inference_logs')
        inference_logs_dir = os.path.join(project_root, inference_logs_rel)
    
    if args.predictions:
        predictions_log = args.predictions
    else:
        predictions_log = os.path.join(project_root, 'prediction_results.json')
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{COLOR_TITLE}{'='*60}{COLOR_RESET}")
    print(f"{COLOR_TITLE}  REVERSAL EXPERIMENT - E.2 & E.2-Prime{COLOR_RESET}")
    print(f"{COLOR_TITLE}{'='*60}{COLOR_RESET}")
    
    # Step 1: Stance classification
    if not args.skip_classification:
        llm_client, llm_model = initialize_llm_client(config)
        classifier = StanceClassifier(llm_client, llm_model, temperature=0.3)
        
        classified_count = classify_existing_logs(
            inference_logs_dir, 
            classifier, 
            args.force_reclassify,
            verbose
        )
        
        if classified_count == 0 and not args.force_reclassify:
            print(f"\n{COLOR_INFO}All logs already classified. Use --force-reclassify to reclassify.{COLOR_RESET}")
    else:
        print(f"\n{COLOR_INFO}Skipping classification step{COLOR_RESET}")
    
    # Step 2: Reversal analysis
    report = run_reversal_analysis(inference_logs_dir, predictions_log, output_dir)
    
    # Step 3: Baseline comparison
    compare_with_baseline(report)
    
    # Step 4: Reversal effectiveness analysis
    analyze_reversal_effectiveness(report)
    
    print(f"\n{COLOR_TITLE}{'='*60}{COLOR_RESET}")
    print(f"{COLOR_SUCCESS}  EXPERIMENT COMPLETED!{COLOR_RESET}")
    print(f"{COLOR_TITLE}{'='*60}{COLOR_RESET}\n")


if __name__ == '__main__':
    main()
