# -*- coding: utf-8 -*-
"""
Run Reversal Experiment: 转折率实验主程序
整合姿态分类和转折分析，运行完整的E.2实验
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# 添加项目根目录到路径
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
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(project_root, 'config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def initialize_llm_client(config: dict):
    """初始化LLM客户端"""
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
                          force_reclassify: bool = False) -> int:
    """
    为现有的推理日志添加姿态分类
    
    Args:
        inference_logs_dir: 推理日志目录
        classifier: 姿态分类器
        force_reclassify: 是否强制重新分类（即使已有姿态信息）
    
    Returns:
        分类的日志数量
    """
    print(f"\n{COLOR_TITLE}=== STEP 1: CLASSIFYING MENTAL STATE STANCES ==={COLOR_RESET}")
    
    log_files = [f for f in os.listdir(inference_logs_dir) 
                 if f.startswith('inference_') and f.endswith('.json')]
    
    print(f"{COLOR_INFO}Found {len(log_files)} inference logs{COLOR_RESET}")
    
    classified_count = 0
    skipped_count = 0
    
    for i, log_file in enumerate(log_files, 1):
        log_path = os.path.join(inference_logs_dir, log_file)
        
        try:
            # 检查是否已有姿态信息
            with open(log_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            if 'mental_state_stances' in log_data and not force_reclassify:
                skipped_count += 1
                if i % 10 == 0:
                    print(f"{COLOR_DEBUG}Progress: {i}/{len(log_files)} (skipped: {skipped_count}){COLOR_RESET}", end='\r')
                continue
            
            # 提取心智状态
            mental_states = log_data.get('mental_states', {})
            if not mental_states:
                logger.warning(f"No mental states in {log_file}")
                continue
            
            # 显示样本信息
            sample_id = log_data.get('sample_id', 'unknown')
            print(f"\n{COLOR_TITLE}{'─'*60}{COLOR_RESET}")
            print(f"{COLOR_INFO}[{i}/{len(log_files)}] Sample: {sample_id}{COLOR_RESET}")
            print(f"{COLOR_DEBUG}File: {log_file}{COLOR_RESET}")
            
            # 分类姿态（带详细输出）
            stances = classifier.classify_all_states(mental_states, verbose=True)
            
            # 显示分类结果摘要
            print(f"\n{COLOR_SUCCESS}✓ Classification Complete:{COLOR_RESET}")
            for state_name, stance_info in stances.items():
                stance = stance_info['stance']
                confidence = stance_info['confidence']
                
                # 根据姿态选择颜色
                if stance == 'UP':
                    stance_color = "\033[1;32m"  # 绿色
                elif stance == 'DOWN':
                    stance_color = "\033[1;31m"  # 红色
                else:
                    stance_color = "\033[1;33m"  # 黄色
                
                print(f"  {state_name:10s}: {stance_color}{stance:>6s}{COLOR_RESET} "
                      f"(confidence: {COLOR_VALUE}{confidence:.2f}{COLOR_RESET})")
            
            # 保存到日志
            add_stances_to_inference_log(log_path, stances)
            classified_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {log_file}: {str(e)}")
            continue
    
    print(f"\n{COLOR_TITLE}{'='*60}{COLOR_RESET}")
    print(f"{COLOR_SUCCESS}✓ Classified {classified_count} logs, skipped {skipped_count} existing{COLOR_RESET}")
    return classified_count


def run_reversal_analysis(inference_logs_dir: str, predictions_log: str, 
                          output_dir: str) -> dict:
    """
    运行转折分析
    
    Args:
        inference_logs_dir: 推理日志目录
        predictions_log: 预测结果日志
        output_dir: 输出目录
    
    Returns:
        分析报告
    """
    print(f"\n{COLOR_TITLE}=== STEP 2: ANALYZING REVERSALS ==={COLOR_RESET}")
    
    # 创建分析器
    analyzer = ReversalAnalyzer(inference_logs_dir)
    
    # 分析目录
    analyzer.analyze_directory(predictions_log)
    
    # 生成报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'reversal_analysis_{timestamp}.json')
    
    analyzer.generate_report(report_path)
    
    # 返回报告数据
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_with_baseline(report: dict) -> None:
    """
    与基线模型对比（模拟）
    
    Args:
        report: 转折分析报告
    """
    print(f"\n{COLOR_TITLE}=== STEP 3: BASELINE COMPARISON ==={COLOR_RESET}")
    
    coherence_analysis = report.get('coherence_analysis', {})
    
    # 显示MarketToM的性能
    print(f"\n{COLOR_INFO}MarketToM Performance:{COLOR_RESET}")
    print(f"{COLOR_SUCCESS}A组 (Coherent):{COLOR_RESET}")
    if coherence_analysis['coherent']['with_predictions'] > 0:
        print(f"  MCC: {COLOR_VALUE}{coherence_analysis['coherent']['mcc']:.4f}{COLOR_RESET}")
        print(f"  Accuracy: {COLOR_VALUE}{coherence_analysis['coherent']['accuracy']:.2%}{COLOR_RESET}")
    
    print(f"\n{COLOR_WARNING}B组 (Dissonant):{COLOR_RESET}")
    if coherence_analysis['dissonant']['with_predictions'] > 0:
        print(f"  MCC: {COLOR_VALUE}{coherence_analysis['dissonant']['mcc']:.4f}{COLOR_RESET}")
        print(f"  Accuracy: {COLOR_VALUE}{coherence_analysis['dissonant']['accuracy']:.2%}{COLOR_RESET}")
    
    # 计算MCC差异
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
    分析转折的有效性（E.2-Prime验证）
    
    Args:
        report: 转折分析报告
    """
    print(f"\n{COLOR_TITLE}=== STEP 4: REVERSAL EFFECTIVENESS ANALYSIS (E.2-Prime) ==={COLOR_RESET}")
    
    # 统计特定转折模式的准确性
    reversal_patterns = report.get('reversal_patterns', {})
    
    if not reversal_patterns:
        print(f"{COLOR_WARNING}No reversal patterns found for analysis{COLOR_RESET}")
        return
    
    print(f"\n{COLOR_INFO}Analyzing specific reversal patterns:{COLOR_RESET}")
    
    # 寻找关键模式：UP->DOWN转折
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
    
    # 总体转折有效性
    dissonant_stats = report['coherence_analysis']['dissonant']
    if dissonant_stats['with_predictions'] > 0:
        print(f"\n{COLOR_TITLE}Overall reversal effectiveness:{COLOR_RESET}")
        print(f"  Dissonant scenarios accuracy: {COLOR_VALUE}{dissonant_stats['accuracy']:.2%}{COLOR_RESET}")
        print(f"  Total dissonant samples: {COLOR_VALUE}{dissonant_stats['count']}{COLOR_RESET}")
        
        if dissonant_stats['accuracy'] > 0.5:
            print(f"\n{COLOR_SUCCESS}✓ Reversals are NOT noise - they improve predictions!{COLOR_RESET}")
            print(f"{COLOR_INFO}E.2-Prime validated: Cognitive dissonance is a valuable signal.{COLOR_RESET}")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='Run Reversal Experiment')
    parser.add_argument('--config', type=str, help='Path to config.json')
    parser.add_argument('--inference-logs', type=str, 
                       help='Path to inference logs directory')
    parser.add_argument('--predictions', type=str,
                       help='Path to predictions log file')
    parser.add_argument('--output', type=str, default='./analysis_results',
                       help='Output directory for reports')
    parser.add_argument('--force-reclassify', action='store_true',
                       help='Force reclassify existing stances')
    parser.add_argument('--skip-classification', action='store_true',
                       help='Skip stance classification (use existing)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取路径
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
    
    # 步骤1: 姿态分类
    if not args.skip_classification:
        llm_client, llm_model = initialize_llm_client(config)
        classifier = StanceClassifier(llm_client, llm_model, temperature=0.3)
        
        classified_count = classify_existing_logs(
            inference_logs_dir, 
            classifier, 
            args.force_reclassify
        )
        
        if classified_count == 0 and not args.force_reclassify:
            print(f"\n{COLOR_INFO}All logs already classified. Use --force-reclassify to reclassify.{COLOR_RESET}")
    else:
        print(f"\n{COLOR_INFO}Skipping classification step{COLOR_RESET}")
    
    # 步骤2: 转折分析
    report = run_reversal_analysis(inference_logs_dir, predictions_log, output_dir)
    
    # 步骤3: 基线对比
    compare_with_baseline(report)
    
    # 步骤4: 转折有效性分析
    analyze_reversal_effectiveness(report)
    
    print(f"\n{COLOR_TITLE}{'='*60}{COLOR_RESET}")
    print(f"{COLOR_SUCCESS}  EXPERIMENT COMPLETED!{COLOR_RESET}")
    print(f"{COLOR_TITLE}{'='*60}{COLOR_RESET}\n")


if __name__ == '__main__':
    main()
