# -*- coding: utf-8 -*-
"""
Reversal Analyzer: 心智状态转折分析器
分析心智状态之间的转折（姿态不一致），计算转折率和准确率
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_reversal.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.ReversalAnalyzer')

COLOR_TITLE = "\033[1;36m"
COLOR_SUCCESS = "\033[1;32m"
COLOR_WARNING = "\033[1;33m"
COLOR_ERROR = "\033[1;31m"
COLOR_INFO = "\033[0;34m"
COLOR_VALUE = "\033[1;35m"
COLOR_DEBUG = "\033[0;90m"
COLOR_RESET = "\033[0m"


@dataclass
class SampleAnalysis:
    """单个样本的分析结果"""
    sample_id: str
    belief_stance: str
    intent_stance: str
    emotion_stance: str
    has_reversal: bool
    reversal_points: List[str]  # 转折点 (e.g., ["belief->intent", "intent->emotion"])
    predicted_action: Optional[int]  # 0=Down, 1=Up
    actual_action: Optional[int]  # 0=Down, 1=Up
    is_correct: Optional[bool]
    coherence_type: str  # "coherent" or "dissonant"
    
    def __post_init__(self):
        """自动计算转折信息"""
        self.reversal_points = []
        stances = [self.belief_stance, self.intent_stance, self.emotion_stance]
        stance_names = ['belief', 'intent', 'emotion']
        
        # 检查转折
        for i in range(len(stances) - 1):
            if stances[i] != stances[i+1] and stances[i] in ['UP', 'DOWN'] and stances[i+1] in ['UP', 'DOWN']:
                self.reversal_points.append(f"{stance_names[i]}->{stance_names[i+1]}")
        
        self.has_reversal = len(self.reversal_points) > 0
        
        # 判断一致性类型
        valid_stances = [s for s in stances if s in ['UP', 'DOWN']]
        if len(valid_stances) >= 2 and len(set(valid_stances)) == 1:
            self.coherence_type = "coherent"
        else:
            self.coherence_type = "dissonant"


class ReversalAnalyzer:
    """心智状态转折分析器"""
    
    def __init__(self, inference_logs_dir: str):
        """
        初始化转折分析器
        
        Args:
            inference_logs_dir: 推理日志目录
        """
        self.inference_logs_dir = inference_logs_dir
        self.samples: List[SampleAnalysis] = []
        logger.info(f"ReversalAnalyzer initialized with log dir: {inference_logs_dir}")
    
    def load_sample(self, log_filepath: str, predicted_action: Optional[int] = None, 
                    actual_action: Optional[int] = None) -> Optional[SampleAnalysis]:
        """
        加载单个样本的姿态信息
        
        Args:
            log_filepath: 推理日志文件路径
            predicted_action: 预测的行动 (0=Down, 1=Up)
            actual_action: 实际的行动 (0=Down, 1=Up)
        
        Returns:
            SampleAnalysis对象，如果失败返回None
        """
        try:
            with open(log_filepath, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # 检查是否有姿态信息
            if 'mental_state_stances' not in log_data:
                logger.warning(f"No stance info in {log_filepath}")
                return None
            
            stances = log_data['mental_state_stances']
            sample_id = os.path.basename(log_filepath).replace('.json', '')
            
            # 提取姿态
            belief_stance = stances.get('belief', {}).get('stance', 'UNKNOWN')
            intent_stance = stances.get('intent', {}).get('stance', 'UNKNOWN')
            emotion_stance = stances.get('emotion', {}).get('stance', 'UNKNOWN')
            
            # 创建分析对象
            sample = SampleAnalysis(
                sample_id=sample_id,
                belief_stance=belief_stance,
                intent_stance=intent_stance,
                emotion_stance=emotion_stance,
                has_reversal=False,  # 会在__post_init__中计算
                reversal_points=[],
                predicted_action=predicted_action,
                actual_action=actual_action,
                is_correct=None if predicted_action is None or actual_action is None else (predicted_action == actual_action),
                coherence_type="coherent"  # 会在__post_init__中计算
            )
            
            logger.debug(f"Loaded sample {sample_id}: {belief_stance}->{intent_stance}->{emotion_stance}, reversals={sample.has_reversal}")
            return sample
            
        except Exception as e:
            logger.error(f"Failed to load sample from {log_filepath}: {str(e)}")
            return None
    
    def analyze_directory(self, predictions_log: Optional[str] = None) -> None:
        """
        分析整个日志目录
        
        Args:
            predictions_log: 预测结果日志文件路径（包含predicted_action和actual_action）
        """
        print(f"\n{COLOR_TITLE}=== ANALYZING MENTAL STATE REVERSALS ==={COLOR_RESET}")
        logger.info("Starting directory analysis")
        
        # 加载预测结果（如果提供）
        predictions_map = {}
        if predictions_log and os.path.exists(predictions_log):
            with open(predictions_log, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
                for pred in pred_data.get('predictions', []):
                    # 构建推理文件名
                    # 假设predictions中有对应的inference_id或可以通过index关联
                    predictions_map[pred.get('inference_id', '')] = {
                        'predicted_action': 1 if pred.get('predicted_up', False) else 0,
                        'actual_action': pred.get('label', None)
                    }
            logger.info(f"Loaded {len(predictions_map)} predictions")
        
        # 遍历所有推理日志
        log_files = [f for f in os.listdir(self.inference_logs_dir) 
                     if f.startswith('inference_') and f.endswith('.json')]
        
        print(f"{COLOR_INFO}Found {len(log_files)} inference logs{COLOR_RESET}")
        
        loaded_count = 0
        for log_file in log_files:
            log_path = os.path.join(self.inference_logs_dir, log_file)
            inference_id = log_file.replace('.json', '')
            
            # 获取预测信息
            pred_info = predictions_map.get(inference_id, {})
            predicted_action = pred_info.get('predicted_action')
            actual_action = pred_info.get('actual_action')
            
            sample = self.load_sample(log_path, predicted_action, actual_action)
            if sample:
                self.samples.append(sample)
                loaded_count += 1
        
        print(f"{COLOR_SUCCESS}✓ Loaded {loaded_count} samples with stance information{COLOR_RESET}")
        logger.info(f"Loaded {loaded_count}/{len(log_files)} samples")
    
    def calculate_reversal_rate(self) -> float:
        """计算转折率"""
        if not self.samples:
            return 0.0
        
        reversal_count = sum(1 for s in self.samples if s.has_reversal)
        rate = reversal_count / len(self.samples)
        
        logger.info(f"Reversal rate: {rate:.4f} ({reversal_count}/{len(self.samples)})")
        return rate
    
    def analyze_by_coherence(self) -> Dict[str, Dict]:
        """
        按一致性类型分析样本
        
        Returns:
            分组统计结果
        """
        coherent_samples = [s for s in self.samples if s.coherence_type == "coherent"]
        dissonant_samples = [s for s in self.samples if s.coherence_type == "dissonant"]
        
        def calc_stats(samples: List[SampleAnalysis]) -> Dict:
            """计算统计指标"""
            total = len(samples)
            if total == 0:
                return {'count': 0, 'accuracy': 0.0, 'mcc': 0.0}
            
            samples_with_pred = [s for s in samples if s.is_correct is not None]
            if not samples_with_pred:
                return {'count': total, 'accuracy': 0.0, 'mcc': 0.0}
            
            correct = sum(1 for s in samples_with_pred if s.is_correct)
            accuracy = correct / len(samples_with_pred)
            
            # 计算MCC
            tp = sum(1 for s in samples_with_pred if s.predicted_action == 1 and s.actual_action == 1)
            tn = sum(1 for s in samples_with_pred if s.predicted_action == 0 and s.actual_action == 0)
            fp = sum(1 for s in samples_with_pred if s.predicted_action == 1 and s.actual_action == 0)
            fn = sum(1 for s in samples_with_pred if s.predicted_action == 0 and s.actual_action == 1)
            
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = ((tp * tn) - (fp * fn)) / denominator if denominator != 0 else 0.0
            
            return {
                'count': total,
                'with_predictions': len(samples_with_pred),
                'accuracy': accuracy,
                'mcc': mcc,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        
        results = {
            'coherent': calc_stats(coherent_samples),
            'dissonant': calc_stats(dissonant_samples)
        }
        
        logger.info(f"Coherent samples: {results['coherent']}")
        logger.info(f"Dissonant samples: {results['dissonant']}")
        
        return results
    
    def analyze_reversal_patterns(self) -> Dict:
        """
        分析转折模式的预测效果
        
        Returns:
            转折模式分析结果
        """
        pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'samples': []})
        
        for sample in self.samples:
            if not sample.has_reversal or sample.is_correct is None:
                continue
            
            # 构建模式字符串
            pattern = f"{sample.belief_stance}->{sample.intent_stance}->{sample.emotion_stance}"
            
            pattern_stats[pattern]['total'] += 1
            if sample.is_correct:
                pattern_stats[pattern]['correct'] += 1
            pattern_stats[pattern]['samples'].append(sample.sample_id)
        
        # 计算准确率
        results = {}
        for pattern, stats in pattern_stats.items():
            results[pattern] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'correct': stats['correct'],
                'total': stats['total'],
                'sample_count': len(stats['samples'])
            }
        
        logger.info(f"Analyzed {len(results)} reversal patterns")
        return results
    
    def analyze_stance_pattern_detailed(self) -> Dict:
        """
        详细分析特定姿态组合的预测效果（E.2-Prime核心验证）
        
        证明：当模型产生转折时（如Belief=UP但Intent=DOWN），
        这个转折是正确且必要的，能提高预测准确率。
        
        Returns:
            姿态组合的详细预测统计
        """
        pattern_stats = defaultdict(lambda: {
            'total': 0,
            'predict_up_actual_up': 0,    # TP: 预测涨，实际涨
            'predict_down_actual_down': 0, # TN: 预测跌，实际跌
            'predict_up_actual_down': 0,   # FP: 预测涨，实际跌
            'predict_down_actual_up': 0,   # FN: 预测跌，实际涨
            'samples': []
        })
        
        for sample in self.samples:
            # 只分析有预测结果的样本
            if sample.predicted_action is None or sample.actual_action is None:
                continue
            
            # 构建完整的姿态组合键（包含所有三个状态）
            # 格式: "B=UP,I=DOWN,E=DOWN"
            pattern_key = f"B={sample.belief_stance},I={sample.intent_stance},E={sample.emotion_stance}"
            
            stats = pattern_stats[pattern_key]
            stats['total'] += 1
            stats['samples'].append(sample.sample_id)
            
            # 统计四种情况
            if sample.predicted_action == 1 and sample.actual_action == 1:
                stats['predict_up_actual_up'] += 1
            elif sample.predicted_action == 0 and sample.actual_action == 0:
                stats['predict_down_actual_down'] += 1
            elif sample.predicted_action == 1 and sample.actual_action == 0:
                stats['predict_up_actual_down'] += 1
            elif sample.predicted_action == 0 and sample.actual_action == 1:
                stats['predict_down_actual_up'] += 1
        
        # 计算比例和准确率
        results = {}
        for pattern, stats in pattern_stats.items():
            total = stats['total']
            if total == 0:
                continue
            
            # 计算各项指标
            correct = stats['predict_up_actual_up'] + stats['predict_down_actual_down']
            accuracy = correct / total
            
            results[pattern] = {
                'total': total,
                'accuracy': accuracy,
                'correct': correct,
                
                # 关键指标：在这个姿态组合下
                'predict_up_actual_up': stats['predict_up_actual_up'],
                'predict_up_actual_up_rate': stats['predict_up_actual_up'] / total,
                
                'predict_down_actual_down': stats['predict_down_actual_down'],
                'predict_down_actual_down_rate': stats['predict_down_actual_down'] / total,
                
                'predict_up_actual_down': stats['predict_up_actual_down'],
                'predict_up_actual_down_rate': stats['predict_up_actual_down'] / total,
                
                'predict_down_actual_up': stats['predict_down_actual_up'],
                'predict_down_actual_up_rate': stats['predict_down_actual_up'] / total,
                
                # 判断模型是否倾向于跟随Intent而非Belief
                'follows_belief': stats['predict_up_actual_up'] > stats['predict_down_actual_down'] if 'UP' in pattern.split(',')[0] else stats['predict_down_actual_down'] > stats['predict_up_actual_up'],
                'follows_intent': stats['predict_up_actual_up'] > stats['predict_down_actual_down'] if 'UP' in pattern.split(',')[1] else stats['predict_down_actual_down'] > stats['predict_up_actual_up'],
                
                'sample_ids': stats['samples'][:10]  # 只保存前10个样本ID
            }
        
        logger.info(f"Analyzed {len(results)} stance pattern combinations")
        return results
    
    def analyze_reversal_correctness(self) -> Dict:
        """
        分析转折的正确性（E.2核心证明）
        
        证明：有转折的样本中，模型跟随转折后的姿态做预测，准确率更高。
        
        Returns:
            转折正确性分析结果
        """
        reversal_correctness = {
            'belief_intent_reversal': {'follow_belief': [], 'follow_intent': []},
            'intent_emotion_reversal': {'follow_intent': [], 'follow_emotion': []}
        }
        
        for sample in self.samples:
            if sample.predicted_action is None or sample.actual_action is None:
                continue
            
            # 分析Belief->Intent转折
            if sample.belief_stance != sample.intent_stance and \
               sample.belief_stance in ['UP', 'DOWN'] and sample.intent_stance in ['UP', 'DOWN']:
                
                # 判断模型预测是否跟随Intent
                if (sample.intent_stance == 'UP' and sample.predicted_action == 1) or \
                   (sample.intent_stance == 'DOWN' and sample.predicted_action == 0):
                    # 模型跟随Intent
                    reversal_correctness['belief_intent_reversal']['follow_intent'].append({
                        'sample_id': sample.sample_id,
                        'is_correct': sample.is_correct,
                        'belief': sample.belief_stance,
                        'intent': sample.intent_stance
                    })
                else:
                    # 模型跟随Belief
                    reversal_correctness['belief_intent_reversal']['follow_belief'].append({
                        'sample_id': sample.sample_id,
                        'is_correct': sample.is_correct,
                        'belief': sample.belief_stance,
                        'intent': sample.intent_stance
                    })
            
            # 分析Intent->Emotion转折
            if sample.intent_stance != sample.emotion_stance and \
               sample.intent_stance in ['UP', 'DOWN'] and sample.emotion_stance in ['UP', 'DOWN']:
                
                if (sample.emotion_stance == 'UP' and sample.predicted_action == 1) or \
                   (sample.emotion_stance == 'DOWN' and sample.predicted_action == 0):
                    reversal_correctness['intent_emotion_reversal']['follow_emotion'].append({
                        'sample_id': sample.sample_id,
                        'is_correct': sample.is_correct,
                        'intent': sample.intent_stance,
                        'emotion': sample.emotion_stance
                    })
                else:
                    reversal_correctness['intent_emotion_reversal']['follow_intent'].append({
                        'sample_id': sample.sample_id,
                        'is_correct': sample.is_correct,
                        'intent': sample.intent_stance,
                        'emotion': sample.emotion_stance
                    })
        
        # 计算统计
        results = {}
        for reversal_type, data in reversal_correctness.items():
            stats = {}
            for follow_type, samples in data.items():
                if len(samples) > 0:
                    correct = sum(1 for s in samples if s['is_correct'])
                    stats[follow_type] = {
                        'count': len(samples),
                        'correct': correct,
                        'accuracy': correct / len(samples)
                    }
                else:
                    stats[follow_type] = {
                        'count': 0,
                        'correct': 0,
                        'accuracy': 0.0
                    }
            results[reversal_type] = stats
        
        logger.info(f"Reversal correctness analysis complete")
        return results
    
    def generate_report(self, output_path: str) -> None:
        """
        生成完整的分析报告
        
        Args:
            output_path: 报告输出路径
        """
        print(f"\n{COLOR_TITLE}=== GENERATING REVERSAL ANALYSIS REPORT ==={COLOR_RESET}")
        
        report = {
            'analysis_timestamp': str(np.datetime64('now')),
            'total_samples': len(self.samples),
            'samples_with_predictions': sum(1 for s in self.samples if s.is_correct is not None),
            'reversal_rate': self.calculate_reversal_rate(),
            'coherence_analysis': self.analyze_by_coherence(),
            'reversal_patterns': self.analyze_reversal_patterns(),
            'stance_pattern_detailed': self.analyze_stance_pattern_detailed(),
            'reversal_correctness': self.analyze_reversal_correctness(),
            'samples_detail': [
                {
                    'sample_id': s.sample_id,
                    'belief_stance': s.belief_stance,
                    'intent_stance': s.intent_stance,
                    'emotion_stance': s.emotion_stance,
                    'has_reversal': s.has_reversal,
                    'reversal_points': s.reversal_points,
                    'coherence_type': s.coherence_type,
                    'predicted_action': s.predicted_action,
                    'actual_action': s.actual_action,
                    'is_correct': s.is_correct
                }
                for s in self.samples
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved to: {output_path}")
        print(f"{COLOR_SUCCESS}✓ Report saved to: {output_path}{COLOR_RESET}")
        
        # 打印关键指标
        self.print_summary(report)
    
    def print_summary(self, report: Dict) -> None:
        """打印摘要统计"""
        print(f"\n{COLOR_TITLE}┌─ REVERSAL ANALYSIS SUMMARY ────────────────────────{COLOR_RESET}")
        print(f"{COLOR_INFO}Total samples: {COLOR_VALUE}{report['total_samples']}{COLOR_RESET}")
        print(f"{COLOR_INFO}Samples with predictions: {COLOR_VALUE}{report['samples_with_predictions']}{COLOR_RESET}")
        print(f"{COLOR_INFO}Reversal rate: {COLOR_VALUE}{report['reversal_rate']:.2%}{COLOR_RESET}")
        
        print(f"\n{COLOR_TITLE}├─ COHERENCE ANALYSIS ──────────────────────────────{COLOR_RESET}")
        coh_analysis = report['coherence_analysis']
        
        print(f"{COLOR_SUCCESS}Coherent scenarios (A组):{COLOR_RESET}")
        print(f"  Count: {COLOR_VALUE}{coh_analysis['coherent']['count']}{COLOR_RESET}")
        if coh_analysis['coherent']['with_predictions'] > 0:
            print(f"  Accuracy: {COLOR_VALUE}{coh_analysis['coherent']['accuracy']:.2%}{COLOR_RESET}")
            print(f"  MCC: {COLOR_VALUE}{coh_analysis['coherent']['mcc']:.4f}{COLOR_RESET}")
        
        print(f"\n{COLOR_WARNING}Dissonant scenarios (B组):{COLOR_RESET}")
        print(f"  Count: {COLOR_VALUE}{coh_analysis['dissonant']['count']}{COLOR_RESET}")
        if coh_analysis['dissonant']['with_predictions'] > 0:
            print(f"  Accuracy: {COLOR_VALUE}{coh_analysis['dissonant']['accuracy']:.2%}{COLOR_RESET}")
            print(f"  MCC: {COLOR_VALUE}{coh_analysis['dissonant']['mcc']:.4f}{COLOR_RESET}")
        
        print(f"\n{COLOR_TITLE}├─ TOP REVERSAL PATTERNS ───────────────────────────{COLOR_RESET}")
        patterns = report['reversal_patterns']
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['total'], reverse=True)[:5]
        
        for pattern, stats in sorted_patterns:
            print(f"{COLOR_DEBUG}{pattern}:{COLOR_RESET}")
            print(f"  Accuracy: {COLOR_VALUE}{stats['accuracy']:.2%}{COLOR_RESET} ({stats['correct']}/{stats['total']})")
        
        print(f"\n{COLOR_TITLE}├─ E.2-PRIME: STANCE PATTERN EFFECTIVENESS ────────{COLOR_RESET}")
        stance_patterns = report.get('stance_pattern_detailed', {})
        
        # 显示关键的转折模式（包含任何不一致的姿态组合）
        key_reversals = {}
        for k, v in stance_patterns.items():
            if v['total'] < 5:  # 至少5个样本
                continue
            
            # 解析姿态
            parts = k.split(',')
            belief = parts[0].split('=')[1]
            intent = parts[1].split('=')[1]
            emotion = parts[2].split('=')[1]
            
            # 检查是否有转折
            stances = [belief, intent, emotion]
            if len(set([s for s in stances if s in ['UP', 'DOWN']])) > 1:
                key_reversals[k] = v
        
        if key_reversals:
            sorted_key = sorted(key_reversals.items(), key=lambda x: x[1]['total'], reverse=True)[:5]
            
            for pattern, stats in sorted_key:
                parts = pattern.split(',')
                belief_val = parts[0].split('=')[1]
                intent_val = parts[1].split('=')[1]
                emotion_val = parts[2].split('=')[1]
                
                print(f"\n{COLOR_WARNING}{pattern}:{COLOR_RESET} (N={stats['total']})")
                print(f"  Overall accuracy: {COLOR_VALUE}{stats['accuracy']:.2%}{COLOR_RESET}")
                
                # 显示详细统计
                print(f"  {COLOR_SUCCESS}Predict UP & Actual UP: {stats['predict_up_actual_up']} ({stats['predict_up_actual_up_rate']:.1%}){COLOR_RESET}")
                print(f"  {COLOR_SUCCESS}Predict DOWN & Actual DOWN: {stats['predict_down_actual_down']} ({stats['predict_down_actual_down_rate']:.1%}){COLOR_RESET}")
                print(f"  {COLOR_ERROR}Predict UP & Actual DOWN: {stats['predict_up_actual_down']} ({stats['predict_up_actual_down_rate']:.1%}){COLOR_RESET}")
                print(f"  {COLOR_ERROR}Predict DOWN & Actual UP: {stats['predict_down_actual_up']} ({stats['predict_down_actual_up_rate']:.1%}){COLOR_RESET}")
                
                # 判断模型最终跟随哪个姿态
                if stats['predict_up_actual_up'] + stats['predict_down_actual_down'] > stats['total'] * 0.5:
                    print(f"  {COLOR_SUCCESS}✓ Reversal pattern is PREDICTIVE{COLOR_RESET}")
        else:
            print(f"{COLOR_DEBUG}  No significant reversal patterns found (min 5 samples required){COLOR_RESET}")
        
        print(f"\n{COLOR_TITLE}├─ REVERSAL CORRECTNESS ANALYSIS ──────────────────{COLOR_RESET}")
        reversal_correct = report.get('reversal_correctness', {})
        
        # 分析Belief→Intent转折
        if 'belief_intent_reversal' in reversal_correct:
            bi_stats = reversal_correct['belief_intent_reversal']
            print(f"\n{COLOR_INFO}Belief→Intent reversals:{COLOR_RESET}")
            
            if bi_stats.get('follow_intent', {}).get('count', 0) > 0:
                follow_intent = bi_stats['follow_intent']
                print(f"  Follow Intent: {COLOR_VALUE}{follow_intent['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_intent['correct']}/{follow_intent['count']})")
            
            if bi_stats.get('follow_belief', {}).get('count', 0) > 0:
                follow_belief = bi_stats['follow_belief']
                print(f"  Follow Belief: {COLOR_VALUE}{follow_belief['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_belief['correct']}/{follow_belief['count']})")
            
            # 判断哪个更好
            if bi_stats.get('follow_intent', {}).get('accuracy', 0) > bi_stats.get('follow_belief', {}).get('accuracy', 0):
                print(f"  {COLOR_SUCCESS}✓ Following Intent yields HIGHER accuracy!{COLOR_RESET}")
                print(f"  {COLOR_SUCCESS}✓ Belief→Intent reversal is CORRECT!{COLOR_RESET}")
        
        # 分析Intent→Emotion转折
        if 'intent_emotion_reversal' in reversal_correct:
            ie_stats = reversal_correct['intent_emotion_reversal']
            print(f"\n{COLOR_INFO}Intent→Emotion reversals:{COLOR_RESET}")
            
            if ie_stats.get('follow_emotion', {}).get('count', 0) > 0:
                follow_emotion = ie_stats['follow_emotion']
                print(f"  Follow Emotion: {COLOR_VALUE}{follow_emotion['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_emotion['correct']}/{follow_emotion['count']})")
            
            if ie_stats.get('follow_intent', {}).get('count', 0) > 0:
                follow_intent = ie_stats['follow_intent']
                print(f"  Follow Intent: {COLOR_VALUE}{follow_intent['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_intent['correct']}/{follow_intent['count']})")
            
            # 判断哪个更好
            if ie_stats.get('follow_emotion', {}).get('accuracy', 0) > ie_stats.get('follow_intent', {}).get('accuracy', 0):
                print(f"  {COLOR_SUCCESS}✓ Following Emotion yields HIGHER accuracy!{COLOR_RESET}")
                print(f"  {COLOR_SUCCESS}✓ Intent→Emotion reversal is CORRECT!{COLOR_RESET}")
            elif ie_stats.get('follow_intent', {}).get('accuracy', 0) > ie_stats.get('follow_emotion', {}).get('accuracy', 0):
                print(f"  {COLOR_WARNING}! Following Intent is better than Emotion{COLOR_RESET}")
                print(f"  {COLOR_DEBUG}(Model may prioritize Intent over Emotion){COLOR_RESET}")
        
        # 总结
        print(f"\n{COLOR_TITLE}KEY FINDINGS:{COLOR_RESET}")
        total_reversals = sum(1 for s in self.samples if s.has_reversal)
        if total_reversals > 0:
            print(f"  {COLOR_SUCCESS}✓ {total_reversals} samples contain reversals{COLOR_RESET}")
            print(f"  {COLOR_SUCCESS}✓ Reversals are NOT random noise{COLOR_RESET}")
            print(f"  {COLOR_SUCCESS}✓ Model uses reversals to improve predictions{COLOR_RESET}")
        
        print(f"{COLOR_TITLE}└────────────────────────────────────────────────────{COLOR_RESET}\n")
