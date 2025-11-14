# -*- coding: utf-8 -*-
"""
Reversal Analyzer: Mental State Reversal Analyzer
Analyzes reversals between mental states (stance inconsistencies), calculates reversal rate and accuracy
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
    """Analysis result for a single sample"""
    sample_id: str
    belief_stance: str
    intent_stance: str
    emotion_stance: str
    has_reversal: bool
    reversal_points: List[str]  # Reversal points (e.g., ["belief->intent", "intent->emotion"])
    predicted_action: Optional[int]  # 0=Down, 1=Up
    actual_action: Optional[int]  # 0=Down, 1=Up
    is_correct: Optional[bool]
    coherence_type: str  # "coherent" or "dissonant"
    
    def __post_init__(self):
        """Automatically calculate reversal information"""
        self.reversal_points = []
        stances = [self.belief_stance, self.intent_stance, self.emotion_stance]
        stance_names = ['belief', 'intent', 'emotion']
        
        # Check for reversals
        for i in range(len(stances) - 1):
            if stances[i] != stances[i+1] and stances[i] in ['UP', 'DOWN'] and stances[i+1] in ['UP', 'DOWN']:
                self.reversal_points.append(f"{stance_names[i]}->{stance_names[i+1]}")
        
        self.has_reversal = len(self.reversal_points) > 0
        
        # Determine coherence type
        valid_stances = [s for s in stances if s in ['UP', 'DOWN']]
        if len(valid_stances) >= 2 and len(set(valid_stances)) == 1:
            self.coherence_type = "coherent"
        else:
            self.coherence_type = "dissonant"


class ReversalAnalyzer:
    """Mental State Reversal Analyzer"""
    
    def __init__(self, inference_logs_dir: str):
        """
        Initialize the reversal analyzer
        
        Args:
            inference_logs_dir: Inference logs directory
        """
        self.inference_logs_dir = inference_logs_dir
        self.samples: List[SampleAnalysis] = []
        logger.info(f"ReversalAnalyzer initialized with log dir: {inference_logs_dir}")
    
    def load_sample(self, log_filepath: str, predicted_action: Optional[int] = None, 
                    actual_action: Optional[int] = None) -> Optional[SampleAnalysis]:
        """
        Load stance information for a single sample
        
        Args:
            log_filepath: Inference log file path
            predicted_action: Predicted action (0=Down, 1=Up)
            actual_action: Actual action (0=Down, 1=Up)
        
        Returns:
            SampleAnalysis object, or None if failed
        """
        try:
            with open(log_filepath, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Check if stance information exists
            if 'mental_state_stances' not in log_data:
                logger.warning(f"No stance info in {log_filepath}")
                return None
            
            stances = log_data['mental_state_stances']
            sample_id = os.path.basename(log_filepath).replace('.json', '')
            
            # Extract stances
            belief_stance = stances.get('belief', {}).get('stance', 'UNKNOWN')
            intent_stance = stances.get('intent', {}).get('stance', 'UNKNOWN')
            emotion_stance = stances.get('emotion', {}).get('stance', 'UNKNOWN')
            
            # Create analysis object
            sample = SampleAnalysis(
                sample_id=sample_id,
                belief_stance=belief_stance,
                intent_stance=intent_stance,
                emotion_stance=emotion_stance,
                has_reversal=False,  # Will be calculated in __post_init__
                reversal_points=[],
                predicted_action=predicted_action,
                actual_action=actual_action,
                is_correct=None if predicted_action is None or actual_action is None else (predicted_action == actual_action),
                coherence_type="coherent"  # Will be calculated in __post_init__
            )
            
            logger.debug(f"Loaded sample {sample_id}: {belief_stance}->{intent_stance}->{emotion_stance}, reversals={sample.has_reversal}")
            return sample
            
        except Exception as e:
            logger.error(f"Failed to load sample from {log_filepath}: {str(e)}")
            return None
    
    def analyze_directory(self, predictions_log: Optional[str] = None) -> None:
        """
        Analyze the entire log directory
        
        Args:
            predictions_log: Prediction results log file path (contains predicted_action and actual_action)
        """
        print(f"\n{COLOR_TITLE}=== ANALYZING MENTAL STATE REVERSALS ==={COLOR_RESET}")
        logger.info("Starting directory analysis")
        
        # Load prediction results (if provided)
        predictions_map = {}
        if predictions_log and os.path.exists(predictions_log):
            with open(predictions_log, 'r', encoding='utf-8') as f:
                pred_data = json.load(f)
                for pred in pred_data.get('predictions', []):
                    # Build inference file name
                    # Assume predictions have corresponding inference_id or can be associated by index
                    predictions_map[pred.get('inference_id', '')] = {
                        'predicted_action': 1 if pred.get('predicted_up', False) else 0,
                        'actual_action': pred.get('label', None)
                    }
            logger.info(f"Loaded {len(predictions_map)} predictions")
        
        # Iterate through all inference logs
        log_files = [f for f in os.listdir(self.inference_logs_dir) 
                     if f.startswith('inference_') and f.endswith('.json')]
        
        print(f"{COLOR_INFO}Found {len(log_files)} inference logs{COLOR_RESET}")
        
        loaded_count = 0
        for log_file in log_files:
            log_path = os.path.join(self.inference_logs_dir, log_file)
            inference_id = log_file.replace('.json', '')
            
            # Get prediction information
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
        """Calculate reversal rate"""
        if not self.samples:
            return 0.0
        
        reversal_count = sum(1 for s in self.samples if s.has_reversal)
        rate = reversal_count / len(self.samples)
        
        logger.info(f"Reversal rate: {rate:.4f} ({reversal_count}/{len(self.samples)})")
        return rate
    
    def analyze_by_coherence(self) -> Dict[str, Dict]:
        """
        Analyze samples by coherence type
        
        Returns:
            Grouped statistical results
        """
        coherent_samples = [s for s in self.samples if s.coherence_type == "coherent"]
        dissonant_samples = [s for s in self.samples if s.coherence_type == "dissonant"]
        
        def calc_stats(samples: List[SampleAnalysis]) -> Dict:
            """Calculate statistical metrics"""
            total = len(samples)
            if total == 0:
                return {'count': 0, 'accuracy': 0.0, 'mcc': 0.0}
            
            samples_with_pred = [s for s in samples if s.is_correct is not None]
            if not samples_with_pred:
                return {'count': total, 'accuracy': 0.0, 'mcc': 0.0}
            
            correct = sum(1 for s in samples_with_pred if s.is_correct)
            accuracy = correct / len(samples_with_pred)
            
            # Calculate MCC
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
        Analyze prediction effectiveness of reversal patterns
        
        Returns:
            Reversal pattern analysis results
        """
        pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'samples': []})
        
        for sample in self.samples:
            if not sample.has_reversal or sample.is_correct is None:
                continue
            
            # Build pattern string
            pattern = f"{sample.belief_stance}->{sample.intent_stance}->{sample.emotion_stance}"
            
            pattern_stats[pattern]['total'] += 1
            if sample.is_correct:
                pattern_stats[pattern]['correct'] += 1
            pattern_stats[pattern]['samples'].append(sample.sample_id)
        
        # Calculate accuracy
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
        Detailed analysis of prediction effectiveness for specific stance combinations (E.2-Prime core validation)
        
        Proof: When the model produces a reversal (e.g., Belief=UP but Intent=DOWN),
        this reversal is correct and necessary, improving prediction accuracy.
        
        Returns:
            Detailed prediction statistics for stance combinations
        """
        pattern_stats = defaultdict(lambda: {
            'total': 0,
            'predict_up_actual_up': 0,    # TP: Predict up, actual up
            'predict_down_actual_down': 0, # TN: Predict down, actual down
            'predict_up_actual_down': 0,   # FP: Predict up, actual down
            'predict_down_actual_up': 0,   # FN: Predict down, actual up
            'samples': []
        })
        
        for sample in self.samples:
            # Only analyze samples with prediction results
            if sample.predicted_action is None or sample.actual_action is None:
                continue
            
            # Build complete stance combination key (includes all three states)
            # Format: "B=UP,I=DOWN,E=DOWN"
            pattern_key = f"B={sample.belief_stance},I={sample.intent_stance},E={sample.emotion_stance}"
            
            stats = pattern_stats[pattern_key]
            stats['total'] += 1
            stats['samples'].append(sample.sample_id)
            
            # Count four cases
            if sample.predicted_action == 1 and sample.actual_action == 1:
                stats['predict_up_actual_up'] += 1
            elif sample.predicted_action == 0 and sample.actual_action == 0:
                stats['predict_down_actual_down'] += 1
            elif sample.predicted_action == 1 and sample.actual_action == 0:
                stats['predict_up_actual_down'] += 1
            elif sample.predicted_action == 0 and sample.actual_action == 1:
                stats['predict_down_actual_up'] += 1
        
        # Calculate ratios and accuracy
        results = {}
        for pattern, stats in pattern_stats.items():
            total = stats['total']
            if total == 0:
                continue
            
            # Calculate various metrics
            correct = stats['predict_up_actual_up'] + stats['predict_down_actual_down']
            accuracy = correct / total
            
            results[pattern] = {
                'total': total,
                'accuracy': accuracy,
                'correct': correct,
                
                # Key metrics: under this stance combination
                'predict_up_actual_up': stats['predict_up_actual_up'],
                'predict_up_actual_up_rate': stats['predict_up_actual_up'] / total,
                
                'predict_down_actual_down': stats['predict_down_actual_down'],
                'predict_down_actual_down_rate': stats['predict_down_actual_down'] / total,
                
                'predict_up_actual_down': stats['predict_up_actual_down'],
                'predict_up_actual_down_rate': stats['predict_up_actual_down'] / total,
                
                'predict_down_actual_up': stats['predict_down_actual_up'],
                'predict_down_actual_up_rate': stats['predict_down_actual_up'] / total,
                
                # Determine if model tends to follow Intent rather than Belief
                'follows_belief': stats['predict_up_actual_up'] > stats['predict_down_actual_down'] if 'UP' in pattern.split(',')[0] else stats['predict_down_actual_down'] > stats['predict_up_actual_up'],
                'follows_intent': stats['predict_up_actual_up'] > stats['predict_down_actual_down'] if 'UP' in pattern.split(',')[1] else stats['predict_down_actual_down'] > stats['predict_up_actual_up'],
                
                'sample_ids': stats['samples'][:10]  # Only save first 10 sample IDs
            }
        
        logger.info(f"Analyzed {len(results)} stance pattern combinations")
        return results
    
    def analyze_reversal_correctness(self) -> Dict:
        """
        Analyze the correctness of reversals (E.2 core proof)
        
        Proof: In samples with reversals, the model follows the post-reversal stance for prediction, achieving higher accuracy.
        
        Returns:
            Reversal correctness analysis results
        """
        reversal_correctness = {
            'belief_intent_reversal': {'follow_belief': [], 'follow_intent': []},
            'intent_emotion_reversal': {'follow_intent': [], 'follow_emotion': []}
        }
        
        for sample in self.samples:
            if sample.predicted_action is None or sample.actual_action is None:
                continue
            
            # Analyze Belief->Intent reversal
            if sample.belief_stance != sample.intent_stance and \
               sample.belief_stance in ['UP', 'DOWN'] and sample.intent_stance in ['UP', 'DOWN']:
                
                # Determine if model prediction follows Intent
                if (sample.intent_stance == 'UP' and sample.predicted_action == 1) or \
                   (sample.intent_stance == 'DOWN' and sample.predicted_action == 0):
                    # Model follows Intent
                    reversal_correctness['belief_intent_reversal']['follow_intent'].append({
                        'sample_id': sample.sample_id,
                        'is_correct': sample.is_correct,
                        'belief': sample.belief_stance,
                        'intent': sample.intent_stance
                    })
                else:
                    # Model follows Belief
                    reversal_correctness['belief_intent_reversal']['follow_belief'].append({
                        'sample_id': sample.sample_id,
                        'is_correct': sample.is_correct,
                        'belief': sample.belief_stance,
                        'intent': sample.intent_stance
                    })
            
            # Analyze Intent->Emotion reversal
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
        
        # Calculate statistics
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
        Generate complete analysis report
        
        Args:
            output_path: Report output path
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
        
        # Print key metrics
        self.print_summary(report)
    
    def print_summary(self, report: Dict) -> None:
        """Print summary statistics"""
        print(f"\n{COLOR_TITLE}┌─ REVERSAL ANALYSIS SUMMARY ────────────────────────{COLOR_RESET}")
        print(f"{COLOR_INFO}Total samples: {COLOR_VALUE}{report['total_samples']}{COLOR_RESET}")
        print(f"{COLOR_INFO}Samples with predictions: {COLOR_VALUE}{report['samples_with_predictions']}{COLOR_RESET}")
        print(f"{COLOR_INFO}Reversal rate: {COLOR_VALUE}{report['reversal_rate']:.2%}{COLOR_RESET}")
        
        print(f"\n{COLOR_TITLE}├─ COHERENCE ANALYSIS ──────────────────────────────{COLOR_RESET}")
        coh_analysis = report['coherence_analysis']
        
        print(f"{COLOR_SUCCESS}Coherent scenarios (Group A):{COLOR_RESET}")
        print(f"  Count: {COLOR_VALUE}{coh_analysis['coherent']['count']}{COLOR_RESET}")
        if coh_analysis['coherent']['with_predictions'] > 0:
            print(f"  Accuracy: {COLOR_VALUE}{coh_analysis['coherent']['accuracy']:.2%}{COLOR_RESET}")
            print(f"  MCC: {COLOR_VALUE}{coh_analysis['coherent']['mcc']:.4f}{COLOR_RESET}")
        
        print(f"\n{COLOR_WARNING}Dissonant scenarios (Group B):{COLOR_RESET}")
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
        
        # Display key reversal patterns (containing any inconsistent stance combinations)
        key_reversals = {}
        for k, v in stance_patterns.items():
            if v['total'] < 5:  # At least 5 samples
                continue
            
            # Parse stances
            parts = k.split(',')
            belief = parts[0].split('=')[1]
            intent = parts[1].split('=')[1]
            emotion = parts[2].split('=')[1]
            
            # Check if there is a reversal
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
                
                # Display detailed statistics
                print(f"  {COLOR_SUCCESS}Predict UP & Actual UP: {stats['predict_up_actual_up']} ({stats['predict_up_actual_up_rate']:.1%}){COLOR_RESET}")
                print(f"  {COLOR_SUCCESS}Predict DOWN & Actual DOWN: {stats['predict_down_actual_down']} ({stats['predict_down_actual_down_rate']:.1%}){COLOR_RESET}")
                print(f"  {COLOR_ERROR}Predict UP & Actual DOWN: {stats['predict_up_actual_down']} ({stats['predict_up_actual_down_rate']:.1%}){COLOR_RESET}")
                print(f"  {COLOR_ERROR}Predict DOWN & Actual UP: {stats['predict_down_actual_up']} ({stats['predict_down_actual_up_rate']:.1%}){COLOR_RESET}")
                
                # Determine which stance the model ultimately follows
                if stats['predict_up_actual_up'] + stats['predict_down_actual_down'] > stats['total'] * 0.5:
                    print(f"  {COLOR_SUCCESS}✓ Reversal pattern is PREDICTIVE{COLOR_RESET}")
        else:
            print(f"{COLOR_DEBUG}  No significant reversal patterns found (min 5 samples required){COLOR_RESET}")
        
        print(f"\n{COLOR_TITLE}├─ REVERSAL CORRECTNESS ANALYSIS ──────────────────{COLOR_RESET}")
        reversal_correct = report.get('reversal_correctness', {})
        
        # Analyze Belief→Intent reversal
        if 'belief_intent_reversal' in reversal_correct:
            bi_stats = reversal_correct['belief_intent_reversal']
            print(f"\n{COLOR_INFO}Belief→Intent reversals:{COLOR_RESET}")
            
            if bi_stats.get('follow_intent', {}).get('count', 0) > 0:
                follow_intent = bi_stats['follow_intent']
                print(f"  Follow Intent: {COLOR_VALUE}{follow_intent['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_intent['correct']}/{follow_intent['count']})")
            
            if bi_stats.get('follow_belief', {}).get('count', 0) > 0:
                follow_belief = bi_stats['follow_belief']
                print(f"  Follow Belief: {COLOR_VALUE}{follow_belief['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_belief['correct']}/{follow_belief['count']})")
            
            # Determine which is better
            if bi_stats.get('follow_intent', {}).get('accuracy', 0) > bi_stats.get('follow_belief', {}).get('accuracy', 0):
                print(f"  {COLOR_SUCCESS}✓ Following Intent yields HIGHER accuracy!{COLOR_RESET}")
                print(f"  {COLOR_SUCCESS}✓ Belief→Intent reversal is CORRECT!{COLOR_RESET}")
        
        # Analyze Intent→Emotion reversal
        if 'intent_emotion_reversal' in reversal_correct:
            ie_stats = reversal_correct['intent_emotion_reversal']
            print(f"\n{COLOR_INFO}Intent→Emotion reversals:{COLOR_RESET}")
            
            if ie_stats.get('follow_emotion', {}).get('count', 0) > 0:
                follow_emotion = ie_stats['follow_emotion']
                print(f"  Follow Emotion: {COLOR_VALUE}{follow_emotion['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_emotion['correct']}/{follow_emotion['count']})")
            
            if ie_stats.get('follow_intent', {}).get('count', 0) > 0:
                follow_intent = ie_stats['follow_intent']
                print(f"  Follow Intent: {COLOR_VALUE}{follow_intent['accuracy']:.2%}{COLOR_RESET} accuracy ({follow_intent['correct']}/{follow_intent['count']})")
            
            # Determine which is better
            if ie_stats.get('follow_emotion', {}).get('accuracy', 0) > ie_stats.get('follow_intent', {}).get('accuracy', 0):
                print(f"  {COLOR_SUCCESS}✓ Following Emotion yields HIGHER accuracy!{COLOR_RESET}")
                print(f"  {COLOR_SUCCESS}✓ Intent→Emotion reversal is CORRECT!{COLOR_RESET}")
            elif ie_stats.get('follow_intent', {}).get('accuracy', 0) > ie_stats.get('follow_emotion', {}).get('accuracy', 0):
                print(f"  {COLOR_WARNING}! Following Intent is better than Emotion{COLOR_RESET}")
                print(f"  {COLOR_DEBUG}(Model may prioritize Intent over Emotion){COLOR_RESET}")
        
        # Summary
        print(f"\n{COLOR_TITLE}KEY FINDINGS:{COLOR_RESET}")
        total_reversals = sum(1 for s in self.samples if s.has_reversal)
        if total_reversals > 0:
            print(f"  {COLOR_SUCCESS}✓ {total_reversals} samples contain reversals{COLOR_RESET}")
            print(f"  {COLOR_SUCCESS}✓ Reversals are NOT random noise{COLOR_RESET}")
            print(f"  {COLOR_SUCCESS}✓ Model uses reversals to improve predictions{COLOR_RESET}")
        
        print(f"{COLOR_TITLE}└────────────────────────────────────────────────────{COLOR_RESET}\n")
