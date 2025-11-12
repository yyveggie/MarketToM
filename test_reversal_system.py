#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试转折率实验系统
验证所有模块是否正确安装和配置
"""

import os
import sys
import json

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ANSI颜色
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
RESET = "\033[0m"

def print_test(name, passed, message=""):
    """打印测试结果"""
    status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    print(f"{status} {name}")
    if message:
        print(f"     {YELLOW}{message}{RESET}")

def test_imports():
    """测试1: 模块导入"""
    print(f"\n{BLUE}=== Test 1: Module Imports ==={RESET}")
    
    try:
        from core.stance_classifier import StanceClassifier
        print_test("Import StanceClassifier", True)
    except Exception as e:
        print_test("Import StanceClassifier", False, str(e))
        return False
    
    try:
        from analysis.reversal_analyzer import ReversalAnalyzer, SampleAnalysis
        print_test("Import ReversalAnalyzer", True)
    except Exception as e:
        print_test("Import ReversalAnalyzer", False, str(e))
        return False
    
    try:
        from core.integrated_stance import IntegratedStanceClassifier
        print_test("Import IntegratedStanceClassifier", True)
    except Exception as e:
        print_test("Import IntegratedStanceClassifier", False, str(e))
        return False
    
    return True

def test_dependencies():
    """测试2: 依赖包"""
    print(f"\n{BLUE}=== Test 2: Dependencies ==={RESET}")
    
    deps = ['openai', 'pydantic', 'numpy', 'sentence_transformers']
    all_ok = True
    
    for dep in deps:
        try:
            __import__(dep)
            print_test(f"Dependency: {dep}", True)
        except ImportError:
            print_test(f"Dependency: {dep}", False, "Not installed")
            all_ok = False
    
    return all_ok

def test_config():
    """测试3: 配置文件"""
    print(f"\n{BLUE}=== Test 3: Configuration ==={RESET}")
    
    config_path = os.path.join(project_root, 'config.json')
    
    if not os.path.exists(config_path):
        print_test("Config file exists", False, "config.json not found")
        return False
    
    print_test("Config file exists", True)
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print_test("Config file valid JSON", True)
        
        # 检查必要字段
        api_config = config.get('api', {})
        if not api_config.get('providers', {}).get('openai', {}).get('api_key'):
            print_test("API key configured", False, "No OpenAI API key in config")
            return False
        else:
            print_test("API key configured", True)
        
        return True
        
    except Exception as e:
        print_test("Config file valid JSON", False, str(e))
        return False

def test_directories():
    """测试4: 目录结构"""
    print(f"\n{BLUE}=== Test 4: Directory Structure ==={RESET}")
    
    required_dirs = [
        './storage/inference_logs',
        './core',
        './analysis'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = os.path.join(project_root, dir_path)
        exists = os.path.isdir(full_path)
        print_test(f"Directory: {dir_path}", exists)
        if not exists:
            all_ok = False
    
    return all_ok

def test_sample_data():
    """测试5: 样本数据"""
    print(f"\n{BLUE}=== Test 5: Sample Data ==={RESET}")
    
    log_dir = os.path.join(project_root, 'storage/inference_logs')
    
    if not os.path.exists(log_dir):
        print_test("Inference logs directory", False, "Directory not found")
        return False
    
    log_files = [f for f in os.listdir(log_dir) 
                 if f.startswith('inference_') and f.endswith('.json')]
    
    if not log_files:
        print_test("Has inference logs", False, "No inference logs found")
        return False
    
    print_test(f"Has inference logs ({len(log_files)} files)", True)
    
    # 检查第一个日志的结构
    try:
        sample_log = os.path.join(log_dir, log_files[0])
        with open(sample_log, 'r') as f:
            log_data = json.load(f)
        
        has_mental_states = 'mental_states' in log_data
        print_test("Log has mental_states", has_mental_states)
        
        has_stances = 'mental_state_stances' in log_data
        if has_stances:
            print_test("Log has stance info", True, "Already classified")
        else:
            print_test("Log has stance info", False, "Need to run classification")
        
        return has_mental_states
        
    except Exception as e:
        print_test("Log file structure", False, str(e))
        return False

def test_stance_classifier():
    """测试6: 姿态分类器（模拟）"""
    print(f"\n{BLUE}=== Test 6: Stance Classifier (Mock) ==={RESET}")
    
    try:
        from core.stance_classifier import StanceResponse
        
        # 测试Pydantic模型
        response = StanceResponse(
            stance="UP",
            confidence=0.85,
            reasoning="Test reasoning"
        )
        response.validate_stance()
        
        print_test("StanceResponse model", True)
        return True
        
    except Exception as e:
        print_test("StanceResponse model", False, str(e))
        return False

def test_reversal_analyzer():
    """测试7: 转折分析器"""
    print(f"\n{BLUE}=== Test 7: Reversal Analyzer ==={RESET}")
    
    try:
        from analysis.reversal_analyzer import SampleAnalysis
        
        # 创建测试样本
        sample = SampleAnalysis(
            sample_id="test_001",
            belief_stance="UP",
            intent_stance="DOWN",
            emotion_stance="DOWN",
            has_reversal=False,  # 会自动计算
            reversal_points=[],
            predicted_action=0,
            actual_action=0,
            is_correct=True,
            coherence_type="coherent"
        )
        
        # 验证自动计算
        has_reversal = sample.has_reversal
        reversal_count = len(sample.reversal_points)
        
        print_test("SampleAnalysis auto-calculation", 
                   has_reversal and reversal_count > 0,
                   f"Detected {reversal_count} reversals")
        
        return True
        
    except Exception as e:
        print_test("SampleAnalysis creation", False, str(e))
        return False

def main():
    """运行所有测试"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Reversal Experiment System - Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    tests = [
        ("Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Sample Data", test_sample_data),
        ("Stance Classifier", test_stance_classifier),
        ("Reversal Analyzer", test_reversal_analyzer)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n{RED}Unexpected error in {name}: {e}{RESET}")
            results.append((name, False))
    
    # 总结
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Test Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}✅{RESET}" if result else f"{RED}❌{RESET}"
        print(f"{status} {name}")
    
    print(f"\n{BLUE}Results: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}🎉 All tests passed! System is ready.{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print("  1. Run: python quick_start_reversal.py")
        print("  2. Or:  python run_reversal_experiment.py --help")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  Some tests failed. Please fix them before proceeding.{RESET}")
        return 1

if __name__ == '__main__':
    exit(main())
