#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import MentalStateVisualizer

def main():
    print("🧠 MarketToM Latest Inference Flow Generator")
    print("=" * 45)
    
    # Initialize visualizer
    visualizer = MentalStateVisualizer()
    
    print("🎨 Generating latest complete inference flow chart...")
    print("   Including: Forward Inference + Strategy Retrieval + Prediction Results + Backward Correction")
    print()
    
    try:
        # Generate latest complete inference flow chart
        graph_path = visualizer.create_latest_complete_inference_graph()
        
        if graph_path:
            print(f"✅ Generation Successful!")
            print(f"📄 Graph File: {graph_path}")
            print()
            print("🎯 Graph Content Description:")
            print("   🔵 Light Blue = Environmental State (with smart line wrapping)")
            print("   🟢 Light Green = Belief State (detailed multi-line text)")  
            print("   🟡 Light Yellow = Intention State (full reasoning displayed)")
            print("   🔴 Light Red = Emotion State (comprehensive descriptions)")
            print("   📝 Strategy Boxes = Retrieved Strategy Content (no text truncation)")
            print("   🎯 Prediction Results = Predicted Action vs Actual Result (Up/Down)")
            print("   🔄 Strategy Updates = Backward Inference Learning Results (if errors)")
            print()
            print("💡 Text Display Features:")
            print("   ✅ Smart line wrapping prevents horizontal text stretching")
            print("   ✅ Extended text length (up to 500-800 characters)")
            print("   ✅ Complete text content - NO keyword filtering")
            print("   ✅ Full strategy and mental state descriptions displayed")
            print("   ✅ All interface labels in English for international accessibility")
            
        else:
            print("❌ Generation Failed!")
            print("Please check:")
            print("  1. Inference log files exist")
            print("  2. Graphviz is correctly installed")
            return 1
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
