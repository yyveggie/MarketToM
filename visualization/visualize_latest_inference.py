#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import MentalStateVisualizer


def main():
    print("🧠 MarketToM Multi-Agent Inference Visualiser")
    print("=" * 50)

    visualizer = MentalStateVisualizer()

    # 1. Architecture graph (CCN)
    print("\n🎨 Generating multi-agent CCN architecture graph...")
    ccn_path = visualizer.create_causal_network_graph()
    if ccn_path:
        print(f"   ✅ {ccn_path}")

    # 2. Latest inference flow
    print("\n🎨 Generating latest complete inference flow chart...")
    print("   Including: Multi-agent CCN + CEP + Dynamic Aggregation + Backward Learning")

    try:
        graph_path = visualizer.create_latest_complete_inference_graph()

        if graph_path:
            print(f"\n✅ Generation Successful!")
            print(f"📄 Graph File: {graph_path}")
            print()
            print("🎯 Graph Content:")
            print("   🔵 Light Blue   = Environmental State")
            print("   🟢 Green border = Retail Agent   (belief / intent / emotion)")
            print("   🔵 Blue border  = Institutional Agent")
            print("   🟠 Orange border= Arbitrageur Agent")
            print("   🟣 Purple       = Dynamic Weighted Aggregation")
            print("   📝 Yellow       = CEP Strategy Nodes")
            print("   🔄 Hexagon      = Inter-Agent Backward Learning (if error)")
            print()
            print("💡 Features:")
            print("   ✅ Supports both legacy (single-agent) and new multi-agent logs")
            print("   ✅ Per-agent mental states with CCN causal edges")
            print("   ✅ Agent weights shown in aggregation node")
            print("   ✅ Smart text wrapping for readability")
        else:
            print("❌ Generation Failed — check inference logs exist")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
