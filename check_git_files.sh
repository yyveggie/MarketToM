#!/bin/bash
# Check what files will be committed to Git

echo "=========================================="
echo "Git 提交文件检查报告"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

echo "📋 将要提交的文件（分类显示）："
echo ""

echo "🔹 Python 核心代码:"
git add -n core/*.py 2>/dev/null | head -20

echo ""
echo "🔹 数据处理模块:"
git add -n data/*.py 2>/dev/null | head -10

echo ""
echo "🔹 Web 应用:"
git add -n web/*.py web/templates/*.html web/static/**/*.* 2>/dev/null | head -20

echo ""
echo "🔹 配置和模板:"
git add -n templates/*.xml config.example.json 2>/dev/null

echo ""
echo "🔹 文档:"
git add -n *.md 2>/dev/null

echo ""
echo "=========================================="
echo "❌ 将要排除的文件（验证）："
echo "=========================================="
echo ""

echo "📊 实验结果文件（应该被忽略）:"
ls -lh *.png *.xlsx 2>/dev/null | grep -v ".gitkeep" || echo "  ✅ 无实验结果文件"

echo ""
echo "🧪 实验脚本（应该被忽略）:"
ls -lh plot_experiment_results.py sample_generator.py analyze_stress_test.py 2>/dev/null || echo "  ✅ 实验脚本将被忽略"

echo ""
echo "💾 策略库文件（应该被忽略）:"
ls storage/strategy_database/*.json 2>/dev/null | head -5 || echo "  ✅ 无策略文件"

echo ""
echo "📝 推理日志（应该被忽略）:"
echo "  - inference_logs: $(ls storage/inference_logs/*.json 2>/dev/null | wc -l) 个文件"
echo "  - backward_logs: $(ls storage/backward_inference_logs/*.json 2>/dev/null | wc -l) 个文件"

echo ""
echo "=========================================="
echo "📦 数据集文件统计："
echo "=========================================="
echo ""

for dataset in "StockNet" "CMIN_CN" "CMIN_US"; do
    if [ -d "data/$dataset" ]; then
        echo "📂 $dataset:"
        for split in "Train" "Test" "Validation"; do
            if [ -d "data/$dataset/$split" ]; then
                total=$(ls -d data/$dataset/$split/*/ 2>/dev/null | wc -l)
                echo "  $split: $total 个股票目录"
                
                # 显示保留的股票
                for stock in AAPL FB T GOOG AMZN; do
                    if [ -d "data/$dataset/$split/$stock" ]; then
                        echo "    ✅ $stock (保留)"
                    fi
                done
            fi
        done
        echo ""
    fi
done

echo "=========================================="
echo "🔍 Git 状态预览"
echo "=========================================="
echo ""
echo "运行 'git status' 查看详细状态..."
echo "运行 'git add -n .' 查看将要添加的文件..."
echo ""
echo "✅ 检查完成！如果一切正常，可以运行："
echo "   git add ."
echo "   git commit -m 'Initial commit: MarketToM framework'"
echo ""

