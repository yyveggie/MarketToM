# GitHub Setup Guide

## ✅ 准备工作已完成

Git 仓库已初始化，所有配置已完成。以下是详细说明：

### 📋 已创建的配置文件

1. **`.gitignore`** - 排除规则配置
   - ❌ 实验结果文件（PNG, Excel）
   - ❌ 实验脚本（plot_experiment_results.py, sample_generator.py, analyze_stress_test.py）
   - ❌ 策略库文件（*.json in storage/）
   - ❌ 推理日志（inference_logs/, backward_inference_logs/）
   - ❌ 大型数据集（只保留5个股票样本）
   - ✅ 核心代码、文档、模板

2. **`config.example.json`** - 配置文件模板
   - 隐藏敏感信息（API keys）
   - 用户需要复制并填写自己的配置

3. **`.gitattributes`** - Git LFS 配置
   - 大型 JSON 文件使用 LFS
   - 正确的行尾符处理

4. **`.gitkeep` 文件** - 保持空目录结构
   - `storage/strategy_database/.gitkeep`
   - `storage/inference_logs/.gitkeep`
   - `storage/backward_inference_logs/.gitkeep`
   - `storage/visualizations/.gitkeep`

### 📊 数据集策略

**仅包含 5 个股票样本：**
- AAPL (Apple)
- FB (Facebook/Meta)  
- T (AT&T)
- GOOG (Google/Alphabet)
- AMZN (Amazon)

**包含的分割：**
- Train
- Test
- Validation

**完整数据集：**
- StockNet: 87 stocks → 只提交 5 stocks
- CMIN-US: 115 stocks → 不提交
- CMIN-CN: 772 stocks → 不提交

完整数据集请参考 `DATA_SETUP.md`

### 🚫 排除的内容

#### 实验结果
- `experiment_results_comparison.png`
- `压力测试_相对下降率分析.xlsx`
- `数据集和实验.xlsx`

#### 实验脚本
- `plot_experiment_results.py`
- `sample_generator.py`
- `analyze_stress_test.py`

#### 运行时生成的文件
- 策略库 JSON 文件（284+ 个）
- 推理日志（298 个）
- 可视化图片

### 📝 新增的文档

1. **`CONTRIBUTING.md`** - 贡献指南
2. **`DATA_SETUP.md`** - 数据集设置说明
3. **`GITHUB_SETUP.md`** (本文件) - GitHub 设置说明

---

## 🚀 提交到 GitHub 的步骤

### 1. 检查将要提交的文件

```bash
# 运行检查脚本
./check_git_files.sh

# 或查看 git 状态
git status
```

### 2. 添加文件到暂存区

```bash
# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 检查暂存的文件
git status
```

### 3. 创建初始提交

```bash
git commit -m "Initial commit: MarketToM - Theory of Mind for Stock Market Prediction

- Core framework implementation with Causal Bayesian Network
- Self-refining cognitive learning mechanism
- Expert perspective method for robust predictions
- Web visualization interface (English)
- Sample data for 5 stocks (AAPL, FB, T, GOOG, AMZN)
- Complete documentation and setup guides
"
```

### 4. 创建 GitHub 仓库

在 GitHub 网站上：
1. 点击 "New repository"
2. 仓库名：`MarketToM`
3. 描述：`Theory of Mind Framework for Stock Market Prediction`
4. 选择 `Public` 或 `Private`
5. **不要**初始化 README（我们已经有了）
6. 点击 "Create repository"

### 5. 连接远程仓库并推送

```bash
# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/MarketToM.git

# 推送到 GitHub
git push -u origin main

# 如果分支名是 master，使用：
# git push -u origin master
```

---

## 📦 仓库大小预估

### 包含的内容
- 核心代码：~50 KB
- 模板文件：~10 KB
- 文档：~100 KB
- 5 个股票样本数据：~5-10 MB
- **总计：约 10-15 MB**

### 不包含的内容（节省空间）
- 完整数据集：~500 MB - 2 GB
- 策略库：~5 MB
- 推理日志：~50 MB
- 实验结果：~10 MB

---

## ⚙️ 用户设置指南

克隆仓库后，用户需要：

### 1. 配置 API Key

```bash
cp config.example.json config.json
# 编辑 config.json，填写 API key
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. （可选）下载完整数据集

参考 `DATA_SETUP.md` 获取完整数据集。

### 4. 运行测试

```bash
# 命令行模式
python run_new.py

# Web 界面
cd web
./start.sh
```

---

## 🔒 安全提示

### 已保护的敏感信息：
- ✅ API keys（通过 config.example.json）
- ✅ 策略库数据（通过 .gitignore）
- ✅ 推理日志（通过 .gitignore）
- ✅ 个人实验结果（通过 .gitignore）

### 注意事项：
- ⚠️ 永远不要提交真实的 `config.json`
- ⚠️ 检查 `.gitignore` 是否正确工作
- ⚠️ 推送前使用 `git status` 确认

---

## 📞 需要帮助？

如果遇到问题：
1. 查看 `README.md` 
2. 查看 `CONTRIBUTING.md`
3. 运行 `./check_git_files.sh` 检查配置
4. 在 GitHub 上开 Issue

---

## ✅ 检查清单

提交前确认：

- [ ] `.gitignore` 文件已创建
- [ ] `config.example.json` 已创建（不含敏感信息）
- [ ] 真实的 `config.json` 被忽略
- [ ] 实验脚本被忽略
- [ ] 实验结果被忽略
- [ ] 策略库文件被忽略
- [ ] 只包含 5 个股票样本
- [ ] 运行了 `./check_git_files.sh` 验证
- [ ] 所有文档都是英文
- [ ] README.md 完整且准确

**全部确认后，即可安全推送到 GitHub！** 🎉

