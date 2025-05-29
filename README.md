# MarketToM

MarketToM是一个使用Theory of Mind(ToM)理论和因果贝叶斯网络(CBN)建模股票市场行为的系统。该项目将市场视为一个具有心智状态的集体认知实体，通过分析环境状态、信念、意图、情绪和行动之间的因果关系，来预测市场行为。

## 核心组件

1. **因果贝叶斯网络(CBN)结构**：
   - 环境状态 → 信念(Belief)
   - 信念 → 意图(Intent)
   - 意图 + 环境状态 → 情绪(Emotion)
   - 意图 + 情绪 → 行动(Action)

2. **核心模块**：
   - 前向推理(Forward Inference)：根据环境状态推断市场的信念、意图和情绪
   - 行动概率计算(Action Probability Calculator)：基于意图和情绪推断市场行动的概率
   - 后向推理(Backward Inference)：根据实际市场行动反向更新和优化策略库
   - 认知增强插件(CEP, Cognitive Enhancement Plugin)：策略库，存储和检索不同心智层次的策略

## 安装和使用

1. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

2. 配置:
   - 修改`config.json`文件配置API密钥和其他参数

3. 运行：
   ```
   python run.py
   ``` 