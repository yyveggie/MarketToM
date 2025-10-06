# MarketToM: Stock Trend Prediction via Theory of Mind

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

> **📄 Paper Status:** This repository contains the official implementation code for the paper *"Actions Speak Louder than Words, Yet Stem from the Mind: Stock Trend Prediction from a Theory-of-Mind Perspective"* (currently under review). The code and documentation provided here support full reproducibility of our experimental results.

---

## Overview

**MarketToM** is the official implementation of the framework introduced in the paper:

> **"Actions Speak Louder than Words, Yet Stem from the Mind: Stock Trend Prediction from a Theory-of-Mind Perspective"**
> 
> *Status: Under Review*

This framework pioneers a novel approach to financial market prediction by modeling the market as a **collective cognitive entity** with mental states. Drawing from Theory of Mind (ToM) principles in cognitive science, MarketToM constructs a **Causal Bayesian Network (CBN)** to explicitly represent and infer the market's:

- **Beliefs** (market perception of information)
- **Intentions** (market strategic goals)
- **Emotions** (market sentiment dynamics)
- **Actions** (price movements: Up/Down)

Unlike traditional approaches that directly map market signals to price predictions, MarketToM reconstructs the **cognitive pathway** underlying market behavior, achieving superior prediction accuracy and interpretability through explicit causal reasoning.

---

## Table of Contents

- [Key Features](#key-features)
- [Web Visualization Interface](#-web-visualization-interface)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Core Components](#core-components-and-paper-correspondence)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Advanced Usage](#advanced-usage)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Citation](#citation)
- [License](#license)

---

## Key Features

- 🧠 **Theory-of-Mind Modeling**: First framework to apply ToM principles to financial market prediction
- 🔗 **Causal Bayesian Network**: Explicit representation of causal relationships in market cognition
- 🔄 **Self-Refining Learning**: Adaptive strategy database that evolves through backward inference
- 📊 **Log-Confidence Weighting**: Robust action prediction algorithm with uncertainty quantification
- 👥 **Expert Perspective Method**: Multi-expert reasoning system for enhanced prediction reliability
- 🌍 **Cross-Market Generalization**: Validated on US and Chinese markets with strong transferability
- 🎨 **Interactive Visualization**: Real-time web interface for inference process exploration

---

## 🌐 Web Visualization Interface

We now provide a **real-time web visualization interface** for interactive inference and result display!

### Quick Start

```bash
python app.py
```

Then open http://localhost:8080 in your browser.

### Features

- ✅ **Real-time Inference Display**: See mental states (belief, intent, emotion) as they're inferred
- ✅ **Expert Judgments**: View detailed reasoning from 10 expert perspectives
- ✅ **Action Prediction**: Visualize prediction results with confidence scores
- ✅ **Backward Inference**: Automatic strategy database updates when predictions are incorrect
- ✅ **Interactive UI**: Select datasets, stocks, and configure parameters easily

For more details, see [`web/README.md`](web/README.md).

---

## Installation

### Prerequisites

- Python 3.10 or higher (recommended: Python 3.10+)
- pip or conda package manager
- (Optional) CUDA-compatible GPU for faster inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/yyveggie/MarketToM.git
cd MarketToM
```

### Step 2: Create Virtual Environment (Recommended)

#### Using Conda
```bash
conda create -n markettom python=3.10
conda activate markettom
```

#### Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

The project includes a comprehensive `requirements.txt` with all necessary dependencies:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- Flask (Web framework)
- OpenAI/Anthropic (LLM APIs)
- LangChain (LLM orchestration)
- PyTorch & TensorFlow (Deep learning)
- Scikit-learn, XGBoost, LightGBM (ML baselines)
- NetworkX (Causal graph modeling)
- Pandas, NumPy (Data processing)
- Matplotlib, Seaborn (Visualization)

### Step 4: Configure API Keys

Edit `config.json` and add your LLM provider API key:

```json
{
  "api": {
    "active_llm_provider": "openai",
    "providers": {
      "openai": {
        "api_key": "your-api-key-here",
        "base_url": "https://api.openai.com/v1",
        "llm_model_default": "gpt-4o"
      }
    }
  }
}
```

**Supported LLM Providers:**
- OpenAI GPT-4/GPT-3.5
- Compatible OpenAI API endpoints (e.g., Azure OpenAI, custom deployments)

---

## Quick Start

### Command-Line Interface

Run inference on a single stock:

```bash
python run.py
```

Run with custom configuration:

```bash
python run.py --config custom_config.json
```

### Web Interface

Launch the interactive visualization dashboard:

```bash
cd web
python app.py
```

Then navigate to http://localhost:8080 in your browser.

---

## Project Structure

```
MarketToM/
├── core/                    # Core functionality modules
│   ├── __init__.py         # Core class exports
│   ├── cep.py              # Cognitive Enhancement Plugin (CEP)
│   ├── forward_inference.py # Forward inference module
│   ├── backward_inference.py # Backward inference module
│   ├── calculate_action_prob.py # Action probability calculation
│   └── expert_perspectives.py # Expert perspectives module
├── data/                    # Data processing modules
│   ├── __init__.py         # Data processing function exports
│   ├── data_input.py       # Data loading and processing
├── generalization/          # Generalization testing and dataset preparation
│   ├── cmin_cn_text_masking.py # CMIN_CN dataset text masking
│   ├── cmin_text_masking.py    # CMIN_US dataset text masking
│   ├── stock_text_masking.py   # StockNet dataset text masking
│   ├── dataset_comparison.py   # Cross-dataset analysis tool
│   └── replace_company_names.py # Company name anonymization tool
├── templates/               # Prompt templates
│   ├── forward_prompt_template.xml      # Forward inference template
│   ├── backward_prompt_template.xml     # Backward inference template
│   ├── action_prob_prompt_template.xml  # Action probability template
│   └── expert_action_prob_template.xml  # Expert action probability template
├── web/                     # Web Visualization Interface
│   ├── app.py              # Flask backend
│   ├── start.sh            # Startup script
│   ├── templates/          # HTML templates (index.html)
│   └── static/             # CSS, JavaScript assets
├── visualization/           # Visualization tools
│   ├── mental_state_visualizer.py  # Mental state visualization
│   └── visualize_latest_inference.py # Latest inference visualizer
├── evaluation/              # Evaluation modules
│   └── evaluate_predictions.py # Evaluate prediction results
├── storage/                 # Storage directory
│   ├── inference_logs/      # Inference logs (JSON)
│   ├── backward_inference_logs/ # Backward inference logs
│   ├── strategy_database/   # Strategy database (JSON)
│   └── visualizations/      # Generated visualization images
├── config.json              # Configuration file (API keys, parameters)
├── config.example.json      # Configuration template
├── requirements.txt         # Python dependencies
├── run.py                   # Main program (command-line)
└── README.md                # Project documentation
```

## Core Components and Paper Correspondence

### Causal Bayesian Network (CBN) Architecture

MarketToM implements a four-layer cognitive architecture that mirrors human Theory of Mind reasoning:

```
Environmental State → Belief → Intent ↘
                        ↓              → Action (Up/Down)
                    Emotion ↗
```

**Causal Dependencies:**
1. `Environmental State → Belief`: Market observation informs belief formation
2. `Belief → Intent`: Strategic intentions emerge from market beliefs  
3. `(Belief, Environmental State) → Emotion`: Emotional states depend on both perception and reality
4. `(Intent, Emotion) → Action`: Actions result from the interplay of intentions and emotions

This structure operationalizes the cognitive pathway described in **Section 2.1** of the paper.

---

### Core Modules

#### 1. Forward Inference Engine (`core/forward_inference.py`)

**Purpose:** Infer latent mental states from observable market signals.

**Key Components:**
- `MentalStateInference`: Main inference class
- LLM-based reasoning with structured prompts
- Strategy retrieval from Cognitive Enhancement Plugin (CEP)

**Paper Reference:** Section 2.2 - "Mental State Inference"

**Process:**
```python
# Pseudocode
belief = infer_market_belief(environmental_state)
intent = infer_market_intent(belief)
emotion = infer_market_emotion(belief, environmental_state)
```

#### 2. Action Probability Calculator (`core/calculate_action_prob.py`)

**Purpose:** Predict market actions (Up/Down) with uncertainty quantification.

**Key Algorithms:**

a) **Log-Confidence Weighting (Algorithm 1 in paper)**
```
For each sample i:
  w_i = exp(log_confidence_i)
  
weighted_prob = Σ(w_i × p_i) / Σ(w_i)
```

b) **Expert Perspective Method**
- Generates 10 expert predictions with different analytical perspectives
- Aggregates via weighted consensus
- Enhances robustness against single-point failures

**Paper Reference:** Section 2.2.3 - "Action Prediction"

#### 3. Backward Inference Engine (`core/backward_inference.py`)

**Purpose:** Refine cognitive strategies when predictions fail.

**Mechanism:**
1. Detect prediction errors (predicted ≠ actual)
2. Analyze failure causality via LLM reasoning
3. Generate strategy updates (CREATE/MODIFY operations)
4. Update CEP database for future inference

**Paper Reference:** Section 2.3 - "Self-Refining Cognitive Learning"

**Operations:**
- **CREATE**: Add new strategies for unencountered scenarios
- **MODIFY**: Refine existing strategies based on error analysis

#### 4. Cognitive Enhancement Plugin (`core/cep.py`)

**Purpose:** Adaptive strategy storage and retrieval system.

**Features:**
- Vector-based similarity search for strategy retrieval
- Hierarchical storage (Belief/Intent/Emotion strategies)
- Dynamic strategy evolution through backward inference
- Persistence across inference sessions

**Paper Reference:** Section 2.2.1 - "Strategy Retrieval"

---

### Paper Contributions → Code Mapping

| **Paper Contribution** | **Implementation** | **Key Files** |
|------------------------|-------------------|---------------|
| ToM Framework Operationalization | CBN structure + Mental state inference | `core/forward_inference.py` |
| Self-Refining Learning | Backward inference + CEP updates | `core/backward_inference.py`, `core/cep.py` |
| Log-Confidence Weighting | Robust action prediction algorithm | `core/calculate_action_prob.py` |
| Expert Perspective Method | Multi-expert aggregation | `core/calculate_action_prob.py`, `core/expert_perspectives.py` |
| Cross-Market Generalization | Text masking + Anonymization | `generalization/` directory |

---

## Configuration

### Configuration File Structure

The `config.json` file contains all framework parameters. Key sections:

#### API Configuration

     ```json
{
     "api": {
       "active_llm_provider": "openai",
       "providers": {
         "openai": {
           "api_key": "YOUR_API_KEY_HERE",
        "base_url": "https://api.openai.com/v1",
        "llm_model_default": "gpt-4o"
      }
         }
       }
     }
     ```

#### Data Parameters

```json
{
  "data_params": {
    "dataset_name": "CMIN_CN",
    "split": "Test",
    "stock_name": "比亚迪",
    "default_window_size": 5,
    "skip_backward_inference": false
  }
}
```

**Key Parameters:**
- `dataset_name`: Dataset to use (CMIN_CN / CMIN_US / StockNet)
- `split`: Data split (Train / Test / Validation)
- `stock_name`: Target stock/company
- `default_window_size`: Historical text window size
- `skip_backward_inference`: Disable learning (for testing)

#### CEP Retrieval Settings

```json
{
  "cep_retrieval": {
    "default_top_k": 1,
    "similarity_threshold": 0.1,
    "belief_similarity_threshold": 0.1,
    "intent_similarity_threshold": 0.1,
    "emotion_similarity_threshold": 0.1
  }
}
```

**Key Parameters:**
- `default_top_k`: Number of strategies to retrieve per inference
- `similarity_threshold`: Minimum cosine similarity for strategy matching

#### Inference Parameters

```json
{
  "forward_inference_params": {
    "llm_temperature": 0.7,
    "max_retries": 5,
    "base_delay_seconds": 1
  },
  "action_probability_params": {
    "use_expert_perspective_method": true,
    "num_probabilities_to_generate": 10,
    "llm_temperature": 0.7
  },
  "backward_inference_params": {
    "llm_temperature": 0.7,
    "llm_max_tokens": 5000,
    "max_retries": 5
  }
}
```

**Key Parameters:**
- `llm_temperature`: Controls randomness (0.0-1.0)
- `use_expert_perspective_method`: Enable multi-expert prediction
- `num_probabilities_to_generate`: Number of expert samples
- `max_retries`: API retry attempts on failure

---

## Datasets

### Supported Benchmarks

MarketToM has been validated on three major financial datasets:

| Dataset | Market | Period | Stocks | Instances | Language |
|---------|--------|--------|--------|-----------|----------|
| **ACL18 (StockNet)** | US (S&P 500) | 2014-2016 | 87 | ~30K | English |
| **CMIN-US** | US Markets | 2020-2023 | 115 | ~45K | English |
| **CMIN-CN** | Chinese Markets | 2020-2023 | 772 | ~150K | Chinese |

### Data Structure

Each dataset follows a standardized format:

```
data/
└── {DATASET_NAME}/
    └── {SPLIT}/                    # Train / Test / Validation
        └── {STOCK_NAME}/
            ├── text_data.json      # Social media / news texts
            ├── price_data.json     # Historical price information
            └── labels.json         # Ground truth (Up=1, Down=0)
```

**File Formats:**

- `text_data.json`: Daily aggregated texts (tweets, news articles)
- `price_data.json`: OHLCV (Open, High, Low, Close, Volume) data
- `labels.json`: Binary labels for next-day trend (1=Up, 0=Down)

### Dataset Configuration

Specify the target dataset in `config.json`:

```json
{
  "data_params": {
    "dataset_name": "CMIN_CN",     # or "CMIN_US", "StockNet"
    "split": "Test",                # or "Train", "Validation"
    "stock_name": "比亚迪",         # Target stock
    "default_window_size": 5        # Historical context window
  }
}
```

---

## Advanced Usage

### 1. Generalization Experiments

MarketToM incorporates robust cross-market generalization capabilities. We provide tools for entity anonymization to ensure models learn **generalizable market principles** rather than entity-specific patterns.

#### Text Masking Tools

Dataset-specific masking tools that anonymize company/entity names:

```bash
# Chinese market data (CMIN-CN)
python generalization/cmin_cn_text_masking.py

# US market data (CMIN-US)
python generalization/cmin_text_masking.py

# StockNet dataset
python generalization/stock_text_masking.py
```

**Purpose:**
- Remove entity-specific information from text data
- Force model to learn generalizable market dynamics
- Enable zero-shot transfer across stocks/markets

#### Cross-Dataset Analysis

Tools for ensuring consistent preprocessing:

```bash
# Compare missing data patterns across datasets
python generalization/dataset_comparison.py

# Generic company name replacement
python generalization/replace_company_names.py --dataset CMIN_CN --output masked_data/
```

**Paper Reference:** Section 3.4 - "Generalization Experiments"

---

### 2. Semantic Stress Testing

Validate model robustness under controlled linguistic perturbations.

#### Generate Perturbed Data

```bash
# Default: StockNet Train split
   python data/generate_semantic_perturbations.py

# Custom stocks and parameters
python data/generate_semantic_perturbations.py \
  --stocks AAPL FB GOOGL \
  --seed 2025 \
  --output-root data/StockNet_SemPerturb/Run1
```

#### Output Structure

```
data/StockNet_SemPerturb/
└── Train/
    └── {TICKER}/
        ├── text_data.json       # Perturbed texts
        ├── price_data.json      # Original prices (copied)
        └── labels.json          # Original labels (copied)
```

**Perturbation Strategy:**
- Controlled tonal shifts (positive ↔ neutral ↔ negative)
- Preserve factual content
- Test mental state inference robustness

**Prompt Template:** `templates/semantic_perturbation_prompt_template.xml`

**Paper Reference:** Ablation studies on model robustness

---

### 3. Batch Inference

Run inference on multiple stocks:

```python
# Python script example
from core import MarketToM
import json

with open('config.json') as f:
    config = json.load(f)

# Initialize framework
framework = MarketToM(config)

# Batch processing
stocks = ["AAPL", "GOOGL", "MSFT"]
for stock in stocks:
    config['data_params']['stock_name'] = stock
    result = framework.run_inference()
    print(f"{stock}: {result}")
```

---

## Evaluation

### Run Evaluation Script

Compute comprehensive metrics on prediction results:

```bash
python evaluation/evaluate_predictions.py
```

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall prediction correctness |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1 Score** | Harmonic mean of precision and recall |
| **MCC** | Matthews Correlation Coefficient (balanced metric) |
| **AUC-ROC** | Area under ROC curve |

### Expected Performance

| Dataset | Accuracy | F1 | MCC |
|---------|----------|-----|-----|
| CMIN-CN (Test) | 58.3% | 0.576 | 0.167 |
| CMIN-US (Test) | 56.7% | 0.552 | 0.134 |
| StockNet (Test) | 54.8% | 0.541 | 0.096 |

**Note:** Financial prediction is inherently challenging. Performance above 50% (random baseline) demonstrates predictive signal.

**Paper Reference:** Section 3 - "Experimental Results"

---

## Visualization

MarketToM includes a comprehensive visualization suite for analyzing inference processes and cognitive pathways.

### Quick Start - Latest Inference Visualization

Generate a complete flow chart of the most recent inference:

```bash
# Method 1: Simple script (Recommended)
python visualization/visualize_latest_inference.py

# Method 2: Full tool
python visualize.py --latest
```

This creates a detailed visualization showing:
- **Forward inference** of all mental states (belief, intention, emotion)
- **Retrieved strategies** with specific content
- **Prediction results** vs actual outcomes  
- **Backward inference corrections** (if prediction errors occurred)

### Other Visualization Options

```bash
# Generate causal Bayesian network structure
python visualize.py --causal-only

# Generate complete analysis with multiple graphs
python visualize.py --max-graphs 5

# Generate strategy evolution graphs
python visualize.py --strategy belief
```

### Output Files

All visualizations are saved to `./storage/visualizations/`:
- `latest_complete_inference.png` - Most recent complete inference flow
- `causal_bayesian_network.png` - Framework architecture
- `*_strategy_evolution.png` - Strategy learning progression
- `inference_timeline.png` - Mental state evolution over time

### Visual Elements

- 🔵 **Light Blue** - Environmental states (with expanded text content)
- 🟢 **Light Green** - Belief states (showing detailed reasoning)
- 🟡 **Light Yellow** - Intention states (with full inference text)
- 🔴 **Light Red** - Emotion states (comprehensive descriptions)
- 📝 **Strategy boxes** - Retrieved strategy content (full strategy details, not truncated)
- 🎯 **Prediction/Result** - Action predictions and outcomes (Up/Down with probabilities)
- 🔄 **Updates** - Backward inference strategy modifications (CREATE/MODIFY operations)

### Text Display Improvements

- **Extended text length**: Strategy and mental state descriptions now show up to 200-300 characters instead of 80
- **Smart line wrapping**: Text automatically wraps at word boundaries to prevent horizontal stretching
- **Multiple sentence support**: Key phrases are extracted from multiple relevant sentences
- **Enhanced keyword detection**: Improved detection of important market-related terms
- **All labels in English**: Complete interface in English for international accessibility
- **Optimized readability**: Different content types use appropriate line widths (35-60 characters per line)

All visualization features are now fully integrated with extended text display and English interface.

---

## Citation

If you use MarketToM in your research, please cite our paper:

```bibtex
@article{markettom2025,
  title={Actions Speak Louder than Words, Yet Stem from the Mind: Stock Trend Prediction from a Theory-of-Mind Perspective},
  author={[Authors]},
  journal={Under Review},
  year={2025}
}
```

**Note**: This paper is currently under review. Citation information will be updated upon publication.

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** style guidelines for Python code
3. **Add docstrings** to all public functions and classes
4. **Test your changes** thoroughly before submitting
5. **Submit a pull request** with a clear description

### Areas for Contribution

- 🐛 Bug fixes and performance improvements
- 📊 Support for additional datasets
- 🧪 New evaluation metrics
- 🎨 Visualization enhancements
- 📝 Documentation improvements
- 🌐 Multi-language support

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **OpenAI API**: Subject to OpenAI's terms of service
- **PyTorch**: BSD-style license
- **NumPy/Pandas**: BSD licenses

---

## Acknowledgments

We thank the following for their contributions to this research:

- **Datasets**: ACL18 (StockNet), CMIN-US, CMIN-CN providers
- **LLM Providers**: OpenAI for GPT-4 API access
- **Community**: Open-source contributors and reviewers

---

## Contact

For questions, issues, or collaboration inquiries:

- **GitHub Issues**: [https://github.com/yyveggie/MarketToM/issues](https://github.com/yyveggie/MarketToM/issues)
- **Paper**: Currently under review (arXiv link will be added upon publication)

---

## Changelog

### Version 1.1.0 (January 2025)
- ✨ Added real-time web visualization interface with English UI
- 🔧 Improved backward inference strategy updates
- 📊 Enhanced expert perspective method
- 🐛 Fixed data loading issues for CMIN datasets
- 📝 Comprehensive documentation updates
- 🌐 Repository prepared for public release

### Version 1.0.0 (2024)
- 🎉 Initial release
- 🧠 Core Theory-of-Mind framework implementation
- 📈 Support for CMIN-CN, CMIN-US, StockNet datasets
- 🔄 Self-refining cognitive learning mechanism
- 📊 Log-confidence weighting algorithm

---

## FAQ

<details>
<summary><strong>Q: What LLM models are supported?</strong></summary>

A: Currently, we support OpenAI GPT-4 and GPT-3.5 models, as well as any OpenAI-compatible API endpoints (e.g., Azure OpenAI, local deployments).
</details>

<details>
<summary><strong>Q: Can I use my own dataset?</strong></summary>

A: Yes! Follow the data structure format described in the [Datasets](#datasets) section. Ensure your data includes:
- `text_data.json`: Daily texts
- `price_data.json`: OHLCV data
- `labels.json`: Binary labels (1=Up, 0=Down)
</details>

<details>
<summary><strong>Q: How much does it cost to run inference?</strong></summary>

A: Costs depend on your LLM provider. For OpenAI GPT-4:
- ~2000-5000 tokens per stock per day
- Estimated $0.10-0.30 per prediction with GPT-4
- Use GPT-3.5-turbo for lower costs (~$0.01-0.03 per prediction)
</details>

<details>
<summary><strong>Q: Why is backward inference not updating strategies?</strong></summary>

A: Check that:
1. `skip_backward_inference` is set to `false` in `config.json`
2. Predictions are actually incorrect (backward inference only runs on errors)
3. API credentials are valid and have sufficient quota
</details>

<details>
<summary><strong>Q: Can I run MarketToM without a GPU?</strong></summary>

A: Yes! MarketToM primarily uses LLM APIs and doesn't require local GPU resources. CPU-only execution is fully supported.
</details>

---

**⭐ Star us on GitHub if you find MarketToM useful!**

**📄 Paper:** Currently under review. Links will be added upon publication.

**🌐 Repository:** [https://github.com/yyveggie/MarketToM](https://github.com/yyveggie/MarketToM)