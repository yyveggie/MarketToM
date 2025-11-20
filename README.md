# MarketToM: Stock Trend Prediction via Theory of Mind

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> This repository contains the implementation code for our paper *"Mining Latent Mental States from Market Data: A Theory-of-Mind Approach to Stock Trend Prediction"* (under review).

## Overview

MarketToM models the market as a cognitive entity with mental states - beliefs, intentions, and emotions - using a Causal Bayesian Network. Instead of directly mapping signals to predictions, we reconstruct the cognitive pathway underlying market behavior.

```
Environmental State → Belief → Intent ↘
                        ↓              → Action (Up/Down)
                    Emotion ↗
```

## Installation

```bash
git clone https://github.com/yyveggie/MarketToM.git
cd MarketToM
pip install -r requirements.txt
```

Configure your API key in `config.json`:

```json
{
  "api": {
    "active_llm_provider": "openai",
    "providers": {
      "openai": {
        "api_key": "your-api-key-here",
        "llm_model_default": "gpt-4o"
      }
    }
  }
}
```

## Quick Start

Command-line inference:
```bash
python run.py
```

Web interface:
```bash
cd web
python app.py
# Open http://localhost:8080
```

## Web Interface

The framework includes a user-friendly web interface for interactive experimentation and visualization.

### Homepage
![MarketToM Homepage](web/user_interface_1.png)

### Multi-Expert Prediction Analysis
![Expert Judgments](web/user_interface_2.png)

### Complete Inference Flow Visualization
![Inference Flow](web/user_interface_3.png)

## Project Structure

```
MarketToM/
├── core/                    # Forward/backward inference, action prediction
├── data/                    # StockNet, CMIN-US, CMIN-CN datasets
├── templates/               # Prompt templates
├── web/                     # Web visualization interface
├── generalization/          # Text masking and cross-market tools
└── evaluation/              # Evaluation metrics
```

## Datasets

We validate on three financial datasets:

| Dataset | Market | Period | Stocks | Instances |
|---------|--------|--------|--------|-----------|
| StockNet | US | 2014-2016 | 87 | ~30K |
| CMIN-US | US | 2020-2023 | 115 | ~45K |
| CMIN-CN | China | 2020-2023 | 772 | ~150K |

Each stock folder contains:
- `text_data.json` - Social media texts
- `price_data.json` - OHLCV data
- `labels.json` - Binary labels (1=Up, 0=Down)

## Key Features

- **Forward Inference**: Infer market mental states from signals
- **Backward Inference**: Self-refining learning from prediction errors
- **Expert Perspective Method**: Multi-expert prediction aggregation
- **Log-Confidence Weighting**: Robust probability calculation
- **CEP (Cognitive Enhancement Plugin)**: Adaptive strategy database

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

Paper currently under review. Citation information will be updated upon publication.

## Contact

For issues or questions, please open a GitHub issue or contact us via the repository.
