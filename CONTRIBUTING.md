# Contributing to MarketToM

Thank you for your interest in contributing to MarketToM!

## Setup for Development

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MarketToM.git
cd MarketToM
```

### 2. Configure API Keys

Copy the example configuration file and add your API keys:

```bash
cp config.example.json config.json
```

Edit `config.json` and replace `YOUR_API_KEY_HERE` with your actual API key.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For the web interface:

```bash
cd web
pip install -r requirements_web.txt
```

### 4. Data Setup

The repository includes sample data for 5 stocks (AAPL, FB, T, GOOG, AMZN).

For full datasets:
- **ACL18 (StockNet)**: Download from [original source]
- **CMIN-US**: Contact authors for access
- **CMIN-CN**: Contact authors for access

### 5. Running Tests

```bash
# Run basic inference
python run_new.py

# Start web interface
cd web
./start.sh
```

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to all functions and classes
- Comment complex logic

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Issues

Please use GitHub Issues to report bugs or suggest features.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

