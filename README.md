# THINKING-LLMS

A project for training and evaluating language models with enhanced thought processes.

## Overview

This project implements a framework for training language models to develop better reasoning and thought processes. It includes components for training, evaluation, and inference with various transformer-based models.

## Project Structure

```
.
├── config/          # Configuration files and parameters
├── data/           # Data processing and dataset management
├── evaluation/     # Model evaluation metrics and tools
├── models/         # Model architectures and management
├── training/       # Training loops and optimization
├── utils/          # Utility functions and helpers
├── tests/          # Unit and integration tests
├── main.py         # Main execution script
└── requirements.txt # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main training script:
```bash
python main.py
```

## Configuration

The project uses YAML configuration files located in the `config/` directory. Modify these files to adjust model parameters, training settings, and evaluation metrics.

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## License

MIT License 