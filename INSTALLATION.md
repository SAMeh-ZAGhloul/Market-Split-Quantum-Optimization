# Installation Guide - Market Split Problem

This guide provides step-by-step instructions for setting up the Market Split Problem repository with all required dependencies.

## Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SAMeh-ZAGhloul/Market-Split-Problem.git
cd Market-Split-Problem
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create a new virtual environment
python3 -m venv market_split_env

# Activate the virtual environment
# On macOS/Linux:
source market_split_env/bin/activate

# On Windows:
# market_split_env\Scripts\activate
```

### 3. Install Dependencies

#### Option A: Install from requirements.txt (Recommended)

```bash
# Install all dependencies from the requirements file
pip install -r source/requirements.txt
```

#### Option B: Manual Installation

If you prefer to install dependencies manually or encounter issues with the requirements file:

```bash
# Core dependencies
pip install numpy>=1.20.0
pip install qiskit>=0.40.0
pip install qiskit-optimization>=0.5.0

# Quantum computing dependencies
pip install dimod>=0.12.0
pip install dwave-system>=1.0.0
pip install dwave-ocean-sdk>=6.0.0

# Lattice-based solver dependencies
pip install fpylll>=0.5.0
pip install cysignals>=1.0.0

# Optional: Classical optimization solvers
pip install pyomo>=6.0.0
pip install ortools>=9.0.0

# Note: Gurobi requires separate license and installation
# Visit: https://www.gurobi.com/downloads/
```

## Verification

### Test Installation

After installation, verify that all components work correctly:

```bash
# Activate the virtual environment if not already active
source market_split_env/bin/activate

# Test Python imports
python3 -c "
import numpy as np
import qiskit
import dimod
from fpylll import *
print('✓ All core dependencies imported successfully!')
print(f'✓ numpy {np.__version__}')
print(f'✓ qiskit {qiskit.__version__}')
print('✓ dimod imported successfully')
print('✓ fpylll imported successfully')
"

# Test solver imports
python3 -c "
import sys
sys.path.append('source')
from qiskit_solver import QiskitMarketSplitSolver
from lattice_solver import LatticeBasedSolver
print('✓ All solver modules imported successfully!')
"
```

### Run Example

Test the complete setup with a simple example:

```bash
# Run the benchmarking framework
python3 source/benchmark_framework.py

# Or run a simple example
python3 source/example_usage.py
```

## Troubleshooting

### Common Issues

#### 1. Python Version Conflicts

If you encounter Python version compatibility issues:

```bash
# Check Python version
python3 --version

# If using Python 3.14, some packages may not be compatible
# Consider using Python 3.9, 3.10, or 3.11
```

#### 2. Permission Errors

If you encounter permission errors:

```bash
# Use user installation
pip install --user -r source/requirements.txt

# Or use virtual environment (recommended)
python3 -m venv myenv
source myenv/bin/activate
pip install -r source/requirements.txt
```

#### 3. Missing System Dependencies

On macOS, you may need to install additional dependencies:

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 4. Virtual Environment Issues

If virtual environment creation fails:

```bash
# Install virtualenv if venv is not available
pip install virtualenv

# Create virtual environment using virtualenv
virtualenv market_split_env

# Activate
source market_split_env/bin/activate
```

#### 5. OR-Tools Installation Issues

If OR-Tools installation fails:

```bash
# Try installing specific version
pip install ortools==9.7.2996

# Or use conda instead of pip
conda install -c conda-forge ortools
```

#### 6. D-Wave SDK Issues

If D-Wave Ocean SDK installation has issues:

```bash
# Install specific components individually
pip install dimod dwave-system dwave-cloud-client
```

### Environment Variables (Optional)

For D-Wave quantum computing, you may need to configure API credentials:

```bash
# Set D-Wave API token (optional, for quantum hardware access)
export DWAVE_API_TOKEN='your_api_token_here'

# Add to your shell profile for persistence
echo 'export DWAVE_API_TOKEN="your_api_token_here"' >> ~/.zshrc
```

## Development Setup

### For Contributors

If you plan to contribute to the project:

```bash
# Clone the repository
git clone https://github.com/SAMeh-ZAGhloul/Market-Split-Problem.git
cd Market-Split-Problem

# Create development virtual environment
python3 -m venv dev_env
source dev_env/bin/activate

# Install in development mode
pip install -e .

# Install additional development dependencies
pip install pytest black flake8 jupyter
```

## Supported Platforms

- **macOS**: ✅ Fully supported
- **Linux**: ✅ Fully supported  
- **Windows**: ✅ Supported (with minor adjustments)

## Performance Notes

- **Classical Solvers**: Work best on problems up to 50 variables
- **Quantum Solvers**: Limited by qubit count (typically 15-20 variables for current hardware)
- **Lattice Solvers**: Very fast for small problems (m < 10, n < 50)

## Additional Resources

- [Qiskit Documentation](https://qiskit.org/documentation/)
- [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/)
- [Pyomo Documentation](https://pyomo.readthedocs.io/)
- [OR-Tools Documentation](https://developers.google.com/optimization)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/SAMeh-ZAGhloul/Market-Split-Problem/issues) page
2. Create a new issue with detailed error messages
3. Include your operating system, Python version, and installation steps

---

**Last Updated**: December 19, 2025  
**Tested with**: Python 3.11, macOS 14.0
