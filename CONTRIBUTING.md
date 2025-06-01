# Contributing to Stock Price Prediction Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dependencies:
   ```bash
   ./setup.sh
   ```
4. Create a new branch for your feature or bugfix

## Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/stock-prediction-lstm.git
cd stock-prediction-lstm

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Code Style

We follow PEP 8 style guidelines. Please ensure your code:
- Has proper docstrings for all functions and classes
- Follows consistent naming conventions
- Includes type hints where appropriate
- Has adequate comments for complex logic

### Testing

Before submitting a pull request:
1. Test your changes thoroughly
2. Ensure all existing functionality still works
3. Add tests for new features
4. Run the example notebook to verify everything works

## Types of Contributions

### Bug Reports
- Use the issue tracker to report bugs
- Include steps to reproduce the issue
- Provide sample data if possible
- Include error messages and stack traces

### Feature Requests
- Describe the feature clearly
- Explain why it would be useful
- Provide examples of how it would be used

### Code Contributions
- Fork the repository and create a feature branch
- Make your changes with clear, descriptive commits
- Update documentation if necessary
- Submit a pull request

## Pull Request Process

1. **Create a descriptive pull request title**
2. **Provide a detailed description** of what your changes do
3. **Link to any related issues**
4. **Ensure tests pass** and functionality works
5. **Update documentation** if needed

### Pull Request Template
```
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] Manual testing performed
- [ ] Example notebook runs successfully

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Development Guidelines

### Adding New Features

1. **Data Processors**: Add new preprocessing methods to `src/data_processor.py`
2. **Models**: Extend `src/model.py` for new architectures
3. **Predictors**: Add prediction methods to `src/predictor.py`
4. **Visualizations**: Add plots to `src/visualizer.py`
5. **Utilities**: Add helper functions to `src/utils.py`

### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add comprehensive docstrings
- Handle errors gracefully
- Log important steps for debugging

### Documentation

- Update README.md for new features
- Add docstrings to all public functions
- Include examples in docstrings
- Update the example notebook if needed

## Areas for Contribution

### High Priority
- Additional model architectures (GRU, Transformer, etc.)
- More sophisticated risk classification methods
- Real-time data integration
- Improved error handling and validation

### Medium Priority
- Additional visualization options
- Performance optimizations
- More comprehensive testing
- Better configuration management

### Low Priority
- Code style improvements
- Documentation enhancements
- Example improvements
- Minor bug fixes

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Code comments for specific implementations

## Questions?

- Open an issue for questions about contributing
- Check existing issues and pull requests first
- Be respectful and constructive in all interactions

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.
