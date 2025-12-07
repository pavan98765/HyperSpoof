# Contributing to HyperSpoof

Thank you for your interest in contributing to HyperSpoof! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use the issue templates provided
3. Provide detailed information about the problem

### Suggesting Enhancements

We welcome suggestions for new features and improvements:
1. Check if the enhancement has been suggested before
2. Provide a clear description of the proposed enhancement
3. Explain why it would be useful
4. Consider the scope and complexity

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- CUDA (for GPU development)

### Setup Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/HyperSpoof.git
   cd HyperSpoof
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

3. **Install development dependencies:**
   ```bash
   pip install pytest pytest-cov black flake8 pre-commit
   ```

4. **Setup pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## 📝 Code Style

### Python Style Guide

We follow PEP 8 with some modifications:
- Maximum line length: 127 characters
- Use type hints where appropriate
- Follow the existing code style

### Code Formatting

We use Black for code formatting:
```bash
black hyperspoof/
```

### Linting

We use flake8 for linting:
```bash
flake8 hyperspoof/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hyperspoof --cov-report=html

# Run specific test files
pytest tests/test_models.py -v
```

### Writing Tests

- Write tests for new features
- Aim for high test coverage
- Use descriptive test names
- Test edge cases and error conditions

### Test Structure

```python
def test_feature_name():
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

## 📦 Pull Request Process

### Before Submitting

1. **Ensure tests pass:**
   ```bash
   pytest
   ```

2. **Check code style:**
   ```bash
   black --check hyperspoof/
   flake8 hyperspoof/
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

### Pull Request Guidelines

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request:**
   - Use the PR template
   - Provide a clear description
   - Link related issues
   - Request reviews from maintainers

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All existing tests still pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🏗️ Project Structure

```
HyperSpoof/
├── hyperspoof/           # Main package
│   ├── models/          # Model definitions
│   ├── data/            # Data handling
│   ├── training/        # Training utilities
│   ├── metrics/         # Evaluation metrics
│   ├── utils/           # Utility functions
│   └── cli/             # Command-line interface
├── tests/               # Test suite
├── configs/             # Configuration files
├── examples/            # Example scripts
└── docs/                # Documentation
```

## 📚 Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google docstring format
- Include type hints
- Provide examples where helpful

### Example Docstring

```python
def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate evaluation metrics for face anti-spoofing.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary containing calculated metrics
        
    Example:
        >>> y_true = [0, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0]
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(metrics['accuracy'])
        0.75
    """
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information:**
   - Python version
   - PyTorch version
   - Operating system
   - CUDA version (if applicable)

2. **Steps to reproduce:**
   - Minimal code example
   - Expected vs actual behavior
   - Error messages and stack traces

3. **Additional context:**
   - Screenshots if applicable
   - Related issues or discussions

## 💡 Feature Requests

When suggesting features:

1. **Describe the feature:**
   - What it does
   - Why it's needed
   - How it would work

2. **Consider implementation:**
   - Technical feasibility
   - Impact on existing code
   - Backward compatibility

3. **Provide examples:**
   - Use cases
   - API design
   - Code examples

## 📋 Issue Labels

We use the following labels:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Incompatible API changes
- `MINOR`: New functionality (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version number updated
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared

## 🤔 Questions?

If you have questions about contributing:

1. Check existing issues and discussions
2. Create a new issue with the `question` label
3. Join our community discussions
4. Contact maintainers directly

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to HyperSpoof! 🎉
