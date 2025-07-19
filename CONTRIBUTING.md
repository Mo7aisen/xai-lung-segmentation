# 🤝 Contributing to XAI Lung Segmentation Analysis

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## 🚀 Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### 📋 Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Add tests** if you've added code that should be tested
3. **Update documentation** if you've changed APIs
4. **Ensure the test suite passes**
5. **Make sure your code follows our style guidelines**
6. **Issue a pull request**

### 🔧 Development Setup

1. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/xai-lung-segmentation.git
   cd xai-lung-segmentation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### 🧪 Testing

Run the full test suite:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest --cov=src tests/
```

### 📏 Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pre-commit** for automated checks

Format your code:
```bash
black src/ tests/
```

Check linting:
```bash
flake8 src/ tests/
```

Type checking:
```bash
mypy src/
```

### 🏷️ Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```
feat(dashboard): add real-time threshold adjustment
fix(model): resolve memory leak in training loop
docs(readme): update installation instructions
```

## 🐛 Bug Reports

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](../../issues).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## 💡 Feature Requests

We welcome feature requests! Please:

1. **Check existing issues** to avoid duplicates
2. **Provide clear motivation** for the feature
3. **Describe the proposed solution** in detail
4. **Consider alternatives** and mention them

## 📄 License

By contributing, you agree that your contributions will be licensed under its MIT License.

## 🏆 Recognition

Contributors will be recognized in:
- The project README
- Release notes for significant contributions
- Our contributors hall of fame

## 🆘 Getting Help

- 📧 **Email**: dev@xai-project.com
- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Use GitHub Issues for bugs and feature requests

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Captum Documentation](https://captum.ai/)
- [Project Documentation](./docs/)

Thank you for contributing! 🙏