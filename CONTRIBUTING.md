# Contributing to MineMEETS

Thank you for your interest in contributing to MineMEETS! This project focuses on **MLOps best practices** and production-ready ML systems.

## Development Setup

### Prerequisites
- Python 3.10+
- Git
- Ollama installed locally
- Pinecone API key

### Setup Steps

1. **Fork and clone the repository**
```bash
git clone https://github.com/yourusername/MineMEETS.git
cd MineMEETS
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
make install-dev
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run tests**
```bash
make test
```

## Development Workflow

### Before Making Changes

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Run quality checks**
```bash
make check
```

### Making Changes

1. **Write clean, documented code**
   - Follow PEP 8 style guide
   - Add docstrings to functions and classes
   - Keep functions focused and testable

2. **Add tests**
   - Unit tests for new functions
   - Integration tests for end-to-end flows
   - Maintain >80% code coverage

3. **Format and lint**
```bash
make format
make lint
```

### Committing Changes

1. **Write clear commit messages**
```
feat: Add support for .mp4 video files
fix: Resolve embedding dimension mismatch
docs: Update README with new deployment options
test: Add integration tests for audio pipeline
```

2. **Keep commits atomic**
   - One logical change per commit
   - Related changes grouped together

### Submitting Pull Requests

1. **Push your branch**
```bash
git push origin feature/your-feature-name
```

2. **Create pull request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure CI checks pass
   - Request review from maintainers

3. **Address review feedback**
   - Make requested changes
   - Push updates to same branch
   - Respond to comments

## Code Quality Standards

### Python Style
- **Formatter**: Black (line length: 100)
- **Linter**: Pylint
- **Type hints**: Encouraged but not required
- **Docstrings**: Required for public APIs

### Testing
- **Framework**: Pytest
- **Coverage**: Minimum 80%
- **Async tests**: Use pytest-asyncio
- **Mocking**: Use unittest.mock

### Documentation
- **Code comments**: For complex logic only
- **Docstrings**: Google style
- **Architecture docs**: Keep ARCHITECTURE.md updated

## Areas for Contribution

### High Priority
- [ ] Additional test coverage
- [ ] Performance optimizations
- [ ] Error handling improvements
- [ ] Documentation enhancements

### Medium Priority
- [ ] New file format support
- [ ] Additional embedding models
- [ ] Enhanced retrieval strategies
- [ ] Monitoring dashboards

### Good First Issues
- [ ] Fix typos in documentation
- [ ] Add example notebooks
- [ ] Improve error messages
- [ ] Add more unit tests

## MLOps Focus

This project emphasizes **operational ML systems**, not research:

✅ **Do:**
- Improve reliability and observability
- Add production-ready features
- Enhance testing and validation
- Optimize performance
- Improve deployment processes

❌ **Don't:**
- Introduce complex research models
- Add experimental features without testing
- Ignore operational concerns
- Break backward compatibility without discussion

## Code Review Process

1. **Automated checks must pass**
   - Linting (Black, Pylint)
   - Tests (Pytest)
   - Docker build

2. **Manual review**
   - Code quality and style
   - Test coverage
   - Documentation updates
   - Operational impact

3. **Approval required**
   - At least one maintainer approval
   - All discussions resolved

## Release Process

Releases follow semantic versioning:
- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check ARCHITECTURE.md for system design

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

---

**Thank you for contributing to MineMEETS!**
