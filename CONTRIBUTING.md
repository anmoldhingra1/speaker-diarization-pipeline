# Contributing to Speaker Diarization Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for development.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- git

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/anmoldhingra1/speaker-diarization-pipeline.git
cd speaker-diarization-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Testing

### Running Tests

Run the full test suite:
```bash
pytest tests/ -v
```

Run tests for a specific module:
```bash
pytest tests/test_pipeline.py -v
```

Run tests matching a pattern:
```bash
pytest tests/ -k "test_cluster" -v
```

### Test Coverage

The test suite includes:
- **Unit tests**: Individual class and method tests
- **Integration tests**: End-to-end pipeline tests
- **Edge case tests**: Boundary conditions, empty inputs, error handling
- **90+ total tests** covering all public APIs

### Writing Tests

Follow these conventions when adding tests:

1. Create test files in `tests/` named `test_<module>.py`
2. Organize tests into classes like `Test<ClassName>`
3. Use descriptive test names: `test_<what>_<condition>`
4. Include docstrings for each test
5. Test both happy paths and edge cases

Example:
```python
class TestMyComponent:
    """Tests for MyComponent class."""
    
    def test_basic_functionality(self) -> None:
        """Test basic component behavior."""
        component = MyComponent()
        result = component.do_something()
        assert result == expected_value
    
    def test_edge_case_empty_input(self) -> None:
        """Test behavior with empty input."""
        component = MyComponent()
        result = component.do_something([])
        assert result == []
```

## Code Style

### Python 3.9 Compatibility

The codebase targets Python 3.9+. Follow these guidelines:

- Use `from __future__ import annotations` at the top of files for forward references
- Use `Optional[X]` instead of `X | None` for type hints
- Use `list[X]` instead of `List[X]` (requires the future import above)

### Type Hints

All public methods and classes should have type hints:

```python
def process(self, path: Path) -> DiarizationResult:
    """Process an audio file."""
    ...
```

### Linting

Run ruff to check code style:
```bash
ruff check .
```

Configuration in `pyproject.toml`:
- Enforces PEP 8 style
- Detects unused imports (F401)
- Enforces naming conventions

Fix issues automatically where possible:
```bash
ruff check . --fix
```

### Import Ordering

Imports are automatically organized by ruff:
1. Standard library imports
2. Third-party imports
3. Local imports

Separate groups with blank lines.

## Code Structure

### Module Organization

- `diarization/types.py`: Data classes and type definitions
- `diarization/vad.py`: Voice activity detection
- `diarization/embeddings.py`: Speaker embedding extraction
- `diarization/clustering.py`: Spectral clustering
- `diarization/segmenter.py`: Segment refinement
- `diarization/pipeline.py`: Main orchestration class

### Documentation

- Add docstrings to all public classes and methods
- Use Google-style docstrings:
```python
def method(self, param: str) -> int:
    """Short description.
    
    Longer description if needed.
    
    Args:
        param: Parameter description.
    
    Returns:
        Return value description.
    
    Raises:
        ValueError: When validation fails.
    """
    ...
```

## Pull Request Guidelines

### Before Submitting

1. Run tests locally:
```bash
pytest tests/ -v
```

2. Run linting:
```bash
ruff check .
```

3. Verify Python 3.9+ compatibility

4. Update documentation if behavior changes

5. Add tests for new features

### PR Checklist

- Tests pass on all Python versions (3.9, 3.10, 3.11, 3.12)
- Code follows style guidelines (ruff check)
- New public APIs have type hints
- Docstrings are added for public classes/methods
- CHANGELOG is updated if applicable
- No unused imports (ruff F401)

### Commit Messages

Write clear commit messages:
- First line: short summary (50 chars max)
- Blank line
- Detailed description if needed
- Reference issues: "Fixes #123"

Example:
```
Add spectral clustering tests

- Test eigenvalue estimation bounds
- Test k-means convergence
- Test affinity matrix properties

Fixes #42
```

## Reporting Issues

When reporting bugs:
1. Use a clear, descriptive title
2. Describe the problem clearly
3. Include steps to reproduce
4. Specify Python version and OS
5. Include error messages/tracebacks

## Questions?

Open a discussion or issue on GitHub for questions about contributing.

---

Thank you for contributing to Speaker Diarization Pipeline!
