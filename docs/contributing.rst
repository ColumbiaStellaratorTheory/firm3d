Contributing to FIRM3D
=====================

We welcome contributions to FIRM3D! This document provides guidelines for contributing to the project.

Quick Start
-----------

1. **Set up development environment**:

   .. code-block:: bash

      pip install -e ".[dev]"
      pre-commit install

2. **Run tests and checks**:

   .. code-block:: bash

      ruff check --fix .
      ruff format .
      python -m unittest discover tests/
      cd docs && make html

Code Style
----------

We use `ruff <https://github.com/astral-sh/ruff>`_ for automated code formatting and linting.

**Key Commands:**

- ``ruff check .`` - Check for style issues
- ``ruff check --fix .`` - Auto-fix issues
- ``ruff format .`` - Format code

**Configuration:**
The project includes a ``pyproject.toml`` file with ruff configuration that excludes ``thirdparty/`` directories and ignores physics-specific conventions (E731, E741, E722).

**Pre-commit Hooks:**
Install with ``pre-commit install`` to automatically run code quality checks before each commit.

Development Workflow
-------------------

1. **Create a feature branch or fork** for your changes
2. **Make your changes** following the style guidelines
3. **Test locally** before pushing:

   .. code-block:: bash

      ruff check --fix .
      ruff format .
      python -m unittest discover tests/
      cd docs && make html

4. **Push and monitor CI**: Check GitHub Actions for automated tests
5. **Submit a pull request**
