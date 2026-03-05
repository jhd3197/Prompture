Contributing to Prompture
=========================

We welcome contributions to Prompture! This guide will help you get started with contributing to the project.

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.8 or higher
- Git for version control
- A GitHub account

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork the Repository**

   Fork the Prompture repository on GitHub to your account.

2. **Clone Your Fork**

   .. code-block:: bash

       git clone https://github.com/your-username/prompture.git
       cd prompture

3. **Create a Virtual Environment**

   .. code-block:: bash

       python -m venv prompture-dev
       source prompture-dev/bin/activate  # On Windows: prompture-dev\Scripts\activate

4. **Install in Development Mode**

   .. code-block:: bash

       pip install -e ".[dev]"

5. **Install Pre-commit Hooks** (if available)

   .. code-block:: bash

       pre-commit install

Types of Contributions
----------------------

We accept several types of contributions:

Bug Reports
~~~~~~~~~~~

If you find a bug, please create an issue with:

- **Clear description** of the problem
- **Steps to reproduce** the issue  
- **Expected vs actual behavior**
- **System information** (Python version, OS, etc.)
- **Code example** that demonstrates the issue

Feature Requests
~~~~~~~~~~~~~~~~

For new features, please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** and why it's needed
- **Provide examples** of how it would be used
- **Consider backward compatibility**

Documentation Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation contributions are highly valued:

- **Fix typos** and grammar errors
- **Improve clarity** of existing documentation  
- **Add examples** and use cases
- **Update outdated information**

Code Contributions
~~~~~~~~~~~~~~~~~~

Code contributions should:

- **Follow coding standards** (see below)
- **Include tests** for new functionality
- **Update documentation** as needed
- **Maintain backward compatibility** when possible

Development Workflow
--------------------

1. **Create a Branch**

   .. code-block:: bash

       git checkout -b feature/your-feature-name
       # or
       git checkout -b bugfix/issue-description

2. **Make Your Changes**

   Write your code following the project standards.

3. **Run Tests**

   .. code-block:: bash

       pytest tests/

4. **Update Documentation**

   Update relevant documentation files and docstrings.

5. **Commit Your Changes**

   .. code-block:: bash

       git add .
       git commit -m "feat: add new extraction method for PDFs"

6. **Push to Your Fork**

   .. code-block:: bash

       git push origin feature/your-feature-name

7. **Create Pull Request**

   Open a pull request from your fork to the main repository.

Coding Standards
----------------

Code Style
~~~~~~~~~~

- Follow **PEP 8** Python style guidelines
- Use **type hints** for function parameters and return values
- Write **clear, descriptive variable names**
- Keep **line length under 100 characters**

.. code-block:: python

    # Good example
    def extract_person_data(text: str, model_name: str = "openai/gpt-4") -> Dict[str, Any]:
        """Extract person information from text using specified model."""
        pass

    # Bad example  
    def extract(txt, m="gpt4"):
        pass

Documentation
~~~~~~~~~~~~~

- Write **comprehensive docstrings** for all public functions and classes
- Use **Google-style docstrings**
- Include **examples** in docstrings when helpful

.. code-block:: python

    def register_field(name: str, definition: Dict[str, Any]) -> None:
        """Register a custom field definition in the global registry.
        
        Args:
            name: Unique identifier for the field
            definition: Field specification dictionary containing type,
                description, instructions, and validation rules
                
        Raises:
            ValueError: If field definition is invalid
            KeyError: If required definition keys are missing
            
        Example:
            >>> register_field("skills", {
            ...     "type": "list",
            ...     "description": "List of professional skills",
            ...     "instructions": "Extract as comma-separated values",
            ...     "default": []
            ... })
        """

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

- Place tests in the ``tests/`` directory
- Name test files with ``test_`` prefix
- Group related tests in test classes
- Use descriptive test method names

.. code-block:: python

    import pytest
    from prompture import extract_and_jsonify, register_field

    class TestFieldRegistration:
        """Test field registration functionality."""
        
        def test_register_custom_field_success(self):
            """Test successful registration of custom field."""
            register_field("test_field", {
                "type": "str",
                "description": "Test field for unit testing"
            })
            # Add assertions here
            
        def test_register_field_invalid_type_raises_error(self):
            """Test that invalid field type raises ValueError."""
            with pytest.raises(ValueError):
                register_field("bad_field", {"type": "invalid_type"})

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest

    # Run specific test file
    pytest tests/test_field_definitions.py

    # Run with coverage
    pytest --cov=prompture tests/

    # Run with verbose output
    pytest -v

Test Coverage
~~~~~~~~~~~~~

- Aim for **high test coverage** (>80%)
- Test **both success and failure cases**
- Include **edge cases** and **boundary conditions**
- Mock external API calls in tests

Adding New LLM Drivers
----------------------

To add support for a new LLM provider:

1. **Create Driver Class**

   Create a new file in ``prompture/drivers/`` following the pattern:

   .. code-block:: python

       from prompture.driver import Driver
       import requests
       from typing import Dict, Any

       class NewProviderDriver(Driver):
           """Driver for NewProvider LLM service."""
           
           def __init__(self, model_name: str, **kwargs):
               super().__init__(model_name, **kwargs)
               self.api_key = kwargs.get('api_key') or os.getenv('NEWPROVIDER_API_KEY')
               self.base_url = kwargs.get('base_url', 'https://api.newprovider.com/v1')
               
           def ask_for_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
               """Send prompt to NewProvider API and return JSON response."""
               # Implement API call logic here
               pass

2. **Update Driver Registry**

   Add your driver to the driver registry in ``prompture/drivers/__init__.py``.

3. **Add Tests**

   Create comprehensive tests for your driver in ``tests/test_drivers.py``.

4. **Update Documentation**

   Add documentation for the new driver including setup instructions and examples.

Adding New Field Types
----------------------

To add built-in field definitions:

1. **Add Field Definition**

   Add the field to ``prompture/field_definitions.py``:

   .. code-block:: python

       FIELD_DEFINITIONS["new_field_type"] = {
           "type": "str",
           "description": "Description of the new field type",
           "instructions": "Instructions for LLM on how to extract this field",
           "default": "default_value",
           "nullable": True,
           "validation": {
               "pattern": r"^[A-Z][a-z]+$"  # Optional regex validation
           }
       }

2. **Add Tests**

   Test the field definition in ``tests/test_field_definitions.py``.

3. **Update Documentation**

   Document the new field type in the API documentation.

Documentation Contributions
---------------------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

To build the documentation locally:

.. code-block:: bash

    cd docs/
    pip install -r requirements.txt
    make html  # On Windows: make.bat html

The built documentation will be in ``docs/_build/html/``.

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

- Use **reStructuredText** format for documentation files
- Include **practical examples** for all features
- Keep **language clear and concise**
- Use **proper cross-references** with ``:doc:`` and ``:ref:`` directives

.. code-block:: rst

    See the :doc:`quickstart` guide for basic usage examples.
    
    For more details, refer to :func:`prompture.extract_and_jsonify`.

Pull Request Guidelines
-----------------------

Creating Good Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Use descriptive titles** that explain what the PR does
- **Write detailed descriptions** explaining the changes and why they were made
- **Reference related issues** using ``Fixes #123`` or ``Closes #456``
- **Keep PRs focused** - one feature or fix per PR
- **Update relevant documentation**

Pull Request Template
~~~~~~~~~~~~~~~~~~~~~

When creating a PR, include:

.. code-block:: markdown

    ## Description
    Brief description of changes made.

    ## Type of Change
    - [ ] Bug fix
    - [ ] New feature  
    - [ ] Documentation update
    - [ ] Performance improvement

    ## Testing
    - [ ] Added tests for new functionality
    - [ ] All existing tests pass
    - [ ] Manual testing completed

    ## Documentation  
    - [ ] Updated relevant documentation
    - [ ] Added docstrings for new functions
    - [ ] Updated API reference if needed

Code Review Process
~~~~~~~~~~~~~~~~~~~

All contributions go through code review:

1. **Automated checks** run on your PR (tests, linting, etc.)
2. **Maintainer review** for code quality and design
3. **Feedback incorporation** - address review comments
4. **Final approval** and merge

Common Review Feedback
~~~~~~~~~~~~~~~~~~~~~~

- **Add type hints** for better code clarity
- **Improve test coverage** for edge cases
- **Update documentation** to reflect changes
- **Follow naming conventions** consistently
- **Handle errors gracefully** with proper exception handling

Release Process
---------------

Prompture follows semantic versioning (SemVer):

- **Major version** (1.0.0) - Breaking changes
- **Minor version** (0.1.0) - New features, backward compatible  
- **Patch version** (0.0.1) - Bug fixes, backward compatible

Releases are managed by maintainers and include:

- Updated version numbers
- Changelog entries
- Git tags
- PyPI package uploads

Getting Help
------------

If you need help with contributing:

- **Check existing documentation** and examples first
- **Search existing issues** for similar problems
- **Create a discussion** for general questions
- **Join community channels** (if available)
- **Tag maintainers** in issues for urgent matters

Recognition
-----------

Contributors are recognized in:

- **CONTRIBUTORS.md** file listing all contributors
- **Release notes** highlighting major contributions  
- **Documentation credits** for documentation contributors

Thank you for contributing to Prompture! Your contributions help make structured LLM data extraction better for everyone.