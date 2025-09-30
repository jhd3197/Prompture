Installation Guide
==================

This guide covers the installation and initial setup of Prompture.

Requirements
------------

* Python 3.8 or higher
* pip (Python package installer)

Installation
------------

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~~

The recommended way to install Prompture is from PyPI using pip:

.. code-block:: bash

    pip install prompture

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

To install the latest development version from GitHub:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-org/prompture.git
    cd prompture

    # Install in development mode
    pip install -e .

    # Or install directly from GitHub
    pip install git+https://github.com/your-org/prompture.git

Virtual Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~

It's recommended to use a virtual environment to avoid dependency conflicts:

.. code-block:: bash

    # Create a virtual environment
    python -m venv prompture-env

    # Activate it (Linux/macOS)
    source prompture-env/bin/activate

    # Activate it (Windows)
    prompture-env\Scripts\activate

    # Install prompture
    pip install prompture

Environment Configuration
-------------------------

Prompture uses environment variables for API configuration. Create a ``.env`` file in your project directory:

.. code-block:: bash

    # OpenAI Configuration
    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, uses default if not set

    # Anthropic Configuration
    ANTHROPIC_API_KEY=your_anthropic_api_key_here

    # Google AI Configuration
    GOOGLE_API_KEY=your_google_api_key_here

    # Groq Configuration
    GROQ_API_KEY=your_groq_api_key_here

    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY=your_azure_api_key_here
    AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    AZURE_OPENAI_API_VERSION=2023-12-01-preview

    # Local/Custom Endpoints
    LOCAL_API_BASE_URL=http://localhost:8000  # For local models
    OLLAMA_BASE_URL=http://localhost:11434    # For Ollama

Required API Keys
~~~~~~~~~~~~~~~~~

Depending on which LLM providers you plan to use, you'll need to obtain API keys:

**OpenAI**
  Get your API key from the `OpenAI Dashboard <https://platform.openai.com/api-keys>`_

**Anthropic (Claude)**
  Get your API key from the `Anthropic Console <https://console.anthropic.com/>`_

**Google AI (Gemini)**
  Get your API key from `Google AI Studio <https://aistudio.google.com/>`_

**Groq**
  Get your API key from the `Groq Console <https://console.groq.com/>`_

**Azure OpenAI**
  Set up Azure OpenAI service and get keys from the Azure Portal

Verification
------------

Verify your installation by running a simple test:

.. code-block:: python

    import prompture

    # Check version
    print(f"Prompture version: {prompture.__version__}")

    # Test basic import
    from prompture import extract_and_jsonify, field_from_registry

    print("✅ Prompture installed successfully!")

If you have API keys configured, you can test a basic extraction:

.. code-block:: python

    from prompture import extract_and_jsonify

    # Simple test (requires valid API key in .env)
    try:
        result = extract_and_jsonify(
            prompt="Extract info: John is 25 years old",
            fields={"name": "string", "age": "integer"},
            model_name="openai/gpt-3.5-turbo"  # or your preferred model
        )
        print("✅ API connection successful!")
        print(f"Result: {result}")
    except Exception as e:
        print(f"⚠️  API test failed: {e}")
        print("Make sure your API keys are properly configured in .env")

Troubleshooting
---------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'prompture'**
  Make sure you've installed prompture in the correct Python environment.

**API Key Errors**
  Ensure your ``.env`` file is in the correct location and contains valid API keys.

**Version Conflicts**
  If you encounter dependency conflicts, try creating a fresh virtual environment.

**Network Issues**
  If installation fails due to network issues, try:

  .. code-block:: bash

      pip install --upgrade pip
      pip install prompture --no-cache-dir

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development work, install with additional development dependencies:

.. code-block:: bash

    # Clone and install in development mode
    git clone https://github.com/your-org/prompture.git
    cd prompture
    pip install -e ".[dev]"

    # Run tests to verify installation
    pytest tests/

Docker Installation
~~~~~~~~~~~~~~~~~~~

You can also use Prompture in a Docker container:

.. code-block:: dockerfile

    FROM python:3.9-slim

    WORKDIR /app
    COPY requirements.txt .
    RUN pip install prompture

    COPY . .
    CMD ["python", "your_script.py"]

Next Steps
----------

After installation, check out the :doc:`quickstart` guide to learn how to use Prompture's core features, or browse the :doc:`examples` for practical use cases.

For detailed API documentation, see the :doc:`api/index` section.