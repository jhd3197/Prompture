Drivers Module
==============

.. automodule:: prompture.drivers
   :members:
   :undoc-members:
   :show-inheritance:

The drivers module provides a unified interface for connecting to various Large Language Model (LLM) providers through a consistent API. Each driver implements the same interface while handling provider-specific authentication, request formatting, and response parsing.

Overview
--------

The drivers system enables Prompture to work with multiple LLM providers seamlessly:

- **Unified Interface**: All drivers implement the same [`Driver`](../api/driver.rst) base class
- **Automatic Selection**: Use [`get_driver_for_model()`](#get_driver_for_model) for automatic driver selection based on model strings
- **Manual Control**: Use [`get_driver()`](#get_driver) for explicit driver instantiation  
- **Provider Flexibility**: Easy switching between providers without code changes
- **Cost Tracking**: Built-in token usage and cost calculation for supported models

Driver Selection Functions
--------------------------

get_driver_for_model()
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_driver_for_model

Automatically select and instantiate the appropriate driver based on a model string in the format ``provider/model``.

**Supported Model String Formats:**

- ``openai/gpt-4`` → [`OpenAIDriver`](#openaiddriver)
- ``anthropic/claude-3-sonnet-20240229`` → [`ClaudeDriver`](#claudedriver)  
- ``google/gemini-pro`` → [`GoogleDriver`](#googledriver)
- ``groq/llama2-70b-4096`` → [`GroqDriver`](#groqdriver)
- ``ollama/llama3`` → [`OllamaDriver`](#ollamadriver)
- ``azure/gpt-4`` → [`AzureDriver`](#azuredriver)

**Example:**

.. code-block:: python

   from prompture.drivers import get_driver_for_model
   
   # Automatic driver selection
   driver = get_driver_for_model("openai/gpt-4")
   
   # Use in extraction
   from prompture.core import manual_extract_and_jsonify
   result = manual_extract_and_jsonify(
       driver=driver,
       text="John is 25 years old",
       json_schema={"type": "object", "properties": {"age": {"type": "integer"}}}
   )

get_driver()
~~~~~~~~~~~

.. autofunction:: get_driver

Manually instantiate a specific driver by provider name with custom configuration.

**Supported Provider Names:**

- ``openai`` → [`OpenAIDriver`](#openaiddriver)
- ``anthropic`` → [`ClaudeDriver`](#claudedriver)
- ``google`` → [`GoogleDriver`](#googledriver)  
- ``groq`` → [`GroqDriver`](#groqdriver)
- ``ollama`` → [`OllamaDriver`](#ollamadriver)
- ``azure`` → [`AzureDriver`](#azuredriver)
- ``grok`` → [`GrokDriver`](#grokdriver)
- ``openrouter`` → [`OpenRouterDriver`](#openrouterdriver)
- ``lmstudio`` → [`LMStudioDriver`](#lmstudiodriver)
- ``huggingface`` → [`HuggingFaceDriver`](#huggingfacedriver)
- ``local`` → [`LocalHTTPDriver`](#localhttpdriver)

**Example:**

.. code-block:: python

   from prompture.drivers import get_driver
   
   # Manual driver instantiation with custom config
   driver = get_driver("openai")
   driver.api_key = "custom-key"
   driver.model = "gpt-4-turbo"

Available Driver Classes
------------------------

OpenAIDriver
~~~~~~~~~~~~

.. autoclass:: prompture.drivers.openai_driver.OpenAIDriver
   :members:
   :show-inheritance:

Driver for OpenAI's GPT models including GPT-4, GPT-3.5-turbo, and other OpenAI API-compatible models.

**Supported Models:**

- ``gpt-4o`` - Latest GPT-4 Omni model (recommended)
- ``gpt-4o-mini`` - Smaller, faster GPT-4 variant  
- ``gpt-4-turbo`` - GPT-4 Turbo with 128k context
- ``gpt-4`` - Standard GPT-4 model
- ``gpt-3.5-turbo`` - Fast and cost-effective model

**Configuration:**

.. code-block:: python

   from prompture.drivers import OpenAIDriver
   
   driver = OpenAIDriver(
       api_key="your-openai-key",  # Or set OPENAI_API_KEY env var
       model="gpt-4o"
   )

ClaudeDriver  
~~~~~~~~~~~~

.. autoclass:: prompture.drivers.claude_driver.ClaudeDriver
   :members:
   :show-inheritance:

Driver for Anthropic's Claude models with advanced reasoning capabilities.

**Supported Models:**

- ``claude-opus-4-6`` - Most capable Claude model
- ``claude-sonnet-4-6`` - Balanced performance model (recommended)
- ``claude-haiku-4-5-20251001`` - Fast and efficient model

**Configuration:**

.. code-block:: python

   from prompture.drivers import ClaudeDriver
   
   driver = ClaudeDriver(
       api_key="your-anthropic-key",  # Or set ANTHROPIC_API_KEY env var
       model="claude-sonnet-4-6"
   )

GoogleDriver
~~~~~~~~~~~~

.. autoclass:: prompture.drivers.google_driver.GoogleDriver
   :members:
   :show-inheritance:

Driver for Google's Gemini models with multimodal capabilities.

**Supported Models:**

- ``gemini-1.5-pro`` - Latest Gemini Pro model (recommended)
- ``gemini-1.5-flash`` - Fast Gemini model
- ``gemini-pro`` - Standard Gemini Pro
- ``gemini-pro-vision`` - Gemini with vision capabilities

**Configuration:**

.. code-block:: python

   from prompture.drivers import GoogleDriver
   
   driver = GoogleDriver(
       api_key="your-google-key",  # Or set GOOGLE_API_KEY env var
       model="gemini-1.5-pro"
   )

GroqDriver
~~~~~~~~~~

.. autoclass:: prompture.drivers.groq_driver.GroqDriver
   :members:
   :show-inheritance:

Driver for Groq's ultra-fast inference platform with open-source models.

**Supported Models:**

- ``llama-3.1-70b-versatile`` - Llama 3.1 70B (recommended)
- ``llama-3.1-8b-instant`` - Fast Llama 3.1 8B
- ``mixtral-8x7b-32768`` - Mixtral 8x7B with large context
- ``gemma2-9b-it`` - Google's Gemma 2 model

**Configuration:**

.. code-block:: python

   from prompture.drivers import GroqDriver
   
   driver = GroqDriver(
       api_key="your-groq-key",  # Or set GROQ_API_KEY env var  
       model="llama-3.1-70b-versatile"
   )

OllamaDriver
~~~~~~~~~~~~

.. autoclass:: prompture.drivers.ollama_driver.OllamaDriver
   :members:  
   :show-inheritance:

Driver for local models running through Ollama, enabling private and offline LLM usage.

**Popular Models:**

- ``llama3`` - Meta's Llama 3 model
- ``mistral`` - Mistral 7B model  
- ``codellama`` - Code-specialized Llama variant
- ``qwen2.5`` - Alibaba's Qwen 2.5 model
- ``deepseek-coder`` - Code-focused model

**Configuration:**

.. code-block:: python

   from prompture.drivers import OllamaDriver
   
   driver = OllamaDriver(
       endpoint="http://localhost:11434",  # Or set OLLAMA_BASE_URL env var
       model="llama3"
   )
   
   # First pull the model if not already available
   # Run: ollama pull llama3

AzureDriver
~~~~~~~~~~~

.. autoclass:: prompture.drivers.azure_driver.AzureDriver
   :members:
   :show-inheritance:

Driver for Azure OpenAI Service with enterprise-grade security and compliance.

**Configuration:**

.. code-block:: python

   from prompture.drivers import AzureDriver
   
   driver = AzureDriver(
       api_key="your-azure-key",
       api_base="https://your-resource.openai.azure.com/",
       api_version="2024-02-15-preview",
       deployment_name="gpt-4-deployment"
   )

GrokDriver
~~~~~~~~~~

.. autoclass:: prompture.drivers.grok_driver.GrokDriver
   :members:
   :show-inheritance:

Driver for xAI's Grok models with real-time information access.

**Supported Models:**

- ``grok-2-1212`` - Latest Grok 2 model (recommended)  
- ``grok-2-vision-1212`` - Grok 2 with vision capabilities
- ``grok-beta`` - Beta version of Grok

**Configuration:**

.. code-block:: python

   from prompture.drivers import GrokDriver
   
   driver = GrokDriver(
       api_key="your-xai-key",  # Or set XAI_API_KEY env var
       model="grok-2-1212"
   )

OpenRouterDriver
~~~~~~~~~~~~~~~

.. autoclass:: prompture.drivers.openrouter_driver.OpenRouterDriver
   :members:
   :show-inheritance:

Driver for OpenRouter, providing access to multiple model providers through a single API.

**Configuration:**

.. code-block:: python

   from prompture.drivers import OpenRouterDriver
   
   driver = OpenRouterDriver(
       api_key="your-openrouter-key",  # Or set OPENROUTER_API_KEY env var
       model="openai/gpt-4-turbo"  # Use OpenRouter model naming
   )

Local and Custom Drivers
------------------------

LMStudioDriver
~~~~~~~~~~~~~

.. autoclass:: prompture.drivers.lmstudio_driver.LMStudioDriver
   :members:
   :show-inheritance:

Driver for LM Studio, enabling local model hosting with OpenAI-compatible API.

**Configuration:**

.. code-block:: python

   from prompture.drivers import LMStudioDriver
   
   driver = LMStudioDriver(
       endpoint="http://localhost:1234",  # Default LM Studio endpoint
       model="local-model-name"
   )

HuggingFaceDriver
~~~~~~~~~~~~~~~~

.. autoclass:: prompture.drivers.hugging_driver.HuggingFaceDriver
   :members:
   :show-inheritance:

Driver for Hugging Face Inference Endpoints and hosted models.

**Configuration:**

.. code-block:: python

   from prompture.drivers import HuggingFaceDriver
   
   driver = HuggingFaceDriver(
       endpoint="https://api-inference.huggingface.co/models/model-name",
       token="your-hf-token"  # Or set HUGGINGFACE_TOKEN env var
   )

LocalHTTPDriver
~~~~~~~~~~~~~~

.. autoclass:: prompture.drivers.local_http_driver.LocalHTTPDriver
   :members:
   :show-inheritance:

Generic driver for custom local HTTP endpoints that implement a simple prompt-response API.

**Configuration:**

.. code-block:: python

   from prompture.drivers import LocalHTTPDriver
   
   driver = LocalHTTPDriver(
       endpoint="http://localhost:8080/generate"
   )

Driver Interface
----------------

All drivers implement the same interface defined by the base [`Driver`](../api/driver.rst) class:

**Core Methods:**

- ``generate(prompt: str, options: Dict[str, Any]) -> Dict[str, Any]`` - Send prompt and get response
- ``__init__(**kwargs)`` - Initialize with provider-specific configuration

**Response Format:**

All drivers return responses in this standardized format:

.. code-block:: python

   {
       "text": "Generated response text",
       "meta": {
           "model": "model-name",
           "prompt_tokens": 150,
           "completion_tokens": 80,
           "total_tokens": 230,
           "cost": 0.00046,  # USD cost estimate
           "provider": "openai"
       }
   }

Cost Tracking and Pricing
-------------------------

Drivers automatically calculate token usage and cost estimates based on current pricing:

**Cost Calculation Example:**

.. code-block:: python

   driver = get_driver_for_model("openai/gpt-4")
   result = driver.generate("Hello, world!", {})
   
   print(f"Tokens used: {result['meta']['total_tokens']}")
   print(f"Estimated cost: ${result['meta']['cost']:.4f}")

**Pricing Information:**

Each driver includes MODEL_PRICING dictionaries with current rates:

- **Input tokens**: Cost per 1K input tokens
- **Output tokens**: Cost per 1K output tokens  
- **Currency**: All costs in USD

Environment Variables
--------------------

Drivers support configuration through environment variables:

.. code-block:: bash

   # API Keys
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key" 
   export GOOGLE_API_KEY="your-google-key"
   export GROQ_API_KEY="your-groq-key"
   export XAI_API_KEY="your-xai-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   
   # Custom Endpoints
   export OPENAI_BASE_URL="https://api.openai.com/v1"
   export OLLAMA_BASE_URL="http://localhost:11434"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   
   # Azure-specific
   export AZURE_OPENAI_API_KEY="your-azure-key"
   export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

Error Handling
--------------

Drivers implement consistent error handling:

**Common Exceptions:**

- **ConnectionError**: Network connectivity issues
- **AuthenticationError**: Invalid API keys or credentials  
- **RateLimitError**: API rate limit exceeded
- **ModelNotFoundError**: Specified model not available
- **ValidationError**: Invalid request parameters

**Error Handling Example:**

.. code-block:: python

   try:
       driver = get_driver_for_model("openai/gpt-4")
       result = driver.generate("Hello", {})
   except ConnectionError as e:
       print(f"Network error: {e}")
   except Exception as e:
       print(f"Driver error: {e}")

Best Practices
--------------

1. **Use Environment Variables** for API keys instead of hardcoding them
2. **Handle Connection Errors** gracefully in production applications
3. **Monitor Token Usage** to control costs, especially with expensive models
4. **Choose Appropriate Models** based on your speed/quality/cost requirements:
   
   - **Speed**: Groq, Ollama (local), GPT-3.5-turbo
   - **Quality**: GPT-4, Claude-3-opus, Gemini-1.5-pro
   - **Cost**: GPT-3.5-turbo, Claude-3-haiku, Ollama (local)
   - **Privacy**: Ollama, LM Studio (local models)

5. **Test Locally First** with Ollama or LM Studio before using paid APIs
6. **Use Model-specific Features** like Claude's large context or Gemini's vision capabilities when needed
7. **Implement Fallbacks** between different providers for reliability