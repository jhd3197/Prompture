Cross-Model Test Harness
========================

Prompture includes a spec-driven test runner that lets you compare extraction
quality, cost, and token usage across multiple LLM providers and models.

Quick Start
-----------

.. code-block:: bash

   # Run a built-in spec with auto-detected providers
   prompture test-suite specs/basic_extraction.json

   # Specify providers explicitly
   prompture test-suite specs/basic_extraction.json --providers openai,groq

   # Override models from the command line
   prompture test-suite specs/basic_extraction.json \
       --models openai/gpt-4o-mini,ollama/llama3.1:8b

   # Save JSON report to file
   prompture test-suite specs/basic_extraction.json -o report.json

   # JSON output to stdout
   prompture test-suite specs/basic_extraction.json --format json

Output
------

The default ``table`` format prints a per-test summary:

.. code-block:: text

   Cross-Model Test Results: basic_extraction
   ===========================================

   Test: person-simple
     Model                  Pass  Tokens      Cost
     ---------------------  -----  ------  --------
     openai/gpt-4o-mini      3/3     847   $0.0003
     ollama/llama3.1:8b      2/3    1203   $0.0000

   Overall: 5/6 passed, total cost: $0.0003

Spec Format
-----------

A spec file is a JSON document with three sections:

.. code-block:: json

   {
     "meta": {
       "project": "my-project",
       "suite": "suite-name",
       "version": "1.0"
     },
     "models": [
       {"id": "openai/gpt-4o-mini", "driver": "openai", "options": {}},
       {"id": "ollama/llama3.1:8b", "driver": "ollama", "options": {}}
     ],
     "tests": [
       {
         "id": "test-name",
         "prompt_template": "Extract info from: {text}",
         "inputs": [
           {"text": "John is 25 years old."}
         ],
         "schema": {
           "type": "object",
           "properties": {
             "name": {"type": "string"},
             "age": {"type": "integer"}
           },
           "required": ["name", "age"]
         }
       }
     ]
   }

**meta** -- Optional metadata included in the report output.

**models** -- List of models to test. Each entry specifies:

- ``id`` -- The full model string (``provider/model``).
- ``driver`` -- The provider name used to look up the driver.
- ``options`` -- Extra options passed to the driver's ``generate()`` call.

**tests** -- List of test cases. Each test has:

- ``id`` -- Unique identifier for the test.
- ``prompt_template`` -- Python format string. Keys come from each input dict.
- ``inputs`` -- List of dicts; each is formatted into the prompt template.
- ``schema`` -- JSON Schema that the response must validate against.

Built-in Specs
--------------

Prompture ships with example specs in the ``specs/`` directory:

``basic_extraction.json``
  Simple person, contact, and product extraction -- good for smoke testing
  new providers.

``schema_validation.json``
  Tricky schemas: nested objects, enums, arrays, nullable fields. Tests
  whether models handle complex structures correctly.

``strategy_comparison.json``
  Invoice and medical record extraction -- designed for comparing
  ``StructuredOutputStrategy`` effectiveness across providers.

Writing Custom Specs
--------------------

1. Copy one of the built-in specs as a starting point.
2. Define your schema using standard JSON Schema.
3. Write prompt templates with ``{placeholder}`` variables matching your
   input dicts.
4. Run with ``prompture test-suite your_spec.json``.

Tips:

- Use ``"required"`` in your schema to catch missing fields.
- Include multiple inputs per test to check consistency.
- Compare ``"options": {"temperature": 0}`` vs default for reproducibility.

For Contributors
----------------

When submitting a driver PR, run the basic extraction spec against your
provider to verify correct behavior:

.. code-block:: bash

   prompture test-suite specs/basic_extraction.json --providers your_provider

Include the table output in your PR description.
