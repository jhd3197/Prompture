Runner Module
=============

.. automodule:: prompture.runner
   :members:
   :undoc-members:
   :show-inheritance:

The runner module provides test suite functionality for executing and validating JSON extraction tests across multiple LLM models and drivers.

Overview
--------

The runner module enables:

- **Batch Testing**: Execute extraction tests across multiple models simultaneously
- **Validation**: Automatic validation of extraction results against expected outputs
- **Comparison**: Compare performance and accuracy across different LLM providers
- **Reporting**: Generate detailed test results and performance metrics

Main Functions
--------------

run_suite_from_spec()
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: run_suite_from_spec

Execute a test suite specification against multiple drivers and collect comprehensive results.

**Features:**

- **Multi-model Testing**: Run the same tests across different LLM providers
- **Automatic Validation**: Compare extraction results against expected outputs
- **Error Handling**: Graceful handling of driver failures and extraction errors
- **Performance Metrics**: Collect timing and token usage data
- **Detailed Reporting**: Generate comprehensive test result summaries

**Test Suite Specification Format:**

.. code-block:: python

   test_spec = {
       "name": "Person Extraction Test Suite",
       "description": "Test extraction of person information",
       "tests": [
           {
               "name": "Basic Person Info",
               "input_text": "John Smith is 25 years old",
               "schema": {
                   "type": "object",
                   "properties": {
                       "name": {"type": "string"},
                       "age": {"type": "integer"}
                   }
               },
               "expected": {
                   "name": "John Smith",
                   "age": 25
               }
           }
       ]
   }

**Example Usage:**

.. code-block:: python

   from prompture.runner import run_suite_from_spec
   from prompture.drivers import get_driver_for_model
   
   # Prepare drivers for testing
   drivers = {
       "gpt-4": get_driver_for_model("openai/gpt-4"),
       "claude": get_driver_for_model("anthropic/claude-3-sonnet-20240229"),
       "gemini": get_driver_for_model("google/gemini-pro")
   }
   
   # Run the test suite
   results = run_suite_from_spec(test_spec, drivers)
   
   # Analyze results
   for driver_name, driver_results in results.items():
       print(f"Results for {driver_name}:")
       print(f"  Tests passed: {driver_results['passed']}/{driver_results['total']}")
       print(f"  Average accuracy: {driver_results['accuracy']:.2%}")

**Return Value Structure:**

.. code-block:: python

   {
       "driver_name": {
           "passed": 8,           # Number of tests passed
           "total": 10,           # Total number of tests
           "accuracy": 0.85,      # Overall accuracy score
           "avg_time": 1.23,      # Average response time (seconds)
           "total_tokens": 1500,  # Total tokens used
           "total_cost": 0.045,   # Total estimated cost (USD)
           "test_results": [      # Individual test results
               {
                   "test_name": "Basic Person Info",
                   "passed": True,
                   "expected": {...},
                   "actual": {...},
                   "accuracy_score": 1.0,
                   "response_time": 1.1,
                   "tokens_used": 150,
                   "cost": 0.0045,
                   "error": None
               }
           ]
       }
   }

Test Suite Components
--------------------

Test Specification Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each test suite specification should contain:

**Suite Level Properties:**

- ``name``: Human-readable name for the test suite
- ``description``: Detailed description of what the suite tests
- ``tests``: Array of individual test cases

**Individual Test Properties:**

- ``name``: Name of the specific test case
- ``input_text``: Text to extract information from
- ``schema``: JSON schema defining the expected output structure
- ``expected``: Expected extraction results for validation
- ``options``: Optional driver-specific options

Validation and Scoring
~~~~~~~~~~~~~~~~~~~~~

The runner automatically validates extraction results:

**Validation Methods:**

1. **Schema Validation**: Ensures output matches the JSON schema structure
2. **Value Comparison**: Compares extracted values against expected results
3. **Type Checking**: Validates that extracted data types are correct
4. **Completeness**: Checks that all required fields are present

**Accuracy Scoring:**

- **Perfect Match**: 1.0 score when extracted data exactly matches expected
- **Partial Match**: Proportional score based on correct fields
- **Schema Valid**: Minimum score for schema-compliant but incorrect data
- **Invalid**: 0.0 score for schema violations or extraction failures

Performance Metrics
~~~~~~~~~~~~~~~~~~~

The runner collects comprehensive performance data:

**Timing Metrics:**

- Response time per test
- Average response time per driver
- Total execution time

**Usage Metrics:**

- Token usage per test and driver
- Cost estimation based on model pricing
- Request/response sizes

**Quality Metrics:**

- Accuracy scores per test
- Pass/fail rates
- Error categorization

Advanced Usage Patterns
-----------------------

Comparative Analysis
~~~~~~~~~~~~~~~~~~~

Run the same tests across multiple providers to compare performance:

.. code-block:: python

   # Test multiple providers
   providers = {
       "openai-gpt4": get_driver_for_model("openai/gpt-4"),
       "openai-gpt35": get_driver_for_model("openai/gpt-3.5-turbo"),
       "anthropic-claude": get_driver_for_model("anthropic/claude-3-sonnet-20240229"),
       "google-gemini": get_driver_for_model("google/gemini-pro"),
       "groq-llama": get_driver_for_model("groq/llama2-70b-4096")
   }
   
   results = run_suite_from_spec(comprehensive_test_spec, providers)
   
   # Analyze comparative performance
   performance_summary = {}
   for provider, result in results.items():
       performance_summary[provider] = {
           "accuracy": result["accuracy"],
           "avg_time": result["avg_time"],
           "cost_per_test": result["total_cost"] / result["total"],
           "tokens_per_test": result["total_tokens"] / result["total"]
       }

Regression Testing
~~~~~~~~~~~~~~~~~

Use the runner for regression testing when updating Prompture:

.. code-block:: python

   # Load test suite from file
   import json
   with open("regression_tests.json") as f:
       regression_spec = json.load(f)
   
   # Run against current production setup
   production_results = run_suite_from_spec(regression_spec, production_drivers)
   
   # Compare against baseline results
   baseline_accuracy = 0.92
   current_accuracy = production_results["main_driver"]["accuracy"]
   
   if current_accuracy < baseline_accuracy * 0.95:  # 5% tolerance
       print("WARNING: Accuracy regression detected!")

Custom Test Generation
~~~~~~~~~~~~~~~~~~~~~

Generate test suites programmatically:

.. code-block:: python

   def generate_field_tests(field_definitions):
       """Generate test cases for field definitions."""
       tests = []
       
       for field_name, field_def in field_definitions.items():
           test = {
               "name": f"Extract {field_name}",
               "input_text": f"Sample text containing {field_name} information",
               "schema": {
                   "type": "object", 
                   "properties": {
                       field_name: {"type": field_def["type"].__name__.lower()}
                   }
               },
               "expected": {field_name: field_def.get("default")}
           }
           tests.append(test)
       
       return {"name": "Field Definition Tests", "tests": tests}

Integration with Testing Frameworks
-----------------------------------

The runner integrates well with standard testing frameworks:

**pytest Integration:**

.. code-block:: python

   import pytest
   from prompture.runner import run_suite_from_spec
   
   def test_extraction_accuracy():
       results = run_suite_from_spec(test_spec, test_drivers)
       
       for driver_name, result in results.items():
           assert result["accuracy"] >= 0.8, f"{driver_name} accuracy too low"
           assert result["passed"] == result["total"], f"{driver_name} had failures"

**unittest Integration:**

.. code-block:: python

   import unittest
   from prompture.runner import run_suite_from_spec
   
   class TestExtractionSuite(unittest.TestCase):
       def setUp(self):
           self.drivers = {...}  # Initialize test drivers
           
       def test_person_extraction(self):
           results = run_suite_from_spec(person_test_spec, self.drivers)
           self.assertGreaterEqual(results["main"]["accuracy"], 0.85)

Best Practices
--------------

1. **Comprehensive Test Coverage**: Create tests covering various input formats and edge cases
2. **Multiple Provider Testing**: Test against multiple LLM providers to ensure robustness
3. **Regular Regression Testing**: Run test suites regularly to catch performance degradation
4. **Baseline Establishment**: Establish accuracy baselines for different test scenarios
5. **Performance Monitoring**: Track token usage and costs over time
6. **Error Analysis**: Analyze failed tests to improve field definitions and prompts
7. **Incremental Testing**: Start with simple tests and gradually increase complexity

Error Handling
--------------

The runner handles various error conditions gracefully:

**Driver Errors:**

- Connection failures
- Authentication issues
- Rate limiting
- Model unavailability

**Extraction Errors:**

- Invalid JSON responses
- Schema validation failures
- Timeout errors
- Unexpected response formats

**Test Specification Errors:**

- Malformed test specifications
- Missing required fields
- Invalid schema definitions

All errors are logged and reported in the test results without stopping the entire suite execution.