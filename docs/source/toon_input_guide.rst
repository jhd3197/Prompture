TOON Input Conversion Guide
===========================

Prompture now supports **TOON input conversion** for structured data, allowing you to achieve significant token savings (typically 45-60%) when analyzing JSON arrays or Pandas DataFrames with LLMs.

Overview
--------

TOON (Tabular Object Oriented Notation) is a compact format for representing uniform data structures. When you have structured data like product catalogs, user lists, or transaction records, converting to TOON format before sending to the LLM can dramatically reduce token usage while maintaining the same analytical capabilities.

**Key Benefits:**

* **45-60% token reduction** for uniform data arrays
* **Automatic conversion** from JSON/DataFrames to TOON
* **JSON responses** for easy consumption
* **Token usage tracking** with savings analysis
* **No changes to existing code** - new functions complement existing API

Quick Start
-----------

Installation
~~~~~~~~~~~~

TOON functionality is included by default when you install Prompture:

.. code-block:: bash

   pip install prompture

Both ``python-toon`` and ``pandas`` are now installed automatically - no extra steps needed!

Basic Usage
~~~~~~~~~~~

Analyze JSON Array Data
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from prompture import extract_from_data

   # Your structured data
   products = [
       {"id": 1, "name": "Laptop", "price": 999.99, "rating": 4.5},
       {"id": 2, "name": "Book", "price": 19.99, "rating": 4.2},
       {"id": 3, "name": "Headphones", "price": 149.99, "rating": 4.7}
   ]

   # Define what you want to extract
   schema = {
       "type": "object",
       "properties": {
           "average_price": {"type": "number"},
           "highest_rated": {"type": "string"},
           "total_items": {"type": "integer"}
       }
   }

   # Ask questions about your data
   result = extract_from_data(
       data=products,
       question="What is the average price, highest rated product, and total count?",
       json_schema=schema,
       model_name="openai/gpt-4"
   )

   print(result["json_object"])
   # {"average_price": 389.96, "highest_rated": "Headphones", "total_items": 3}

   # Check token savings
   savings = result["token_savings"]
   print(f"Token savings: {savings['percentage_saved']}%")
   # Token savings: 62.3%

Analyze Pandas DataFrames
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from prompture import extract_from_pandas
   import pandas as pd

   # Load your DataFrame
   df = pd.read_csv("sales_data.csv")
   # or create from existing data
   df = pd.DataFrame(products)

   schema = {
       "type": "object",
       "properties": {
           "top_category": {"type": "string"},
           "price_trend": {"type": "string"},
           "outliers": {"type": "array", "items": {"type": "string"}}
       }
   }

   result = extract_from_pandas(
       df=df,
       question="What category has the highest average price? Any pricing outliers?",
       json_schema=schema,
       model_name="openai/gpt-4"
   )

   print(result["json_object"])
   print(f"DataFrame shape: {result['dataframe_info']['shape']}")

Advanced Usage
--------------

Working with Nested Data
~~~~~~~~~~~~~~~~~~~~~~~~~

When your data is nested in an API response or larger structure:

.. code-block:: python

   api_response = {
       "status": "success",
       "page": 1,
       "results": [
           {"user_id": 101, "score": 85, "level": "advanced"},
           {"user_id": 102, "score": 72, "level": "intermediate"}
       ]
   }

   result = extract_from_data(
       data=api_response,
       data_key="results",  # Specify which key contains the array
       question="What is the average score?",
       json_schema={"type": "object", "properties": {"avg_score": {"type": "number"}}},
       model_name="openai/gpt-4"
   )

Custom Instructions and Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = extract_from_data(
       data=products,
       question="Find products under $100",
       json_schema=schema,
       model_name="openai/gpt-4",
       instruction_template="Analyze this product data carefully and answer: {question}",
       ai_cleanup=True,
       options={"temperature": 0.1, "max_tokens": 500}
   )

Token Savings Analysis
----------------------

Understanding the Efficiency Gains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functions provide detailed token usage analysis:

.. code-block:: python

   result = extract_from_data(data=products, question=question, json_schema=schema, model_name=model)
   
   # Token savings breakdown
   savings = result["token_savings"]
   print(f"JSON characters: {savings['json_characters']}")
   print(f"TOON characters: {savings['toon_characters']}")
   print(f"Character reduction: {savings['saved_characters']} ({savings['percentage_saved']}%)")
   print(f"Estimated token savings: ~{savings['estimated_saved_tokens']} tokens")

   # The TOON data that was sent to the LLM
   print("TOON format used:")
   print(result["toon_data"])

Preview Savings Without LLM Calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the token comparison utility to analyze your data structure efficiency:

.. code-block:: bash

   python examples/token_comparison_utility.py

Or programmatically:

.. code-block:: python

   from examples.token_comparison_utility import compare_formats, print_comparison_report
   
   stats = compare_formats(your_data)
   print_comparison_report(stats)

Best Practices
--------------

When TOON is Most Effective
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**✅ Ideal for TOON:**

* **Uniform data structures** (all objects have same keys)
* **Tabular data** from databases, CSVs, APIs
* **Product catalogs, user lists, transaction records**
* **Arrays with 3+ objects**

**⚠️ Less effective:**

* Non-uniform objects (different key sets)
* Deeply nested structures
* Very small arrays (1-2 items)
* Already compact data

Data Structure Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # ✅ Perfect for TOON - uniform structure
   good_data = [
       {"id": 1, "name": "A", "price": 10.0},
       {"id": 2, "name": "B", "price": 20.0},
       {"id": 3, "name": "C", "price": 30.0}
   ]

   # ⚠️ Less efficient - non-uniform structure  
   mixed_data = [
       {"id": 1, "name": "A", "price": 10.0},
       {"id": 2, "title": "B", "cost": 20.0, "extra": "data"},
       {"user": 3, "label": "C"}
   ]

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Large datasets**: TOON conversion is fast, but consider chunking very large DataFrames
* **Model compatibility**: All models work with TOON input (it's converted automatically)
* **Cost optimization**: Higher token savings = lower API costs

Error Handling
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       result = extract_from_data(data=data, question=question, json_schema=schema, model_name=model)
   except ValueError as e:
       if "empty" in str(e):
           print("Data array is empty")
       elif "dictionaries" in str(e):
           print("All array items must be dictionaries")
       elif "python-toon" in str(e):
           print("Install python-toon: pip install python-toon")
   except RuntimeError as e:
       if "pandas" in str(e):
           print("Install pandas: pip install pandas")

Migration from Existing Code
----------------------------

Easy Upgrade Path
~~~~~~~~~~~~~~~~~

Replace existing extraction calls to get automatic token savings:

.. code-block:: python

   # Before: Using extract_and_jsonify with JSON text
   json_text = json.dumps(products)
   result = extract_and_jsonify(
       text=f"Analyze this data: {json_text}",
       json_schema=schema,
       model_name=model_name
   )

   # After: Using extract_from_data with automatic TOON conversion
   result = extract_from_data(
       data=products,  # Pass data directly
       question="Analyze this data comprehensively",
       json_schema=schema,
       model_name=model_name
   )
   # Automatic 45-60% token savings!

API Reference
-------------

extract_from_data()
~~~~~~~~~~~~~~~~~~~

.. autofunction:: prompture.extract_from_data

extract_from_pandas()
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: prompture.extract_from_pandas

Examples and Use Cases
----------------------

E-commerce Analysis
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   products = load_product_catalog()
   
   result = extract_from_data(
       data=products,
       question="Which products are underperforming? Consider price, rating, and sales.",
       json_schema={
           "type": "object",
           "properties": {
               "underperforming_products": {"type": "array", "items": {"type": "string"}},
               "recommended_actions": {"type": "array", "items": {"type": "string"}},
               "price_optimization": {"type": "object"}
           }
       },
       model_name="openai/gpt-4"
   )

Financial Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   transactions_df = pd.read_csv("transactions.csv")
   
   result = extract_from_pandas(
       df=transactions_df,
       question="Identify spending patterns and categorize expenses by priority",
       json_schema=expense_analysis_schema,
       model_name="anthropic/claude-3-sonnet"
   )

User Behavior Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   user_activity = fetch_user_data_from_api()["users"]
   
   result = extract_from_data(
       data=user_activity,
       question="Segment users by engagement level and recommend retention strategies",
       json_schema=user_segmentation_schema,
       model_name="openai/gpt-4"
   )

Troubleshooting
---------------

Installation Issues
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # If TOON conversion fails
   pip install --upgrade python-toon
   
   # If pandas functions fail
   pip install --upgrade pandas
   
   # Or install with optional dependencies
   pip install prompture[pandas]

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large DataFrames, consider sampling
   if len(df) > 1000:
       sample_df = df.sample(n=500, random_state=42)
       result = extract_from_pandas(df=sample_df, ...)
   
   # Or chunk processing
   chunk_size = 100
   results = []
   for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
       result = extract_from_pandas(df=chunk, ...)
       results.append(result)

Conclusion
----------

TOON input conversion represents a significant advancement in LLM efficiency for structured data analysis. By automatically converting your JSON arrays and DataFrames to TOON format, you can achieve substantial token savings while maintaining full analytical capabilities.

**Key takeaways:**

* **45-60% token reduction** for uniform data structures
* **Drop-in replacement** for existing extraction workflows  
* **Comprehensive analysis capabilities** with automatic optimization
* **Cost reduction** through more efficient token usage

Try the token comparison utility to see how much you can save with your own data!