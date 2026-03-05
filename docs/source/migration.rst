Migration Guide
===============

This guide covers common upgrade scenarios when moving between Prompture
versions.

Response Shape
--------------

All extraction functions return a dict with these **stable** keys:

.. code-block:: python

   result = extract_and_jsonify(...)
   result["json_string"]   # raw JSON string
   result["json_object"]   # parsed dict/list
   result["usage"]         # token counts and cost

New keys may be added in minor versions. Always use ``.get()`` for optional
keys to stay forward-compatible:

.. code-block:: python

   # Safe — works before and after the strategy field was added
   strategy_used = result["usage"].get("strategy", "unknown")

Strategy Parameter
------------------

The ``strategy`` parameter was added to ``ask_for_json``,
``extract_and_jsonify``, and ``extract_with_model`` to give explicit control
over how structured output is obtained.

**Before (still works):**

.. code-block:: python

   result = extract_with_model(MyModel, text, model_name="openai/gpt-4o")

**After (opt-in to specific strategy):**

.. code-block:: python

   from prompture.extraction.strategy import StructuredOutputStrategy

   result = extract_with_model(
       MyModel, text,
       model_name="openai/gpt-4o",
       strategy=StructuredOutputStrategy.TOOL_CALL,
   )

The default is ``"auto"``, which picks the best available strategy for each
provider. Existing code that doesn't pass ``strategy`` continues to work
unchanged.

Cost and Pricing
----------------

``usage["cost"]`` is calculated from two sources:

1. **models.dev live cache** -- queried at runtime for up-to-date pricing.
2. **Local rate files** in ``prompture/infra/rates/`` -- fallback when
   models.dev is unavailable or the model isn't listed.

If costs seem wrong:

- Check that your models.dev cache is fresh (it auto-refreshes periodically).
- For custom or private models, pricing data may not exist -- the cost will
  be reported as ``0.0``.
- File a `pricing mismatch issue
  <https://github.com/jhd3197/prompture/issues/new?template=pricing_cost_mismatch.yml>`_
  with the expected and actual values.

Driver Interface
----------------

The ``Driver`` base class provides a stable ``generate(prompt, options)``
contract. If you've written a custom driver:

- **New capability flags** (e.g., ``supports_json_schema``) may be added to
  the base class with ``False`` defaults. Your driver will inherit the default
  automatically -- no code change needed.
- **New optional parameters** may be added to ``generate()``. Custom drivers
  that accept ``**kwargs`` are forward-compatible.

.. code-block:: python

   class MyDriver(Driver):
       def generate(self, prompt, options=None, **kwargs):
           # **kwargs absorbs new parameters added in future versions
           ...

Import Path Changes
-------------------

Some modules were relocated for better organization. Old import paths still
work via re-export shims but will emit ``DeprecationWarning``:

.. list-table::
   :header-rows: 1
   :widths: 45 45 10

   * - Old Path
     - New Path
     - Status
   * - ``prompture.bridges.tukuy_backend``
     - ``prompture.infra.tukuy_backend``
     - Shimmed
   * - ``prompture.integrations.tukuy_bridge``
     - ``prompture.extraction.tukuy_bridge``
     - Shimmed

To migrate, update your imports to the new paths. The old paths will be
removed in a future major version.
