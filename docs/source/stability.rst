API Stability
=============

Prompture follows `Semantic Versioning <https://semver.org/>`_ (SemVer).
This page describes which surfaces are stable and which are experimental,
so you can adopt Prompture with confidence.

Stability Tiers
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Surface
     - Stability
     - Notes
   * - ``extract_with_model``, ``stepwise_extract_with_model``
     - **Stable**
     - Pydantic-based extraction. Signature and return shape will not change without a deprecation cycle.
   * - ``ask_for_json``, ``extract_and_jsonify``
     - **Stable**
     - Schema-enforced JSON extraction.
   * - ``extract_from_data``, ``extract_from_pandas``
     - **Stable**
     - TOON input conversion + extraction.
   * - ``StructuredOutputStrategy``
     - **Stable**
     - Enum controlling how structured output is obtained (``provider_native``, ``tool_call``, ``prompted_repair``).
   * - ``ProviderCapabilities``, ``get_capabilities``
     - **Stable**
     - Capability registry for providers and models.
   * - ``Driver`` interface (``generate``, capability flags)
     - **Stable**
     - All drivers implement the same ``generate(prompt, options)`` contract.
   * - Response shape (``json_string``, ``json_object``, ``usage``)
     - **Stable**
     - New keys may be added; existing keys will not be removed or renamed.
   * - ``Conversation``, ``ToolRegistry``
     - **Stable**
     - Multi-turn sessions and tool registration.
   * - ``SkillPipeline``, agent abstractions
     - Experimental
     - May change between minor versions.
   * - ``groups/`` (consensus, debate)
     - Experimental
     - May change between minor versions.
   * - ``pipeline/`` (model routing)
     - Experimental
     - May change between minor versions.

What "Stable" Means
~~~~~~~~~~~~~~~~~~~~

- The function signature will not change in a backward-incompatible way
  within the same major version.
- New **optional** parameters or new keys in returned dicts may be added
  in minor releases.
- If a breaking change is unavoidable, it will go through the deprecation
  process described below.

What "Experimental" Means
~~~~~~~~~~~~~~~~~~~~~~~~~~

- The API may change between minor versions without a deprecation cycle.
- We will document changes in ``BREAKING_CHANGES.md`` regardless.

Deprecation Policy
------------------

When a stable API needs to change:

1. **Deprecation warning** — The old API continues to work but emits a
   ``DeprecationWarning`` with a message explaining the replacement.

2. **Minimum one minor version** — The deprecated API is kept for at
   least one minor release cycle (e.g., deprecated in 0.5.0, removed
   no earlier than 0.6.0).

3. **Breaking changes log** — Every removal or incompatible change is
   recorded in ``BREAKING_CHANGES.md`` at the repository root.

4. **Migration guidance** — Deprecation warnings and the breaking
   changes log include instructions for migrating to the new API.

Example:

.. code-block:: python

   import warnings

   def old_function():
       warnings.warn(
           "old_function() is deprecated, use new_function() instead. "
           "It will be removed in v0.6.0.",
           DeprecationWarning,
           stacklevel=2,
       )
       return new_function()

Provider Compatibility Matrix
------------------------------

The table below is auto-generated from the ``ProviderCapabilities``
registry.  It shows which structured-output strategies each provider
supports out of the box.

.. code-block:: python

   from prompture.infra.capabilities import get_compatibility_matrix
   print(get_compatibility_matrix())

.. include:: _generated/compatibility_matrix.rst
   :parser: rst

To regenerate this table, run:

.. code-block:: bash

   python -m prompture.infra.capabilities --matrix > docs/source/_generated/compatibility_matrix.rst
