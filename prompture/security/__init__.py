"""Input-side safety helpers for Prompture: prompt-injection detection
and PII redaction.

These are the symmetric counterparts to :mod:`prompture.refusal`
(output-side classification).  Together they give you a complete
input → LLM → output safety story:

* :class:`PromptInjectionDetector` flags untrusted user input for
  jailbreak attempts, instruction-override patterns, role-hijack
  templates, and known prompt-extraction lures.
* :class:`PIIRedactor` scrubs personal data (emails, phone numbers,
  credit cards with Luhn-check, IBANs, IP addresses, common
  ID-number shapes) from prompts before they leave your process
  and from responses before they reach the user.

Both modules are clean-room MIT implementations with zero new
dependencies.  Pair them with :class:`~prompture.RefusalDetector` to
classify outgoing responses.

Quick start::

    from prompture.security import PromptInjectionDetector, PIIRedactor

    detector = PromptInjectionDetector()
    if detector.is_injection(user_input):
        return "Sorry, that prompt looks like an injection attempt."

    redactor = PIIRedactor()
    safe_prompt = redactor.redact(user_input).text
    response = agent.run(safe_prompt)
"""

from .injection import (
    InjectionCategory,
    InjectionResult,
    PromptInjectionDetector,
    is_prompt_injection,
)
from .redaction import (
    PIICategory,
    PIIMatch,
    PIIRedactor,
    RedactionResult,
    redact_pii,
)

__all__ = [
    "InjectionCategory",
    "InjectionResult",
    "PIICategory",
    "PIIMatch",
    "PIIRedactor",
    "PromptInjectionDetector",
    "RedactionResult",
    "is_prompt_injection",
    "redact_pii",
]
