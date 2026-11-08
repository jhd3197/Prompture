"""Actionable failures without treating a permission denial as model failure."""

from __future__ import annotations

import re


def classify_failure(message: str, *, permission: bool = False) -> dict[str, str]:
    """Return a stable category and a recovery hint; never grant permissions."""
    if permission or re.search(
        r"permission (?:was )?denied by the user|requires? (?:tool |command )?approval|"
        r"waiting for (?:user |your )?(?:approval|permission)|approval (?:required|denied)|"
        r"not allowed to use (?:the )?tool|permission prompt",
        message,
        re.I,
    ):
        return {
            "category": "permission",
            "hint": "Review the blocked tool in the native agent and grant only the intended permission there, then run prompture delegate-resume. Resume does not grant permission.",
        }
    if re.search(
        r"not inside a trusted directory|not a git repository|working directory.*(?:missing|not found)", message, re.I
    ):
        return {
            "category": "directory",
            "hint": "Check the project's cwd and the native agent's repository trust settings, then resume. Prompture will not disable repository checks.",
        }
    if re.search(
        r"model.*(?:requires a newer|not supported|unsupported|not found|does not exist)|unsupported model",
        message,
        re.I,
    ):
        return {
            "category": "model",
            "hint": "Update the selected agent CLI or choose a compatible model for a new run. A model change must not silently replace an existing session.",
        }
    if re.search(r"not logged in|authentication|invalid api key|unauthorized|login required", message, re.I):
        return {
            "category": "authentication",
            "hint": "Sign in using the selected agent's native login, then resume. Do not paste credentials into the task.",
        }
    if re.search(r"usage limit|limit reached|rate limit|quota|hit your limit", message, re.I):
        return {
            "category": "limit",
            "hint": "Wait for the usage window to reset; the queue can resume automatically when its reset time is known.",
        }
    return {
        "category": "task",
        "hint": "Inspect the step log, fix the reported task failure, then run prompture delegate-resume.",
    }
