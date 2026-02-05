# Prompture Feedback & Suggestions

Based on integrating Prompture into CachiBot, here are findings and suggestions for improvement.

## Bug Report: Usage Tracking Inconsistency with Tools

### Issue Summary
When tools are registered with an Agent, `result.run_usage` returns empty/zero values while `result.usage` contains the actual usage data.

### Reproduction Steps

```python
from prompture import Agent, ToolRegistry

# Test 1: Agent WITHOUT tools - WORKS
agent1 = Agent(model='moonshot/kimi-k2.5', system_prompt='You are helpful.')
result1 = agent1.run('Say hello')
print(result1.run_usage)
# Output: {'prompt_tokens': 17, 'completion_tokens': 85, 'total_tokens': 102, ...}  ✅

# Test 2: Agent WITH tools - BROKEN
registry = ToolRegistry()

@registry.register
def my_tool(arg: str) -> str:
    """A simple tool."""
    return f"Result: {arg}"

agent2 = Agent(
    model='moonshot/kimi-k2.5',
    system_prompt='You are helpful.',
    tools=registry,
)
result2 = agent2.run('Say hello')
print(result2.run_usage)
# Output: {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'call_count': 0, ...}  ❌

print(result2.usage)
# Output: {'prompt_tokens': 61, 'completion_tokens': 110, 'total_tokens': 171, 'cost': 0.000367}  ✅
```

### Expected Behavior
Both `result.run_usage` and `result.usage` should return consistent, accurate usage data regardless of whether tools are registered.

### Current Workaround
```python
# Use .usage as primary source, fall back to .run_usage
usage = result.usage or result.run_usage
```

### Impact
- Applications relying on `run_usage` for billing/monitoring get zero values when tools are used
- Confusing API surface with two similar attributes that behave differently

---

## Suggestions for Improvement

### 1. Consolidate Usage Attributes
Consider deprecating one of `run_usage` / `usage` and having a single source of truth:

```python
# Instead of two attributes:
result.run_usage  # Session-level aggregated usage
result.usage      # Per-call usage

# Consider a unified approach:
result.usage.total        # Total for this run
result.usage.per_call     # List of per-call breakdowns
result.usage.per_model    # Breakdown by model
```

### 2. Add Type Hints for Usage Dict
The usage dict structure isn't immediately clear. Consider a typed dataclass:

```python
@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    call_count: int
    per_model: dict[str, ModelUsage]

    @property
    def total_cost(self) -> float:
        """Alias for cost (backwards compatibility)."""
        return self.cost
```

### 3. Document the Difference Between `usage` and `run_usage`
If both are intentional, document when to use which:
- `usage`: Use for X scenario
- `run_usage`: Use for Y scenario

### 4. Cost Field Naming Consistency
Currently:
- `result.usage` has `cost`
- `result.run_usage` has `total_cost`

This inconsistency requires defensive coding:
```python
cost = usage.get("cost", 0.0) or usage.get("total_cost", 0.0)
```

### 5. Consider a `last_result` Attribute on Agent
Currently, there's no way to access the last result from the Agent itself:

```python
agent = Agent(...)
agent.run("Hello")

# This doesn't work:
agent.last_result  # AttributeError

# Would be useful for:
# - Getting usage after the fact
# - Debugging
# - Building wrappers
```

### 6. Streaming Usage Events
For real-time dashboards, it would be helpful to have usage callbacks:

```python
callbacks = AgentCallbacks(
    on_thinking=...,
    on_tool_start=...,
    on_tool_end=...,
    on_usage=lambda usage: update_dashboard(usage),  # New!
)
```

### 7. Model-Specific Pricing Transparency
When `cost` is calculated, it would be helpful to know the rates used:

```python
result.usage = {
    'total_tokens': 171,
    'cost': 0.000367,
    'pricing': {
        'model': 'kimi-k2.5',
        'input_rate_per_1k': 0.0012,
        'output_rate_per_1k': 0.0018,
        'source': 'models.dev',  # Where the pricing came from
    }
}
```

---

## What Worked Well

1. **Clean Agent API** - Simple to create and run agents
2. **Tool Registration** - Decorator-based registration is intuitive
3. **AgentCallbacks** - Great for real-time UI updates
4. **Multi-provider Support** - Moonshot, OpenAI, Anthropic work seamlessly
5. **Automatic Cost Calculation** - Having built-in pricing is valuable

---

## Environment

- Prompture Version: 1.0.2
- Python Version: 3.11.7
- Platform: Windows 11
- Models Tested: moonshot/kimi-k2.5

---

## Summary

The main issue is the `run_usage` vs `usage` inconsistency when tools are registered. This caused hours of debugging in CachiBot until we discovered that `result.usage` (not `result.run_usage`) has the correct data.

A quick fix would be to ensure `run_usage` aggregates usage data even when tools are involved, or clearly document that `usage` should be preferred.

Thanks for building Prompture - it's a great library!
