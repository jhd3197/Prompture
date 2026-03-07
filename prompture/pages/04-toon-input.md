# TOON Input Conversion

Prompture supports **TOON input conversion** for structured data, allowing 45-60% token savings when analyzing JSON arrays or Pandas DataFrames with LLMs.

## Overview

TOON (Tabular Object Oriented Notation) is a compact format for representing uniform data structures. When you have structured data like product catalogs, user lists, or transaction records, converting to TOON format before sending to the LLM dramatically reduces token usage.

**Key Benefits:**

- **45-60% token reduction** for uniform data arrays
- **Automatic conversion** from JSON/DataFrames to TOON
- **JSON responses** for easy consumption
- **Token usage tracking** with savings analysis

## Basic Usage

### Analyze JSON Array Data

```python
from prompture import extract_from_data

products = [
    {"id": 1, "name": "Laptop", "price": 999.99, "rating": 4.5},
    {"id": 2, "name": "Book", "price": 19.99, "rating": 4.2},
    {"id": 3, "name": "Headphones", "price": 149.99, "rating": 4.7}
]

schema = {
    "type": "object",
    "properties": {
        "average_price": {"type": "number"},
        "highest_rated": {"type": "string"},
        "total_items": {"type": "integer"}
    }
}

result = extract_from_data(
    data=products,
    question="What is the average price, highest rated product, and total count?",
    json_schema=schema,
    model_name="openai/gpt-4"
)

print(result["json_object"])
print(f"Token savings: {result['token_savings']['percentage_saved']}%")
```

### Analyze Pandas DataFrames

```python
from prompture import extract_from_pandas
import pandas as pd

df = pd.DataFrame(products)

result = extract_from_pandas(
    df=df,
    question="What category has the highest average price?",
    json_schema=schema,
    model_name="openai/gpt-4"
)
```

## When TOON is Most Effective

**Ideal for:**
- Uniform data structures (all objects have same keys)
- Tabular data from databases, CSVs, APIs
- Arrays with 3+ objects

**Less effective for:**
- Non-uniform objects (different key sets)
- Deeply nested structures
- Very small arrays (1-2 items)
