"""
Financial Document Analysis Example

This example shows how to use Prompture for processing financial statements and reports.
It demonstrates extraction of financial metrics such as revenue, profit margins, and fiscal data.
"""

from typing import Optional

from pydantic import BaseModel

from prompture import extract_with_model, field_from_registry, register_field

# Financial fields
register_field(
    "revenue",
    {
        "type": "float",
        "description": "Total revenue or income amount",
        "instructions": "Extract revenue figures in millions, convert to number",
        "default": 0.0,
        "nullable": True,
    },
)

register_field(
    "profit_margin",
    {
        "type": "float",
        "description": "Profit margin as percentage",
        "instructions": "Extract profit margin as decimal (e.g., 15% = 0.15)",
        "default": 0.0,
        "nullable": True,
    },
)

register_field(
    "year",
    {
        "type": "int",
        "description": "Year value",
        "instructions": "Extract 4-digit year",
        "default": None,
        "nullable": True,
    },
)

register_field(
    "currency",
    {
        "type": "str",
        "description": "Currency code or symbol",
        "instructions": "Extract currency code (e.g., USD, EUR) or symbol",
        "default": "USD",
        "nullable": True,
    },
)


class FinancialSummary(BaseModel):
    company: str = field_from_registry("company")
    revenue: Optional[float] = field_from_registry("revenue")
    profit_margin: Optional[float] = field_from_registry("profit_margin")
    fiscal_year: Optional[int] = field_from_registry("year")
    currency: Optional[str] = field_from_registry("currency")


# Sample financial report
financial_text = """
TECHCORP INC. - FISCAL YEAR 2023 FINANCIAL SUMMARY

Revenue: $2.8 billion USD (up 12% from previous year)
Net Income: $420 million USD
Profit Margin: 15%
Fiscal Year Ending: December 31, 2023

Strong performance across all business segments with continued growth
in cloud services and enterprise software solutions.
"""

# Extract financial data
financial = extract_with_model(FinancialSummary, financial_text, "openai/gpt-4")

print(f"Company: {financial.model.company}")
print(f"Revenue: ${financial.model.revenue}B {financial.model.currency}")
print(f"Profit Margin: {financial.model.profit_margin * 100}%")
