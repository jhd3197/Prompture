"""
E-commerce Product Extraction Example

This example shows how to use Prompture for extracting product information from e-commerce descriptions.
It demonstrates extraction of product features, pricing, brand information, and categorization.
"""

from pydantic import BaseModel
from typing import List, Optional
from prompture import register_field, field_from_registry, extract_with_model

# E-commerce specific fields
register_field("product_features", {
    "type": "list",
    "description": "Key product features and specifications",
    "instructions": "Extract main features, specs, and selling points",
    "default": [],
    "nullable": True
})

register_field("price", {
    "type": "float",
    "description": "Product price in decimal format",
    "instructions": "Extract price as number, remove currency symbols",
    "default": 0.0,
    "nullable": True
})

register_field("brand", {
    "type": "str",
    "description": "Product brand or manufacturer",
    "instructions": "Extract brand name or manufacturer",
    "default": "Unknown",
    "nullable": True
})

register_field("description", {
    "type": "str",
    "description": "Product description text",
    "instructions": "Extract the main product description",
    "default": "",
    "nullable": True
})

register_field("category", {
    "type": "str",
    "description": "Product category classification",
    "instructions": "Extract product category or classify appropriately",
    "default": "General",
    "nullable": True
})

class Product(BaseModel):
    name: str = field_from_registry("name")
    brand: Optional[str] = field_from_registry("brand")
    price: Optional[float] = field_from_registry("price")
    description: Optional[str] = field_from_registry("description")
    features: Optional[List[str]] = field_from_registry("product_features")
    category: Optional[str] = field_from_registry("category")

# Sample product description
product_text = """
Apple MacBook Pro 14-inch (2023) - $1,999.00

The new MacBook Pro delivers exceptional performance with the M2 Pro chip.

KEY FEATURES:
- M2 Pro chip with 10-core CPU and 16-core GPU
- 14-inch Liquid Retina XDR display
- 16GB unified memory
- 512GB SSD storage
- Up to 18 hours battery life
- Three Thunderbolt 4 ports
- MagSafe 3 charging port

Perfect for professional video editing, software development, and creative work.
Category: Laptops & Computers
"""

# Extract product information
product = extract_with_model(
    Product,
    product_text,
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"
)

print(f"Product: {product.model.name}")
print(f"Brand: {product.model.brand}")
print(f"Price: ${product.model.price}")
print(f"Features: {product.model.features}")