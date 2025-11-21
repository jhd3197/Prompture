"""Example demonstrating TOON input conversion for token savings with product data."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompture import extract_from_data, extract_from_pandas

# Sample product data - this represents the kind of uniform arrays where TOON shines
PRODUCT_DATA = [
    {
        "id": 1,
        "name": "Cacao Dark Chocolate Bar",
        "price": 12.5,
        "rating": 4.7,
        "category": "chocolate",
        "in_stock": True,
        "supplier": "Artisan Cocoa Co",
        "launch_date": "2024-01-15"
    },
    {
        "id": 2,
        "name": "Coconut Energy Squares",
        "price": 6.0,
        "rating": 4.3,
        "category": "snack",
        "in_stock": True,
        "supplier": "Healthy Bites Inc",
        "launch_date": "2024-02-01"
    },
    {
        "id": 3,
        "name": "Coffee Protein Cookies",
        "price": 4.2,
        "rating": 4.1,
        "category": "baked",
        "in_stock": False,
        "supplier": "Morning Fuel LLC",
        "launch_date": "2024-01-20"
    },
    {
        "id": 4,
        "name": "Organic Almond Butter",
        "price": 15.99,
        "rating": 4.8,
        "category": "spreads",
        "in_stock": True,
        "supplier": "Pure Nut Company",
        "launch_date": "2023-12-10"
    },
    {
        "id": 5,
        "name": "Green Tea Matcha Powder",
        "price": 24.99,
        "rating": 4.6,
        "category": "beverages",
        "in_stock": True,
        "supplier": "Zen Leaf Products",
        "launch_date": "2024-01-05"
    }
]

# Analysis schema - what we want to extract
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "total_products": {"type": "integer", "description": "Total number of products"},
        "average_price": {"type": "number", "description": "Average price across all products"},
        "highest_rated_product": {"type": "string", "description": "Name of highest rated product"},
        "out_of_stock_count": {"type": "integer", "description": "Number of out of stock items"},
        "price_range": {
            "type": "object",
            "properties": {
                "min": {"type": "number"},
                "max": {"type": "number"}
            }
        },
        "categories": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of unique categories"
        },
        "premium_products": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of products over $20"
        }
    },
    "required": ["total_products", "average_price", "highest_rated_product"]
}

# Default model to test (can be overridden with environment variable)
MODEL_TO_TEST = os.getenv(
    "PROMPTURE_TEST_MODEL",
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"
)

def demonstrate_extract_from_data():
    """Demonstrate extract_from_data with comprehensive product analysis."""
    print("\n" + "="*80)
    print("EXTRACT_FROM_DATA DEMONSTRATION")
    print("="*80)
    
    question = (
        "Analyze this product data comprehensively. Calculate statistics, "
        "identify patterns, and extract key insights about inventory, pricing, and ratings."
    )
    
    print(f"Question: {question}")
    print(f"Number of products: {len(PRODUCT_DATA)}")
    print(f"Model: {MODEL_TO_TEST}")
    
    try:
        result = extract_from_data(
            data=PRODUCT_DATA,
            question=question,
            json_schema=ANALYSIS_SCHEMA,
            model_name=MODEL_TO_TEST
        )
        
        print(f"\n✓ Extraction successful!")
        
        # Show token savings
        savings = result["token_savings"]
        print(f"\nToken Savings Analysis:")
        print(f"  JSON format: ~{savings['estimated_json_tokens']} tokens ({savings['json_characters']} chars)")
        print(f"  TOON format: ~{savings['estimated_toon_tokens']} tokens ({savings['toon_characters']} chars)")
        print(f"  Estimated savings: ~{savings['estimated_saved_tokens']} tokens ({savings['percentage_saved']}%)")
        
        # Show the TOON data that was sent
        print(f"\nTOON Data sent to LLM:")
        print("-" * 40)
        print(result["toon_data"])
        print("-" * 40)
        
        # Show extracted results
        print(f"\nExtracted Analysis:")
        analysis = result["json_object"]
        print(f"  Total Products: {analysis.get('total_products', 'N/A')}")
        print(f"  Average Price: ${analysis.get('average_price', 0):.2f}")
        print(f"  Highest Rated: {analysis.get('highest_rated_product', 'N/A')}")
        print(f"  Out of Stock: {analysis.get('out_of_stock_count', 0)}")
        
        price_range = analysis.get('price_range', {})
        if price_range:
            print(f"  Price Range: ${price_range.get('min', 0):.2f} - ${price_range.get('max', 0):.2f}")
        
        categories = analysis.get('categories', [])
        if categories:
            print(f"  Categories: {', '.join(categories)}")
            
        premium = analysis.get('premium_products', [])
        if premium:
            print(f"  Premium Products (>$20): {', '.join(premium)}")
        
        # Show usage info
        usage = result["usage"]
        print(f"\nUsage Statistics:")
        print(f"  Prompt tokens: {usage.get('prompt_tokens', 0)}")
        print(f"  Completion tokens: {usage.get('completion_tokens', 0)}")
        print(f"  Total tokens: {usage.get('total_tokens', 0)}")
        print(f"  Cost: ${usage.get('cost', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Extraction failed: {e}")
        return False

def demonstrate_extract_from_pandas():
    """Demonstrate extract_from_pandas with the same data."""
    try:
        import pandas as pd
    except ImportError:
        print("\n⚠ Pandas not installed, skipping DataFrame demonstration")
        print("  Install with: pip install pandas or pip install prompture[pandas]")
        return False
    
    print("\n" + "="*80)
    print("EXTRACT_FROM_PANDAS DEMONSTRATION")
    print("="*80)
    
    # Convert to DataFrame
    df = pd.DataFrame(PRODUCT_DATA)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    
    question = (
        "Focus on pricing analysis: What's the correlation between price and rating? "
        "Which categories have the highest average prices? Are there any pricing outliers?"
    )
    
    pricing_schema = {
        "type": "object",
        "properties": {
            "price_rating_correlation": {"type": "string", "description": "Correlation between price and rating"},
            "highest_priced_category": {"type": "string", "description": "Category with highest average price"},
            "lowest_priced_category": {"type": "string", "description": "Category with lowest average price"},
            "pricing_outliers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Products that are unusually expensive or cheap for their category"
            },
            "value_recommendations": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "Products offering best value (high rating, reasonable price)"
            }
        },
        "required": ["price_rating_correlation", "highest_priced_category"]
    }
    
    print(f"Question: {question}")
    
    try:
        result = extract_from_pandas(
            df=df,
            question=question,
            json_schema=pricing_schema,
            model_name=MODEL_TO_TEST
        )
        
        print(f"\n✓ DataFrame extraction successful!")
        
        # Show token savings
        savings = result["token_savings"]
        print(f"\nToken Savings Analysis:")
        print(f"  JSON format: ~{savings['estimated_json_tokens']} tokens ({savings['json_characters']} chars)")
        print(f"  TOON format: ~{savings['estimated_toon_tokens']} tokens ({savings['toon_characters']} chars)")
        print(f"  Estimated savings: ~{savings['estimated_saved_tokens']} tokens ({savings['percentage_saved']}%)")
        
        # Show DataFrame info
        df_info = result["dataframe_info"]
        print(f"\nDataFrame Info:")
        print(f"  Shape: {df_info['shape']}")
        print(f"  Data types: {df_info['dtypes']}")
        
        # Show pricing analysis results
        analysis = result["json_object"]
        print(f"\nPricing Analysis Results:")
        print(f"  Price-Rating Correlation: {analysis.get('price_rating_correlation', 'N/A')}")
        print(f"  Highest Priced Category: {analysis.get('highest_priced_category', 'N/A')}")
        print(f"  Lowest Priced Category: {analysis.get('lowest_priced_category', 'N/A')}")
        
        outliers = analysis.get('pricing_outliers', [])
        if outliers:
            print(f"  Pricing Outliers: {', '.join(outliers)}")
            
        recommendations = analysis.get('value_recommendations', [])
        if recommendations:
            print(f"  Value Recommendations: {', '.join(recommendations)}")
            
        return True
        
    except Exception as e:
        print(f"\n✗ DataFrame extraction failed: {e}")
        return False

def demonstrate_data_with_key():
    """Demonstrate extract_from_data with data nested under a key."""
    print("\n" + "="*80)
    print("NESTED DATA DEMONSTRATION")
    print("="*80)
    
    # Wrap data in a structure like an API response
    nested_data = {
        "products": PRODUCT_DATA,
        "total_count": len(PRODUCT_DATA),
        "page": 1,
        "per_page": 50
    }
    
    question = "What are the top 2 most expensive products and their suppliers?"
    
    simple_schema = {
        "type": "object",
        "properties": {
            "top_expensive_products": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                        "supplier": {"type": "string"}
                    }
                }
            }
        },
        "required": ["top_expensive_products"]
    }
    
    print(f"Data structure: API response with 'products' key")
    print(f"Question: {question}")
    
    try:
        result = extract_from_data(
            data=nested_data,
            data_key="products",  # Specify which key contains the array
            question=question,
            json_schema=simple_schema,
            model_name=MODEL_TO_TEST
        )
        
        print(f"\n✓ Nested data extraction successful!")
        
        savings = result["token_savings"]
        print(f"  Token savings: {savings['percentage_saved']}% (~{savings['estimated_saved_tokens']} tokens)")
        
        products = result["json_object"].get("top_expensive_products", [])
        print(f"\nTop Expensive Products:")
        for i, product in enumerate(products, 1):
            print(f"  {i}. {product.get('name', 'N/A')} - ${product.get('price', 0):.2f} ({product.get('supplier', 'N/A')})")
            
        return True
        
    except Exception as e:
        print(f"\n✗ Nested data extraction failed: {e}")
        return False

def main():
    """Run all demonstrations."""
    print("TOON INPUT CONVERSION EXAMPLES")
    print("Demonstrating token savings with structured data")
    print(f"Using model: {MODEL_TO_TEST}")
    
    results = []
    
    # Test basic extract_from_data
    results.append(("Extract from Data", demonstrate_extract_from_data()))
    
    # Test extract_from_pandas (if pandas available)
    results.append(("Extract from Pandas", demonstrate_extract_from_pandas()))
    
    # Test nested data
    results.append(("Nested Data", demonstrate_data_with_key()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    successful_tests = sum(1 for _, success in results if success)
    print(f"\nResults: {successful_tests}/{len(results)} tests successful")
    
    if successful_tests > 0:
        print("\n🎉 TOON input conversion is working!")
        print("   Your structured data is being converted to TOON format,")
        print("   saving tokens on input while getting JSON responses.")
        print(f"\n   Key benefits demonstrated:")
        print(f"   • Significant token reduction (typically 45-60% for uniform arrays)")
        print(f"   • Automatic conversion of JSON arrays and DataFrames")
        print(f"   • Token usage tracking and savings analysis")
        print(f"   • Flexible question-answering over structured data")
    else:
        print("\n❌ No tests passed. Check your model configuration and dependencies.")

if __name__ == "__main__":
    main()