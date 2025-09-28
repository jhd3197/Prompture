"""
Comparison example: stepwise_extract_with_model() vs manual_extract_and_jsonify()

This script demonstrates the differences between the two extraction methods:
1. stepwise_extract_with_model() - Makes separate API calls for each field
2. manual_extract_and_jsonify() - Makes a single API call for all fields

Both methods extract the same Person data and the comparison shows:
- Token usage patterns (multiple calls vs single call)  
- Extraction accuracy
- Cost analysis
- Performance characteristics
"""

from datetime import date
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from prompture import stepwise_extract_with_model, manual_extract_and_jsonify
from prompture.drivers import get_driver


# Define comprehensive Person model
class Person(BaseModel):
    """Comprehensive person model with various field types for testing."""
    
    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., ge=0, le=150, description="Age in years")
    birth_date: Optional[date] = Field(None, description="Date of birth in YYYY-MM-DD format")
    profession: str = Field(..., description="Current job or profession")
    is_employed: bool = Field(..., description="Whether the person is currently employed")
    salary: Optional[Decimal] = Field(None, ge=0, description="Annual salary in USD")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    city: Optional[str] = Field(None, description="City of residence")
    years_experience: Optional[int] = Field(None, ge=0, description="Years of work experience")
    
    @field_validator('birth_date', mode='before')
    @classmethod
    def parse_birth_date(cls, v):
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                return None
        return v
    
    @field_validator('salary', mode='before')
    @classmethod
    def parse_salary(cls, v):
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            try:
                # Remove currency symbols and commas
                clean_v = v.replace('$', '').replace(',', '').strip()
                return Decimal(clean_v)
            except:
                return None
        return v


# Detailed biographical text for extraction
SAMPLE_TEXT = """
Meet Dr. Sarah Elizabeth Johnson, a 34-year-old software engineering manager who has revolutionized 
the tech industry with her innovative approaches to AI development. Born on March 15, 1989, Sarah 
has built an impressive career spanning over 12 years in the technology sector.

Currently employed at a Fortune 500 tech company, Sarah oversees a team of 25 engineers working 
on cutting-edge artificial intelligence projects. Her expertise and leadership have earned her 
an annual salary of $185,000, reflecting her significant contributions to the field.

Sarah can be reached at sarah.johnson@techcorp.com or by phone at (555) 123-4567. She currently 
resides in San Francisco, California, where she enjoys the vibrant tech ecosystem and collaborates 
with other industry leaders.

With her extensive background in machine learning, natural language processing, and team management, 
Sarah represents the new generation of tech leaders who combine technical excellence with strong 
leadership skills. Her 12 years of experience have positioned her as a key decision-maker in 
her organization's strategic technology initiatives.

Sarah's career trajectory from junior developer to engineering manager demonstrates her commitment 
to both technical excellence and professional growth. She continues to be actively employed and 
is considered one of the rising stars in the AI development community.
"""

# JSON Schema for manual_extract_and_jsonify
PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Full name of the person"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150, "description": "Age in years"},
        "birth_date": {"type": "string", "format": "date", "description": "Date of birth in YYYY-MM-DD format"},
        "profession": {"type": "string", "description": "Current job or profession"},
        "is_employed": {"type": "boolean", "description": "Whether the person is currently employed"},
        "salary": {"type": "number", "minimum": 0, "description": "Annual salary in USD"},
        "email": {"type": "string", "format": "email", "description": "Email address"},
        "phone": {"type": "string", "description": "Phone number"},
        "city": {"type": "string", "description": "City of residence"},
        "years_experience": {"type": "integer", "minimum": 0, "description": "Years of work experience"}
    },
    "required": ["name", "age", "profession", "is_employed"]
}

# List of models to test - you can modify this based on your available models
MODELS_TO_TEST = [
    "gpt-oss:20b",
    "llama3.1:8b",
    "gemma3:latest",
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "mistral:latest",
    "magistral:latest"
]

def test_stepwise_method(model_name=None):
    """Test stepwise_extract_with_model method with optional model specification."""
    print(f"Testing stepwise_extract_with_model() with model: {model_name or 'default'}...")
    
    try:
        kwargs = {
            'model_cls': Person,
            'text': SAMPLE_TEXT,
        }
        if model_name:
            kwargs['model_name'] = model_name
            
        result = stepwise_extract_with_model(**kwargs)
        
        return {
            'success': True,
            'method': 'stepwise_extract_with_model',
            'model_name': model_name or 'default',
            'extracted_model': result['model'],
            'usage': result['usage'],
            'field_count': len(result['model'].__dict__),
            'api_calls': len(result['usage'].get('field_usages', {}))
        }
        
    except Exception as e:
        return {
            'success': False,
            'method': 'stepwise_extract_with_model',
            'model_name': model_name or 'default',
            'error': str(e),
            'extracted_model': None,
            'usage': {},
            'field_count': 0,
            'api_calls': 0
        }


def test_manual_method(model_name=None):
    """Test manual_extract_and_jsonify method with optional model specification."""
    print(f"Testing manual_extract_and_jsonify() with model: {model_name or 'default'}...")
    
    try:
        # Import both functions to see what's available
        from prompture import extract_and_jsonify
        driver = get_driver()
        
        # Try the manual function first
        try:
            kwargs = {
                'driver': driver,
                'text': SAMPLE_TEXT,
                'json_schema': PERSON_SCHEMA
            }
            if model_name:
                kwargs['model_name'] = model_name
                
            result = manual_extract_and_jsonify(**kwargs)
            method_used = "manual_extract_and_jsonify"
        except Exception as manual_error:
            print(f"manual_extract_and_jsonify failed: {manual_error}")
            kwargs = {
                'text': SAMPLE_TEXT,
                'json_schema': PERSON_SCHEMA
            }
            if model_name:
                kwargs['model_name'] = model_name
                
            result = extract_and_jsonify(**kwargs)
            method_used = "extract_and_jsonify (fallback)"
        
        # Convert to Person model for comparison
        person_model = Person(**result['json_object'])
        
        return {
            'success': True,
            'method': method_used,
            'model_name': model_name or 'default',
            'extracted_model': person_model,
            'usage': result.get('usage', {}),
            'field_count': len(result['json_object']),
            'api_calls': 1  # Single API call
        }
        
    except Exception as e:
        print(f"Manual extraction error: {e}")
        return {
            'success': False,
            'method': 'manual_extract_and_jsonify',
            'model_name': model_name or 'default',
            'error': str(e)[:50],  # Truncate error message
            'extracted_model': None,
            'usage': {},
            'field_count': 0,
            'api_calls': 0
        }


def validate_extraction_accuracy(result):
    """Validate that key fields were extracted correctly."""
    if not result['success'] or not result['extracted_model']:
        return False, "Extraction failed"
    
    person = result['extracted_model']
    errors = []
    
    # Validate required fields
    if not person.name or 'sarah' not in person.name.lower():
        errors.append("Name not correctly extracted")
    
    if not person.age or person.age != 34:
        errors.append(f"Age incorrect: expected 34, got {person.age}")
        
    if not person.profession or 'manager' not in person.profession.lower():
        errors.append("Profession not correctly extracted")
        
    if person.is_employed is not True:
        errors.append("Employment status incorrect")
    
    # Validate optional fields where we have clear expectations
    if person.salary and float(person.salary) != 185000.0:
        errors.append(f"Salary incorrect: expected 185000, got {person.salary}")
        
    if person.email and 'sarah.johnson@techcorp.com' not in person.email:
        errors.append("Email not correctly extracted")
        
    if person.city and 'san francisco' not in person.city.lower():
        errors.append("City not correctly extracted")
        
    if person.years_experience and person.years_experience != 12:
        errors.append(f"Experience incorrect: expected 12, got {person.years_experience}")
    
    return len(errors) == 0, errors


def print_comparison_table(results):
    """Print detailed comparison table showing model names in the results."""
    
    print("\n" + "="*160)
    print("STEPWISE VS MANUAL EXTRACTION COMPARISON REPORT")
    print("="*160)
    
    # Headers with Model column
    row_format = "{:<25} {:<15} {:<8} {:<8} {:<12} {:<8} {:<10} {:<8} {:<12} {:<15} {:<8} {:<10} {:<15}"
    headers = [
        "Method", "Model", "Success", "Prompt", "Completion", "Total", "Cost ($)", "Calls",
        "Fields", "Validation", "Name ✓", "Age ✓", "Error"
    ]
    print(row_format.format(*headers))
    print("-" * 160)
    
    for result in results:
        if result['success']:
            usage = result['usage']
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            cost = usage.get('cost', 0.0)
            
            # Validate accuracy
            is_valid, validation_errors = validate_extraction_accuracy(result)
            validation = "✓" if is_valid else "✗"
            
            # Check specific fields
            person = result['extracted_model']
            name_check = "✓" if person.name and 'sarah' in person.name.lower() else "✗"
            age_check = "✓" if person.age == 34 else "✗"
            
            error = ""
            
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cost = 0.0
            validation = "✗"
            name_check = "✗"
            age_check = "✗"
            error = result.get('error', 'Unknown error')[:15]
        
        print(row_format.format(
            result['method'][:25],
            result['model_name'][:15],
            "True" if result['success'] else "False",
            str(prompt_tokens)[:8],
            str(completion_tokens)[:12],
            str(total_tokens)[:8],
            f"{cost:.4f}"[:10],
            str(result['api_calls'])[:8],
            str(result['field_count'])[:12],
            validation[:15],
            name_check[:8],
            age_check[:10],
            error[:15]
        ))
    
    print("\n" + "="*160)
    print("DETAILED EXTRACTION RESULTS")
    print("="*160)
    
    for result in results:
        if result['success']:
            print(f"\n{result['method'].upper()} RESULTS (Model: {result['model_name']}):")
            person = result['extracted_model']
            print(f"  Name: {person.name}")
            print(f"  Age: {person.age}")
            print(f"  Birth Date: {person.birth_date}")
            print(f"  Profession: {person.profession}")
            print(f"  Is Employed: {person.is_employed}")
            print(f"  Salary: ${person.salary}" if person.salary else "  Salary: None")
            print(f"  Email: {person.email}")
            print(f"  Phone: {person.phone}")
            print(f"  City: {person.city}")
            print(f"  Years Experience: {person.years_experience}")
            
            # Usage details
            usage = result['usage']
            print(f"\n  USAGE STATS:")
            print(f"    Prompt tokens: {usage.get('prompt_tokens', 0)}")
            print(f"    Completion tokens: {usage.get('completion_tokens', 0)}")
            print(f"    Total tokens: {usage.get('total_tokens', 0)}")
            print(f"    Cost: ${usage.get('cost', 0.0):.4f}")
            print(f"    API calls: {result['api_calls']}")
            
            if 'field_usages' in usage:
                print(f"    Per-field usage tracking: {len(usage['field_usages'])} fields")
                
    print("\n" + "="*160)
    print("COMPARISON SUMMARY")
    print("="*160)
    
    # Group results by model
    models_tested = list(set(r['model_name'] for r in results))
    successful_results = [r for r in results if r['success']]
    
    print(f"MODELS TESTED: {', '.join(models_tested)}")
    print(f"SUCCESSFUL EXTRACTIONS: {len(successful_results)} out of {len(results)}")
    
    # Compare methods for each model
    for model_name in models_tested:
        model_results = [r for r in results if r['model_name'] == model_name]
        stepwise = next((r for r in model_results if 'stepwise' in r['method']), None)
        manual = next((r for r in model_results if 'manual' in r['method']), None)
        
        if stepwise and manual and stepwise['success'] and manual['success']:
            print(f"\n--- MODEL: {model_name} ---")
            print(f"TOKEN USAGE:")
            print(f"  Stepwise: {stepwise['usage'].get('total_tokens', 0)} tokens ({stepwise['api_calls']} calls)")
            print(f"  Manual:   {manual['usage'].get('total_tokens', 0)} tokens ({manual['api_calls']} call)")
            
            stepwise_cost = stepwise['usage'].get('cost', 0.0)
            manual_cost = manual['usage'].get('cost', 0.0)
            print(f"COST COMPARISON:")
            print(f"  Stepwise: ${stepwise_cost:.4f}")
            print(f"  Manual:   ${manual_cost:.4f}")
            
            if stepwise_cost > 0 and manual_cost > 0:
                ratio = stepwise_cost / manual_cost
                print(f"  Cost ratio (stepwise/manual): {ratio:.2f}x")
            
            print(f"FIELD EXTRACTION:")
            print(f"  Stepwise: {stepwise['field_count']} fields extracted")
            print(f"  Manual:   {manual['field_count']} fields extracted")
            
            stepwise_valid, _ = validate_extraction_accuracy(stepwise)
            manual_valid, _ = validate_extraction_accuracy(manual)
            print(f"ACCURACY:")
            print(f"  Both methods produce equivalent results: {'✓' if stepwise_valid and manual_valid else '✗'}")
            print(f"  Both extract core information accurately: {'✓' if stepwise_valid and manual_valid else '✗'}")
            
            if 'field_usages' in stepwise['usage']:
                print(f"  Stepwise provides per-field usage tracking: ✓")
            else:
                print(f"  Stepwise provides per-field usage tracking: ✗")
        elif stepwise or manual:
            print(f"\n--- MODEL: {model_name} ---")
            if stepwise and not stepwise['success']:
                print(f"  Stepwise method failed: {stepwise.get('error', 'Unknown error')}")
            if manual and not manual['success']:
                print(f"  Manual method failed: {manual.get('error', 'Unknown error')}")


def compare_methods_with_models():
    """Run comparison with multiple models."""
    results = []
    
    print(f"Testing with {len(MODELS_TO_TEST)} models: {', '.join(MODELS_TO_TEST)}")
    
    for model_name in MODELS_TO_TEST:
        print(f"\n--- Testing Model: {model_name} ---")
        
        # Test stepwise method
        stepwise_result = test_stepwise_method(model_name)
        results.append(stepwise_result)
        print(f"Stepwise result: {'SUCCESS' if stepwise_result['success'] else 'FAILED'}")
        
        # Test manual method
        manual_result = test_manual_method(model_name)
        results.append(manual_result)
        print(f"Manual result: {'SUCCESS' if manual_result['success'] else 'FAILED'}")
    
    return results


def main():
    """Run the stepwise vs manual comparison with multiple models."""
    print("="*80)
    print("STEPWISE vs MANUAL EXTRACTION COMPARISON - MULTI-MODEL")
    print("="*80)
    print(f"Sample text length: {len(SAMPLE_TEXT)} characters")
    print(f"Person model has {len(Person.model_fields)} fields")
    print(f"Testing both methods with {len(MODELS_TO_TEST)} models...\n")
    
    # Test both methods with multiple models
    results = compare_methods_with_models()
    
    # Print comparison table
    print_comparison_table(results)


if __name__ == "__main__":
    main()