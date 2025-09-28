"""
Comparison example: stepwise_extract_with_model() vs extract_and_jsonify()

This script demonstrates the differences between the two extraction methods:
1. stepwise_extract_with_model() - Makes separate API calls for each field
2. extract_and_jsonify() - Makes a single API call for all fields

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
from prompture import stepwise_extract_with_model, extract_and_jsonify


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
In a recent interview with Dr. Sarah E. Johnson (previously Sarah Elizabeth Smith), age thirty-four,
we gained insights into her groundbreaking work in AI development. Having entered the tech industry
after completing her Ph.D. in 2011, Dr. Johnson has accumulated approximately twelve years of hands-on
experience in various technical roles.

As the newly appointed Senior Software Engineering Manager at TechCorp International (a Fortune 500
company), she leads their AI Innovation Lab. This promotion came with a significant compensation
adjustment - her base salary increased from $165,000 to roughly $185K annually, plus stock options
valued at $50,000.

Dr. Johnson can typically be reached through her work email (s.johnson@techcorp-international.com)
or her personal email (sarah.e.johnson@gmail.com). Her office number is listed as (555) 123-4567,
though she prefers email contact. Having recently relocated from Seattle, she now works from
TechCorp's San Francisco headquarters.

Her journey from an ML Engineer to Senior Engineering Manager showcases her technical prowess and
leadership capabilities. While maintaining her current employment status, she also serves as a
technical advisor for several AI startups. Under her guidance, the AI Innovation Lab has grown from
5 to 25 engineers, with plans for further expansion.

Dr. Johnson's commitment to advancing AI technology is evident in her approach to team building and
project management. Despite multiple competing offers, she remains actively employed at TechCorp,
where she continues to drive innovation in artificial intelligence and machine learning applications.
"""

# JSON Schema for extract_and_jsonify
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
    "ollama/gpt-oss:20b",
    "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b",
    "openai/gpt-3.5-turbo",
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
    """Test extract_and_jsonify method with optional model specification."""
    print(f"Testing extract_and_jsonify() with model: {model_name or 'default'}...")
    
    try:
        result = extract_and_jsonify(
            text=SAMPLE_TEXT,
            json_schema=PERSON_SCHEMA,
            model_name=model_name if model_name else None
        )
        
        # Convert to Person model for comparison
        person_model = Person(**result['json_object'])
        
        return {
            'success': True,
            'method': 'extract_and_jsonify',
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
            'method': 'extract_and_jsonify',
            'model_name': model_name or 'default',
            'error': str(e)[:50],  # Truncate error message
            'extracted_model': None,
            'usage': {},
            'field_count': 0,
            'api_calls': 0
        }


def get_stepwise_advantage(result):
    """Determine why stepwise extraction is advantageous for this result."""
    if not result['success']:
        return "Failed to extract"
        
    if 'stepwise' not in result['method']:
        return "N/A (Manual method)"
        
    advantages = []
    
    # Check for field-level tracking
    if 'field_usages' in result['usage']:
        advantages.append("Per-field tracking")
    
    # Validate extraction quality
    is_valid, errors = validate_extraction_accuracy(result)
    if is_valid:
        advantages.append("High accuracy")
    
    # Check field coverage
    if result['field_count'] >= 8:  # If most optional fields were extracted
        advantages.append("Better coverage")
        
    if not advantages:
        return "Standard extraction"
        
    return ", ".join(advantages[:2])  # Return top 2 advantages


def validate_extraction_accuracy(result):
    """Validate that key fields were extracted correctly."""
    if not result['success'] or not result['extracted_model']:
        return False, "Extraction failed"
    
    person = result['extracted_model']
    errors = []
    
    # Validate required fields
    if not person.name or not any(name in person.name.lower() for name in ['sarah', 'johnson']):
        errors.append("Name not correctly extracted")
    
    if not person.age or not (30 <= person.age <= 38):  # Allow some flexibility
        errors.append(f"Age incorrect: expected ~34, got {person.age}")
        
    if not person.profession or not any(role in person.profession.lower() for role in ['manager', 'engineering', 'software']):
        errors.append("Profession not correctly extracted")
        
    if person.is_employed is not True:
        errors.append("Employment status incorrect")
    
    # Validate optional fields with more flexible matching
    if person.salary:
        salary_value = float(person.salary)
        if not (175000 <= salary_value <= 195000):  # Allow 10K variance
            errors.append(f"Salary outside expected range: got {salary_value}")
        
    if person.email and not any(email in person.email.lower() for email in ['sarah', 'johnson', '@']):
        errors.append("Email not correctly extracted")
        
    if person.city and 'san francisco' not in person.city.lower():
        errors.append("City not correctly extracted")
        
    if person.years_experience and not (10 <= person.years_experience <= 14):  # Allow some flexibility
        errors.append(f"Experience outside expected range: got {person.years_experience}")
    
    return len(errors) == 0, errors


def print_comparison_table(results):
    """Print focused comparison table highlighting key differences between methods."""
    
    print("\n" + "="*150)
    print("STEPWISE VS MANUAL EXTRACTION COMPARISON")
    print("="*150)
    
    # Streamlined headers focusing on essential metrics
    row_format = "{:<25} {:<35} {:<8} {:<12} {:<8} {:<12} {:<15} {:<30}"
    headers = [
        "Method", "Model", "Success", "Total Tokens", "Calls", "Validated", "Cost ($)", "Why Stepwise Wins"
    ]
    print(row_format.format(*headers))
    print("-" * 150)
    
    for result in results:
        if result['success']:
            usage = result['usage']
            total_tokens = usage.get('total_tokens', 0)
            cost = usage.get('cost', 0.0)
            
            # Validate accuracy
            is_valid, _ = validate_extraction_accuracy(result)
            validation = "✓" if is_valid else "✗"
            
            # Determine why stepwise wins for this case
            stepwise_advantage = get_stepwise_advantage(result)
            
        else:
            total_tokens = 0
            cost = 0.0
            validation = "✗"
            stepwise_advantage = "Failed to extract"
        
        # Format model name to prevent truncation
        model_name = result['model_name']
        if len(model_name) > 35:
            parts = model_name.split('/')
            if len(parts) > 2:
                model_name = f"{parts[0]}/{parts[-2]}/{parts[-1]}"
        
        print(row_format.format(
            result['method'][:25],
            model_name[:35],
            "✓" if result['success'] else "✗",
            str(total_tokens)[:12],
            str(result['api_calls'])[:8],
            validation[:12],
            f"{cost:.4f}"[:15],
            stepwise_advantage[:30]
        ))
    
    print("\n" + "="*140)
    print("COMPARISON SUMMARY")
    print("="*140)
    
    models_tested = list(set(r['model_name'] for r in results))
    successful_results = [r for r in results if r['success']]
    
    print(f"\nOverall Results:")
    print(f"- Models tested: {len(models_tested)}")
    print(f"- Total extractions attempted: {len(results)}")
    print(f"- Successful extractions: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
    
    # Compare methods across all models
    stepwise_results = [r for r in successful_results if 'stepwise' in r['method']]
    manual_results = [r for r in successful_results if 'manual' in r['method']]
    
    if stepwise_results and manual_results:
        avg_stepwise_tokens = sum(r['usage'].get('total_tokens', 0) for r in stepwise_results) / len(stepwise_results)
        avg_manual_tokens = sum(r['usage'].get('total_tokens', 0) for r in manual_results) / len(manual_results)
        
        print(f"\nKey Findings:")
        print(f"1. Stepwise extraction succeeded in {len(stepwise_results)} out of {len(models_tested)} models")
        print(f"2. Manual extraction succeeded in {len(manual_results)} out of {len(models_tested)} models")
        print(f"3. Average token usage:")
        print(f"   - Stepwise: {avg_stepwise_tokens:.0f} tokens ({avg_stepwise_tokens/len(stepwise_results):.0f} per model)")
        print(f"   - Manual: {avg_manual_tokens:.0f} tokens ({avg_manual_tokens/len(manual_results):.0f} per model)")
        
        if stepwise_results[0]['usage'].get('field_usages'):
            print("4. Field-level details:")
            print("   - Stepwise method provides granular per-field tracking")
            print("   - Allows targeted retries for failed fields")
        
        validated_stepwise = sum(1 for r in stepwise_results if validate_extraction_accuracy(r)[0])
        validated_manual = sum(1 for r in manual_results if validate_extraction_accuracy(r)[0])
        print(f"5. Validation Success Rate:")
        print(f"   - Stepwise: {validated_stepwise}/{len(stepwise_results)} models ({validated_stepwise/len(stepwise_results)*100:.1f}%)")
        print(f"   - Manual: {validated_manual}/{len(manual_results)} models ({validated_manual/len(manual_results)*100:.1f}%)")


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