"""
Batch Processing with Error Handling Example

This example shows how to use Prompture for processing multiple documents with robust error handling.
It demonstrates batch processing of contact information with validation and comprehensive error tracking.
"""

from prompture import extract_with_model, field_from_registry
from pydantic import BaseModel
from typing import List, Optional
import json

class ContactInfo(BaseModel):
    name: str = field_from_registry("name")
    email: Optional[str] = field_from_registry("email")
    phone: Optional[str] = field_from_registry("phone")
    company: Optional[str] = field_from_registry("company")

# Sample contact data
contact_texts = [
    "John Smith, Software Engineer at TechCorp - john@techcorp.com, (555) 123-4567",
    "Alice Johnson | Marketing Director | alice.j@startup.com | +1-555-987-6543",
    "Invalid contact info without proper structure...",
    "Bob Wilson - CEO, Wilson Industries - bwilson@wilson.com - 555.111.2222",
    ""  # Empty text
]

def process_contacts_batch(texts: List[str], model_name: str = "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"):
    """Process multiple contact texts with error handling."""
    results = []
    errors = []
    
    for i, text in enumerate(texts):
        if not text.strip():
            errors.append(f"Empty text at index {i}")
            continue
            
        try:
            contact = extract_with_model(
                ContactInfo,
                text,
                model_name
            )
            
            # Validate required fields
            if not contact.model.name:
                errors.append(f"No name found in text {i}: '{text[:50]}...'")
                continue
                
            results.append({
                "index": i,
                "original_text": text,
                "extracted_data": contact,
                "success": True
            })
            
        except Exception as e:
            errors.append(f"Extraction failed for text {i}: {str(e)}")
            results.append({
                "index": i,
                "original_text": text,
                "extracted_data": None,
                "success": False,
                "error": str(e)
            })
    
    return results, errors

# Process the batch
results, errors = process_contacts_batch(contact_texts)

# Display results
print("SUCCESSFUL EXTRACTIONS:")
for result in results:
    if result["success"]:
        contact = result["extracted_data"]
        print(f"  {contact.model.name} - {contact.model.email} ({contact.model.company})")

print(f"\nERRORS ({len(errors)}):")
for error in errors:
    print(f"  - {error}")