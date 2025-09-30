Examples
========

This page provides practical examples of using Prompture for various data extraction scenarios. Each example demonstrates different features and use cases you might encounter in real applications.

Resume and CV Processing
------------------------

Extract structured information from resumes and CVs:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Optional
    from prompture import register_field, field_from_registry, extract_with_model

    # Register domain-specific fields
    register_field("education", {
        "type": "list",
        "description": "Educational background with degrees and institutions",
        "instructions": "Extract education as list of objects with degree, institution, year",
        "default": [],
        "nullable": True
    })

    register_field("work_experience", {
        "type": "list",
        "description": "Work history with companies, roles, and durations",
        "instructions": "Extract work experience chronologically",
        "default": [],
        "nullable": True
    })

    register_field("skills", {
        "type": "list",
        "description": "Technical skills and competencies",
        "instructions": "Extract as a list of individual skills",
        "default": [],
        "nullable": True
    })

    class Resume(BaseModel):
        name: str = field_from_registry("name")
        email: Optional[str] = field_from_registry("email")
        phone: Optional[str] = field_from_registry("phone")
        skills: Optional[List[str]] = field_from_registry("skills")
        education: Optional[List[dict]] = field_from_registry("education")
        work_experience: Optional[List[dict]] = field_from_registry("work_experience")

    # Sample resume text
    resume_text = """
    SARAH JOHNSON
    Email: sarah.johnson@email.com | Phone: (555) 123-4567

    EDUCATION
    - Master of Science in Computer Science, Stanford University (2018)
    - Bachelor of Science in Mathematics, UC Berkeley (2016)

    EXPERIENCE  
    Senior Software Engineer, TechCorp (2020-Present)
    - Lead development of microservices architecture
    - Mentor junior developers

    Software Engineer, StartupXYZ (2018-2020)  
    - Built full-stack web applications
    - Implemented CI/CD pipelines

    SKILLS: Python, JavaScript, AWS, Docker, Kubernetes, React
    """

    # Extract structured resume data
    resume = extract_with_model(
        Resume,
        resume_text,
        "openai/gpt-4"
    )

    print(f"Candidate: {resume.model.name}")
    print(f"Contact: {resume.model.email}")
    print(f"Skills: {', '.join(resume.model.skills or [])}")

Medical Record Processing
-------------------------

Extract patient information from medical documents:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Optional
    from prompture import register_field, field_from_registry, extract_with_model

    # Medical-specific field definitions
    register_field("medical_conditions", {
        "type": "list",
        "description": "List of diagnosed medical conditions",
        "instructions": "Extract diagnosed conditions, symptoms, and medical issues",
        "default": [],
        "nullable": True
    })

    register_field("medications", {
        "type": "list",
        "description": "Current medications and prescriptions", 
        "instructions": "Extract medication names, dosages, and frequencies",
        "default": [],
        "nullable": True
    })

    register_field("allergies", {
        "type": "list",
        "description": "Known allergies and adverse reactions",
        "instructions": "Extract all known allergies, food sensitivities, drug reactions",
        "default": [],
        "nullable": True
    })

    class PatientRecord(BaseModel):
        name: str = field_from_registry("name")
        age: int = field_from_registry("age")
        date_of_birth: Optional[str] = field_from_registry("date_of_birth")
        medical_conditions: Optional[List[str]] = field_from_registry("medical_conditions")
        medications: Optional[List[str]] = field_from_registry("medications")
        allergies: Optional[List[str]] = field_from_registry("allergies")

    # Sample medical record
    medical_text = """
    Patient: Robert Martinez, DOB: 1975-03-15, Age: 49

    MEDICAL HISTORY:
    - Type 2 Diabetes diagnosed 2018
    - Hypertension since 2020
    - Mild sleep apnea

    CURRENT MEDICATIONS:
    - Metformin 500mg twice daily
    - Lisinopril 10mg once daily  
    - CPAP therapy for sleep apnea

    ALLERGIES:
    - Penicillin (causes rash)
    - Shellfish (severe reaction)
    """

    # Extract patient data
    patient = extract_with_model(
        PatientRecord,
        medical_text,
        "openai/gpt-4"
    )

    print(f"Patient: {patient.model.name}, Age: {patient.model.age}")
    print(f"Conditions: {patient.model.medical_conditions}")
    print(f"Medications: {patient.model.medications}")
    print(f"Allergies: {patient.model.allergies}")

E-commerce Product Extraction
-----------------------------

Extract product information from e-commerce descriptions:

.. code-block:: python

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
        "openai/gpt-4"
    )

    print(f"Product: {product.model.name}")
    print(f"Brand: {product.model.brand}")
    print(f"Price: ${product.model.price}")
    print(f"Features: {product.model.features}")

Financial Document Analysis
---------------------------

Process financial statements and reports:

.. code-block:: python

    from pydantic import BaseModel
    from typing import Optional
    from prompture import register_field, field_from_registry, extract_with_model

    # Financial fields
    register_field("revenue", {
        "type": "float",
        "description": "Total revenue or income amount",
        "instructions": "Extract revenue figures in millions, convert to number",
        "default": 0.0,
        "nullable": True
    })

    register_field("profit_margin", {
        "type": "float", 
        "description": "Profit margin as percentage",
        "instructions": "Extract profit margin as decimal (e.g., 15% = 0.15)",
        "default": 0.0,
        "nullable": True
    })

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
    financial = extract_with_model(
        FinancialSummary,
        financial_text,
        "openai/gpt-4"
    )

    print(f"Company: {financial.model.company}")
    print(f"Revenue: ${financial.model.revenue}B {financial.model.currency}")
    print(f"Profit Margin: {financial.model.profit_margin * 100}%")

Social Media Content Analysis
-----------------------------

Extract insights from social media posts:

.. code-block:: python

    from pydantic import BaseModel
    from typing import List, Optional
    from prompture import register_field, field_from_registry, extract_with_model

    # Social media fields
    register_field("sentiment", {
        "type": "str",
        "description": "Overall sentiment of the content",
        "instructions": "Classify as 'positive', 'negative', or 'neutral'",
        "default": "neutral",
        "nullable": False
    })

    register_field("hashtags", {
        "type": "list",
        "description": "Hashtags mentioned in the content",
        "instructions": "Extract all hashtags including the # symbol",
        "default": [],
        "nullable": True
    })

    register_field("mentions", {
        "type": "list",
        "description": "User mentions in the content", 
        "instructions": "Extract @username mentions",
        "default": [],
        "nullable": True
    })

    class SocialPost(BaseModel):
        content: str = field_from_registry("content")
        sentiment: str = field_from_registry("sentiment")
        hashtags: Optional[List[str]] = field_from_registry("hashtags")
        mentions: Optional[List[str]] = field_from_registry("mentions")
        topic: Optional[str] = field_from_registry("topic")

    # Sample social media post
    social_text = """
    Just had an amazing experience at @StarbucksCoffee! Their new winter blend 
    is absolutely delicious ☕️ Perfect way to start the morning. Highly recommend 
    trying it! #coffee #winterblend #morningvibes #recommendation #delicious

    The barista was super friendly and the service was quick. Will definitely 
    be back soon! 5 stars ⭐⭐⭐⭐⭐
    """

    # Extract social media insights
    post = extract_with_model(
        SocialPost,
        social_text,
        "openai/gpt-4"
    )

    print(f"Sentiment: {post.model.sentiment}")
    print(f"Hashtags: {post.model.hashtags}")
    print(f"Mentions: {post.model.mentions}")
    print(f"Topic: {post.model.topic}")

Cross-Model Comparison
----------------------

Compare extraction results across different LLM providers:

.. code-block:: python

    from prompture import extract_with_model
    from pydantic import BaseModel
    from typing import Optional

    class BasicInfo(BaseModel):
        name: str = field_from_registry("name")
        age: int = field_from_registry("age") 
        location: Optional[str] = field_from_registry("location")

    text = "Jennifer Liu, 34, currently living in San Francisco, California"

    # Test across multiple models
    models = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4", 
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-sonnet-20240229",
        "google/gemini-pro"
    ]

    results = {}
    for model in models:
        try:
            result = extract_with_model(
                BasicInfo,
                text,
                model
            )
            results[model] = result
        except Exception as e:
            print(f"Failed with {model}: {e}")

    # Compare results
    for model, result in results.items():
        print(f"{model}: {result.model.name}, {result.model.age}, {result.model.location}")

Batch Processing with Error Handling
------------------------------------

Process multiple documents with robust error handling:

.. code-block:: python

    from prompture import extract_with_model
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

    def process_contacts_batch(texts: List[str], model_name: str = "openai/gpt-3.5-turbo"):
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

Working with Local Models (Ollama)
----------------------------------

Use Prompture with locally hosted models via Ollama:

.. code-block:: python

    from prompture import extract_and_jsonify

    # Make sure Ollama is running locally with a model installed
    # Example: ollama pull llama2

    fields = {
        "name": "name",
        "age": "age",
        "occupation": "occupation"
    }

    text = "Dr. Sarah Chen is a 42-year-old neurosurgeon at City Hospital."

    # Extract using local Ollama model
    result = extract_and_jsonify(
        prompt=text,
        fields=fields,
        model_name="ollama/llama2"  # or ollama/mistral, ollama/codellama, etc.
    )

    print("Local extraction result:", result)

Real-time Data Processing
-------------------------

Process streaming or real-time data:

.. code-block:: python

    import time
    from prompture import extract_with_model
    from pydantic import BaseModel
    from typing import Optional

    class NewsArticle(BaseModel):
        headline: str = field_from_registry("title") 
        sentiment: str = field_from_registry("sentiment")
        category: Optional[str] = field_from_registry("category")
        key_points: Optional[list] = field_from_registry("key_points")

    def process_news_stream(articles: list):
        """Simulate real-time news processing."""
        for i, article_text in enumerate(articles):
            print(f"Processing article {i+1}...")
            
            try:
                # Extract structured data
                article = extract_with_model(
                    NewsArticle,
                    article_text,
                    "openai/gpt-3.5-turbo"
                )
                
                print(f"  Headline: {article.model.headline}")
                print(f"  Sentiment: {article.model.sentiment}")
                print(f"  Category: {article.model.category}")
                print("  ---")
                
                # Simulate processing time
                time.sleep(1)
                
            except Exception as e:
                print(f"  Error processing article {i+1}: {e}")

    # Sample news articles
    news_articles = [
        "Tech stocks surge as AI company announces breakthrough in quantum computing...",
        "Local sports team wins championship after decade-long drought...", 
        "New environmental regulations announced to combat climate change..."
    ]

    process_news_stream(news_articles)

Tips and Best Practices
------------------------

**Field Definition Strategy**
  - Use built-in fields when possible for consistency
  - Register custom fields for domain-specific needs
  - Include clear instructions in field definitions

**Model Selection**  
  - Use GPT-3.5-turbo for fast, cost-effective processing
  - Use GPT-4 for complex or nuanced extraction tasks
  - Use Claude models for alternative perspectives
  - Use local models for privacy-sensitive data

**Error Handling**
  - Always wrap extraction calls in try-catch blocks
  - Validate critical fields after extraction
  - Log failures for debugging and improvement

**Performance Optimization**
  - Batch similar extractions together
  - Cache field definitions for reuse
  - Use appropriate model sizes for your use case

**Data Quality**
  - Use Pydantic models for type safety
  - Validate extracted data against schemas
  - Implement fallback strategies for missing data

For more advanced usage patterns and complete API documentation, see the :doc:`api/index` section.