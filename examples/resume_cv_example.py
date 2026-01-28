from typing import Optional

from pydantic import BaseModel

from prompture import extract_with_model, field_from_registry, register_field

register_field(
    "work_experience",
    {
        "type": "list",
        "description": "Work history with companies, roles, and durations",
        "instructions": "Extract work experience chronologically",
        "default": [],
        "nullable": True,
    },
)

register_field(
    "skills",
    {
        "type": "list",
        "description": "Technical skills and competencies",
        "instructions": "Extract as a list of individual skills",
        "default": [],
        "nullable": True,
    },
)


class Resume(BaseModel):
    name: str = field_from_registry("name")
    email: Optional[str] = field_from_registry("email")
    phone: Optional[str] = field_from_registry("phone")
    skills: Optional[list[str]] = field_from_registry("skills")
    education: Optional[list[dict]] = field_from_registry("education_level")
    work_experience: Optional[list[dict]] = field_from_registry("work_experience")


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
resume = extract_with_model(Resume, resume_text, "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b")

# Display extracted information
print(f"Candidate: {resume.model.name}")
print(f"Contact: {resume.model.email}")
print(f"Skills: {', '.join(resume.model.skills or [])}")
