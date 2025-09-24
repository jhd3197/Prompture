from setuptools import setup, find_packages
from pathlib import Path

# Read README.md with UTF-8 encoding
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="prompture",
    use_scm_version=True, 
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.8.0",
        "click>=8.0",
        "httpx>=0.25.0",
        "jsonschema>=4.0",
        "openai>=1.0.0",
        "pydantic>=1.10",
        "pydantic-settings>=2.0",
        "python-dotenv>=0.19.0",
        "requests>=2.28"
    ],
    author="Juan Denis",
    author_email="juan@vene.co",
    description="Ask LLMs to return structured JSON and run cross-model tests. API-first.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jhd3197/prompture",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "prompture=prompture.cli:cli",
        ],
    },
    extras_require={
        "test": ["pytest>=7.0"],
    },
)