"""Project scaffolding generator.

Renders Jinja2 templates into a standalone FastAPI project directory
that users can customize and deploy.
"""

from __future__ import annotations

from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None  # type: ignore[assignment,misc]
    FileSystemLoader = None  # type: ignore[assignment,misc]

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Map from template file -> output path (relative to project root).
_FILE_MAP = {
    "main.py.j2": "app/main.py",
    "models.py.j2": "app/models.py",
    "config.py.j2": "app/config.py",
    "requirements.txt.j2": "requirements.txt",
    "env.example.j2": ".env.example",
    "README.md.j2": "README.md",
}

_DOCKER_FILES = {
    "Dockerfile.j2": "Dockerfile",
}


def scaffold_project(
    output_dir: str,
    project_name: str = "my_app",
    model_name: str = "openai/gpt-4o-mini",
    include_docker: bool = True,
) -> Path:
    """Render all templates and write the project to *output_dir*.

    Parameters:
        output_dir: Destination directory (created if needed).
        project_name: Human-friendly project name used in templates.
        model_name: Default model string baked into config.
        include_docker: Whether to include Dockerfile.

    Returns:
        The :class:`Path` to the generated project root.
    """
    if Environment is None:
        raise ImportError("jinja2 is required for scaffolding: pip install prompture[scaffold]")

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )

    context = {
        "project_name": project_name,
        "model_name": model_name,
        "include_docker": include_docker,
    }

    out = Path(output_dir)

    file_map = dict(_FILE_MAP)
    if include_docker:
        file_map.update(_DOCKER_FILES)

    for template_name, rel_path in file_map.items():
        template = env.get_template(template_name)
        rendered = template.render(**context)

        dest = out / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8")

    # Create empty __init__.py for the app package
    init_path = out / "app" / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    return out
