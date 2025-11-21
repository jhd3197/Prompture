import sys
from pathlib import Path

import toml

WRAPPER_ROOT = Path("packages")


def update_pyproject(pyproject_path: Path, new_version: str) -> None:
    with pyproject_path.open("r", encoding="utf-8") as f:
        data = toml.load(f)

    data["project"]["version"] = new_version

    if "dependencies" in data["project"]:
        for i, dep in enumerate(data["project"]["dependencies"]):
            if dep.startswith("prompture"):
                data["project"]["dependencies"][i] = f"prompture>={new_version}"
                break

    with pyproject_path.open("w", encoding="utf-8") as f:
        toml.dump(data, f)

    print(f"Updated {pyproject_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python update_wrapper_version.py <version>")
        sys.exit(1)
    new_version = sys.argv[1]

    candidates = sorted(
        path / "pyproject.toml"
        for path in WRAPPER_ROOT.iterdir()
        if path.is_dir() and (path / "pyproject.toml").exists()
    )

    if not candidates:
        print("No wrapper pyproject.toml files found under 'packages/'.")
        sys.exit(1)

    for pyproject_path in candidates:
        update_pyproject(pyproject_path, new_version)


if __name__ == "__main__":
    main()
