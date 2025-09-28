import sys
import toml

PYPROJECT_PATH = "packages/pyproject.toml"

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_wrapper_version.py <version>")
        sys.exit(1)
    new_version = sys.argv[1]

    with open(PYPROJECT_PATH, "r", encoding="utf-8") as f:
        data = toml.load(f)

    # Update version
    data["project"]["version"] = new_version

    # Update prompture dependency
    if "dependencies" in data["project"]:
        for i, dep in enumerate(data["project"]["dependencies"]):
            if dep.startswith("prompture"):
                data["project"]["dependencies"][i] = f"prompture>={new_version}"
                break

    with open(PYPROJECT_PATH, "w", encoding="utf-8") as f:
        toml.dump(data, f)

    # Print updated content for verification
    with open(PYPROJECT_PATH, "r", encoding="utf-8") as f:
        print(f.read())

if __name__ == "__main__":
    main()