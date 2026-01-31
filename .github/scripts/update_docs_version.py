#!/usr/bin/env python3
"""
Script to dynamically update the version number in Sphinx documentation.

This script updates the hardcoded version in docs/source/index.rst before
the Sphinx build runs. It reads the version from multiple sources in priority order:
1. VERSION file at project root (if exists)
2. setuptools_scm (if available)
3. pyproject.toml as fallback

Usage:
    python .github/scripts/update_docs_version.py
"""

import re
import sys
from pathlib import Path


def get_version_from_file(project_root):
    """
    Read version from the VERSION file at project root.

    Args:
        project_root: Path to the project root directory

    Returns:
        str: Version string if found, None otherwise
    """
    version_file = project_root / "VERSION"
    if version_file.exists():
        try:
            version = version_file.read_text().strip()
            print(f"✓ Found version in VERSION file: {version}")
            return version
        except Exception as e:
            print(f"✗ Error reading VERSION file: {e}", file=sys.stderr)
    return None


def get_version_from_setuptools_scm(project_root):
    """
    Get version from setuptools_scm if available.

    Args:
        project_root: Path to the project root directory

    Returns:
        str: Version string if found, None otherwise
    """
    try:
        from setuptools_scm import get_version

        version = get_version(root=str(project_root))
        print(f"✓ Found version from setuptools_scm: {version}")
        return version
    except ImportError:
        print("✗ setuptools_scm not available", file=sys.stderr)
    except Exception as e:
        print(f"✗ Error getting version from setuptools_scm: {e}", file=sys.stderr)
    return None


def get_version_from_pyproject(project_root):
    """
    Extract version from pyproject.toml as a fallback.

    Args:
        project_root: Path to the project root directory

    Returns:
        str: Version string if found, None otherwise
    """
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            content = pyproject_file.read_text()
            # Look for version = "X.X.X" pattern
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                version = match.group(1)
                print(f"✓ Found version in pyproject.toml: {version}")
                return version
        except Exception as e:
            print(f"✗ Error reading pyproject.toml: {e}", file=sys.stderr)
    return None


def get_version(project_root):
    """
    Get version from available sources in priority order.

    Priority:
    1. VERSION file
    2. setuptools_scm
    3. pyproject.toml

    Args:
        project_root: Path to the project root directory

    Returns:
        str: Version string

    Raises:
        RuntimeError: If no version can be determined
    """
    # Try VERSION file first
    version = get_version_from_file(project_root)
    if version:
        return version

    # Try setuptools_scm
    version = get_version_from_setuptools_scm(project_root)
    if version:
        return version

    # Fallback to pyproject.toml
    version = get_version_from_pyproject(project_root)
    if version:
        return version

    raise RuntimeError("Could not determine version from any source")


def update_index_rst(index_file, version):
    """
    Update the version number in docs/source/index.rst.

    Args:
        index_file: Path to the index.rst file
        version: Version string to insert

    Returns:
        bool: True if file was updated, False otherwise
    """
    if not index_file.exists():
        print(f"✗ File not found: {index_file}", file=sys.stderr)
        return False

    try:
        # Read the file
        content = index_file.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)

        # Pattern to match the version line
        pattern = re.compile(
            r"^(\s*Prompture is currently in development \(version )"
            r"[^)]+"
            r"(\)\. APIs may change between versions\.\s*)$"
        )

        updated = False
        for i, line in enumerate(lines):
            if pattern.match(line):
                # Replace with new version
                new_line = pattern.sub(rf"\g<1>{version}\g<2>", line)
                lines[i] = new_line
                print(f"✓ Updated version in {index_file} at line {i + 1}")
                print(f"  Old: {line.strip()}")
                print(f"  New: {new_line.strip()}")
                updated = True
                break

        if updated:
            # Write back to file
            index_file.write_text("".join(lines), encoding="utf-8")
            return True
        else:
            print(f"✗ Version pattern not found in {index_file}", file=sys.stderr)
            return False

    except Exception as e:
        print(f"✗ Error updating index.rst: {e}", file=sys.stderr)
        return False


def update_conf_py(conf_file, version):
    """
    Update the release version in docs/source/conf.py fallback.

    Args:
        conf_file: Path to the conf.py file
        version: Version string to insert

    Returns:
        bool: True if file was updated, False otherwise
    """
    if not conf_file.exists():
        print(f"✗ File not found: {conf_file}", file=sys.stderr)
        return False

    try:
        content = conf_file.read_text(encoding="utf-8")
        # Update the hardcoded fallback version in conf.py
        new_content, count = re.subn(
            r'(release\s*=\s*")[^"]+(")',
            rf"\g<1>{version}\g<2>",
            content,
        )
        if count > 0:
            conf_file.write_text(new_content, encoding="utf-8")
            print(f"✓ Updated fallback release version in {conf_file}")
            return True
        else:
            print(f"✗ release pattern not found in {conf_file}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"✗ Error updating conf.py: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point for the script."""
    # Determine project root (two levels up from this script)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    print(f"Project root: {project_root}")

    # Get version
    try:
        version = get_version(project_root)
        print(f"\n→ Using version: {version}\n")
    except RuntimeError as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

    # Update index.rst
    index_file = project_root / "docs" / "source" / "index.rst"
    rst_ok = update_index_rst(index_file, version)

    # Update conf.py fallback version
    conf_file = project_root / "docs" / "source" / "conf.py"
    conf_ok = update_conf_py(conf_file, version)

    if rst_ok or conf_ok:
        print("\n✓ Documentation version updated successfully")
        sys.exit(0)
    else:
        print("\n✗ Failed to update documentation version", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
