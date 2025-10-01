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

import os
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
        content = index_file.read_text(encoding='utf-8')
        lines = content.splitlines(keepends=True)
        
        # Pattern to match the version line (line 20, 0-indexed as 19)
        pattern = re.compile(
            r'^(\s*Prompture is currently in development \(version )'
            r'[^)]+' 
            r'(\)\. APIs may change between versions\.\s*)$'
        )
        
        # Update line 20 (index 19)
        if len(lines) >= 20:
            line_idx = 19  # Line 20 is at index 19
            original_line = lines[line_idx]
            
            # Check if the line matches the expected pattern
            if pattern.match(original_line):
                # Replace with new version
                new_line = pattern.sub(
                    rf'\g<1>{version}\g<2>',
                    original_line
                )
                lines[line_idx] = new_line
                
                # Write back to file
                index_file.write_text(''.join(lines), encoding='utf-8')
                print(f"✓ Updated version in {index_file}")
                print(f"  Old: {original_line.strip()}")
                print(f"  New: {new_line.strip()}")
                return True
            else:
                print(f"✗ Line 20 does not match expected pattern", file=sys.stderr)
                print(f"  Found: {original_line.strip()}", file=sys.stderr)
                return False
        else:
            print(f"✗ File has fewer than 20 lines", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error updating index.rst: {e}", file=sys.stderr)
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
    success = update_index_rst(index_file, version)
    
    if success:
        print("\n✓ Documentation version updated successfully")
        sys.exit(0)
    else:
        print("\n✗ Failed to update documentation version", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()