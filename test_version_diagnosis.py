#!/usr/bin/env python3
"""
Diagnostic script to test version detection methods.
"""

from pathlib import Path

# Set project root
project_root = Path(__file__).parent

print("=" * 60)
print("VERSION DETECTION DIAGNOSIS")
print("=" * 60)
print()

# Test 1: Check VERSION file
print("1. Testing VERSION file:")
version_file = project_root / "VERSION"
if version_file.exists():
    version_content = version_file.read_text().strip()
    print("   ✓ VERSION file exists")
    print(f"   Content: {version_content}")
else:
    print("   ✗ VERSION file not found")
print()

# Test 2: Check setuptools_scm
print("2. Testing setuptools_scm:")
try:
    from setuptools_scm import get_version

    print("   ✓ setuptools_scm is installed")
    try:
        scm_version = get_version(root=str(project_root))
        print(f"   ✓ setuptools_scm version: {scm_version}")
    except Exception as e:
        print("   ✗ Error getting version from setuptools_scm:")
        print(f"      {type(e).__name__}: {e}")
except ImportError:
    print("   ✗ setuptools_scm not installed")
print()

# Test 3: Check git tags
print("3. Testing git repository:")
try:
    import subprocess

    result = subprocess.run(["git", "tag", "--list"], capture_output=True, text=True, cwd=project_root)
    if result.returncode == 0:
        tags = result.stdout.strip().split("\n") if result.stdout.strip() else []
        print("   ✓ Git repository found")
        print(f"   Number of tags: {len([t for t in tags if t])}")
        if tags and tags[0]:
            print(f"   Tags: {', '.join(tags[:10])}")
    else:
        print("   ✗ Error running git command")
except Exception as e:
    print(f"   ✗ Error checking git: {e}")
print()

# Test 4: Check what the update script would use
print("4. Simulating update_docs_version.py logic:")
print("   Priority order:")
print("   1. VERSION file")
print("   2. setuptools_scm")
print("   3. pyproject.toml")
print()

# Determine which version would be used
final_version = None
source = None

# Check VERSION file first
if version_file.exists():
    final_version = version_file.read_text().strip()
    source = "VERSION file"
else:
    # Try setuptools_scm
    try:
        from setuptools_scm import get_version

        final_version = get_version(root=str(project_root))
        source = "setuptools_scm"
    except Exception:
        source = "Failed - no version found"

print(f"   → Version that WOULD be used: {final_version}")
print(f"   → Source: {source}")
print()

print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
