import pytest
import sys

# Add src directory to path for prompture package imports
sys.path.append('src')

# Run pytest and capture exit code
exit_code = pytest.main()

# Exit with the pytest exit code for CI/CD integration
sys.exit(exit_code)