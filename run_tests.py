import pytest
import sys

def main():
    # Run all the tests in the unittests directory
    result = pytest.main(['unittests'])

    # Exit with the appropriate status code
    sys.exit(result)

if __name__ == "__main__":
    main()