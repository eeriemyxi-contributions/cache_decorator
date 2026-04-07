import os

def pytest_sessionstart(session):
    """Generate the readme test file before starting the test session."""
    from pytest_readme import setup
    setup()
    
    if os.path.exists("test_readme.py"):
        os.replace("test_readme.py", "tests/test_readme.py")
