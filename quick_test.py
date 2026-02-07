# Quick Test - Check if we can import modules
import sys
import os

print('='*60)
print('QUICK IMPORT TEST')
print('='*60 + '\n')

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), 'backend'))

tests_passed = 0
tests_total = 0

def test_import(module_name, description):
    global tests_passed, tests_total
    tests_total += 1
    try:
        __import__(module_name)
        print(f'✓ {description}')
        tests_passed += 1
        return True
    except ImportError as e:
        print(f'✗ {description}: {str(e)}')
        return False
    except Exception as e:
        print(f'✗ {description}: {str(e)}')
        return False

print('Testing Python imports:\n')

# Test standard library imports
test_import('json', 'JSON module')
test_import('sqlite3', 'SQLite3 module')
test_import('logging', 'Logging module')

# Test if files can be imported as modules
print('\nTesting project modules:\n')

try:
    from backend.config import Config
    print('✓ Config class')
    tests_passed += 1
except:
    print('✗ Config class - Run: cd backend && pip install -r requirements.txt')

tests_total += 1

print('\n' + '='*60)
print(f'SUMMARY: {tests_passed}/{tests_total} tests passed')
print('='*60 + '\n')

if tests_passed < tests_total:
    print('Note: Install dependencies to pass all tests:')
    print('  cd backend')
    print('  python -m venv venv')
    print('  .\\venv\\Scripts\\Activate.ps1')
    print('  pip install -r requirements.txt')
