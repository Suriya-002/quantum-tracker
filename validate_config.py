# Configuration Files Validation Script
import json
import os

def validate_json_file(filepath, description):
    '''Validate a JSON file'''
    print(f'\nValidating {description}...')
    
    if not os.path.exists(filepath):
        print(f'  ✗ File not found: {filepath}')
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f'  ✓ Valid JSON format')
        print(f'  ✓ Keys: {list(data.keys())}')
        return True
    except json.JSONDecodeError as e:
        print(f'  ✗ Invalid JSON: {str(e)}')
        return False
    except Exception as e:
        print(f'  ✗ Error: {str(e)}')
        return False

def validate_config_file(filepath):
    '''Validate Python config file'''
    print(f'\nValidating config.py...')
    
    if not os.path.exists(filepath):
        print(f'  ✗ File not found: {filepath}')
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_vars = [
            'SECRET_KEY',
            'DATABASE_PATH',
            'RSS_UPDATE_INTERVAL',
            'STOCK_UPDATE_INTERVAL',
            'QUANTUM_TICKERS'
        ]
        
        missing = [var for var in required_vars if var not in content]
        
        if missing:
            print(f'  ✗ Missing variables: {missing}')
            return False
        
        print(f'  ✓ All required variables present')
        return True
        
    except Exception as e:
        print(f'  ✗ Error: {str(e)}')
        return False

def main():
    print('='*60)
    print('CONFIGURATION FILES VALIDATION')
    print('='*60)
    
    checks = []
    
    # Backend config
    checks.append(validate_config_file('backend/config.py'))
    
    # Data files
    checks.append(validate_json_file(
        'backend/src/data/quantum_companies.json',
        'Quantum Companies Data'
    ))
    checks.append(validate_json_file(
        'backend/src/data/rss_feeds.json',
        'RSS Feeds Data'
    ))
    
    # Frontend config
    checks.append(validate_json_file(
        'frontend/package.json',
        'Frontend Package Config'
    ))
    
    # Summary
    print('\n' + '='*60)
    total = len(checks)
    passed = sum(checks)
    print(f'VALIDATION SUMMARY: {passed}/{total} checks passed')
    
    if passed == total:
        print('\033[92m✓ All configuration files are valid!\033[0m')
    else:
        print(f'\033[91m✗ {total - passed} configuration file(s) have issues!\033[0m')
    print('='*60 + '\n')

if __name__ == '__main__':
    main()
