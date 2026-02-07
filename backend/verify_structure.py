# Backend Structure Verification Script
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    '''Check if a file exists and print result'''
    exists = os.path.exists(filepath)
    status = '✓' if exists else '✗'
    color = '\033[92m' if exists else '\033[91m'
    reset = '\033[0m'
    print(f'{color}{status}{reset} {description}: {filepath}')
    return exists

def verify_backend_structure():
    '''Verify all backend files and folders exist'''
    
    print('\n' + '='*60)
    print('BACKEND STRUCTURE VERIFICATION')
    print('='*60 + '\n')
    
    checks = []
    
    # Root files
    print('📁 Root Files:')
    checks.append(check_file_exists('requirements.txt', 'Requirements file'))
    checks.append(check_file_exists('config.py', 'Config file'))
    checks.append(check_file_exists('app.py', 'Main app file'))
    checks.append(check_file_exists('.env.example', 'Env example'))
    
    # Source structure
    print('\n📁 Source Structure:')
    checks.append(check_file_exists('src', 'Source directory'))
    checks.append(check_file_exists('src/routes', 'Routes directory'))
    checks.append(check_file_exists('src/services', 'Services directory'))
    checks.append(check_file_exists('src/models', 'Models directory'))
    checks.append(check_file_exists('src/utils', 'Utils directory'))
    checks.append(check_file_exists('src/data', 'Data directory'))
    
    # Route files
    print('\n📁 Route Files:')
    checks.append(check_file_exists('src/routes/__init__.py', 'Routes init'))
    checks.append(check_file_exists('src/routes/news.py', 'News routes'))
    checks.append(check_file_exists('src/routes/stocks.py', 'Stocks routes'))
    checks.append(check_file_exists('src/routes/companies.py', 'Companies routes'))
    checks.append(check_file_exists('src/routes/sentiment.py', 'Sentiment routes'))
    
    # Service files
    print('\n📁 Service Files:')
    checks.append(check_file_exists('src/services/__init__.py', 'Services init'))
    checks.append(check_file_exists('src/services/rss_scraper.py', 'RSS scraper'))
    checks.append(check_file_exists('src/services/stock_fetcher.py', 'Stock fetcher'))
    checks.append(check_file_exists('src/services/sentiment_analyzer.py', 'Sentiment analyzer'))
    
    # Util files
    print('\n📁 Utility Files:')
    checks.append(check_file_exists('src/utils/__init__.py', 'Utils init'))
    checks.append(check_file_exists('src/utils/database.py', 'Database utility'))
    checks.append(check_file_exists('src/utils/logger.py', 'Logger utility'))
    checks.append(check_file_exists('src/utils/scheduler.py', 'Scheduler utility'))
    
    # Data files
    print('\n📁 Data Files:')
    checks.append(check_file_exists('src/data/quantum_companies.json', 'Companies data'))
    checks.append(check_file_exists('src/data/rss_feeds.json', 'RSS feeds data'))
    
    # Summary
    print('\n' + '='*60)
    total = len(checks)
    passed = sum(checks)
    print(f'SUMMARY: {passed}/{total} checks passed')
    
    if passed == total:
        print('\033[92m✓ All files and folders are present!\033[0m')
    else:
        print(f'\033[91m✗ {total - passed} files/folders are missing!\033[0m')
    print('='*60 + '\n')
    
    return passed == total

if __name__ == '__main__':
    success = verify_backend_structure()
    sys.exit(0 if success else 1)
