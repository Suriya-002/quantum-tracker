// Frontend Structure Verification Script
const fs = require('fs');
const path = require('path');

const checkFileExists = (filepath, description) => {
    const exists = fs.existsSync(filepath);
    const status = exists ? '✓' : '✗';
    const color = exists ? '\x1b[92m' : '\x1b[91m';
    const reset = '\x1b[0m';
    console.log(${color} : );
    return exists;
};

const verifyFrontendStructure = () => {
    console.log('\n' + '='.repeat(60));
    console.log('FRONTEND STRUCTURE VERIFICATION');
    console.log('='.repeat(60) + '\n');
    
    const checks = [];
    
    // Root files
    console.log('📁 Root Files:');
    checks.push(checkFileExists('package.json', 'Package file'));
    checks.push(checkFileExists('vite.config.js', 'Vite config'));
    checks.push(checkFileExists('index.html', 'Index HTML'));
    checks.push(checkFileExists('.env.example', 'Env example'));
    
    // Source structure
    console.log('\n📁 Source Structure:');
    checks.push(checkFileExists('src', 'Source directory'));
    checks.push(checkFileExists('src/main.jsx', 'Main entry file'));
    checks.push(checkFileExists('src/App.jsx', 'App component'));
    checks.push(checkFileExists('src/index.css', 'Main CSS'));
    
    // Component directories
    console.log('\n📁 Component Directories:');
    checks.push(checkFileExists('src/components', 'Components directory'));
    checks.push(checkFileExists('src/components/layout', 'Layout components'));
    checks.push(checkFileExists('src/components/news', 'News components'));
    checks.push(checkFileExists('src/components/stocks', 'Stock components'));
    checks.push(checkFileExists('src/components/companies', 'Company components'));
    checks.push(checkFileExists('src/components/common', 'Common components'));
    
    // Other directories
    console.log('\n📁 Other Directories:');
    checks.push(checkFileExists('src/pages', 'Pages directory'));
    checks.push(checkFileExists('src/services', 'Services directory'));
    checks.push(checkFileExists('src/context', 'Context directory'));
    checks.push(checkFileExists('src/hooks', 'Hooks directory'));
    checks.push(checkFileExists('src/utils', 'Utils directory'));
    
    // Summary
    console.log('\n' + '='.repeat(60));
    const total = checks.length;
    const passed = checks.filter(Boolean).length;
    console.log(SUMMARY: / checks passed);
    
    if (passed === total) {
        console.log('\x1b[92m✓ All files and folders are present!\x1b[0m');
    } else {
        console.log(\x1b[91m✗  files/folders are missing!\x1b[0m);
    }
    console.log('='.repeat(60) + '\n');
    
    return passed === total;
};

const success = verifyFrontendStructure();
process.exit(success ? 0 : 1);
