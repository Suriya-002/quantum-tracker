# Automated Backend Setup Script
Write-Host "
============================================================" -ForegroundColor Cyan
Write-Host "QUANTUM TRACKER - AUTOMATED BACKEND SETUP" -ForegroundColor Cyan
Write-Host "============================================================
" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"

try {
    # Step 1: Navigate to backend
    Write-Host "[1/7] Navigating to backend directory..." -ForegroundColor Yellow
    Set-Location backend
    Write-Host "✓ In backend directory
" -ForegroundColor Green

    # Step 2: Check if Python is available
    Write-Host "[2/7] Checking Python installation..." -ForegroundColor Yellow
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Python found: $pythonVersion
" -ForegroundColor Green
    } catch {
        Write-Host "✗ Python not found! Please install Python 3.8+ first." -ForegroundColor Red
        exit 1
    }

    # Step 3: Create virtual environment
    Write-Host "[3/7] Creating virtual environment..." -ForegroundColor Yellow
    if (Test-Path "venv") {
        Write-Host "  Virtual environment already exists, skipping..." -ForegroundColor Cyan
    } else {
        python -m venv venv
        Write-Host "✓ Virtual environment created
" -ForegroundColor Green
    }

    # Step 4: Activate virtual environment
    Write-Host "[4/7] Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated
" -ForegroundColor Green

    # Step 5: Upgrade pip
    Write-Host "[5/7] Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip --quiet
    Write-Host "✓ Pip upgraded
" -ForegroundColor Green

    # Step 6: Install dependencies
    Write-Host "[6/7] Installing Python dependencies..." -ForegroundColor Yellow
    Write-Host "  This may take 5-10 minutes, please wait..." -ForegroundColor Cyan
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed
" -ForegroundColor Green

    # Step 7: Create .env file
    Write-Host "[7/7] Setting up environment file..." -ForegroundColor Yellow
    if (Test-Path ".env") {
        Write-Host "  .env file already exists, skipping..." -ForegroundColor Cyan
    } else {
        Copy-Item ".env.example" ".env"
        Write-Host "✓ .env file created
" -ForegroundColor Green
    }

    # Success message
    Write-Host "
============================================================" -ForegroundColor Green
    Write-Host "✅ BACKEND SETUP COMPLETE!" -ForegroundColor Green
    Write-Host "============================================================
" -ForegroundColor Green

    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Review/edit .env file if needed" -ForegroundColor White
    Write-Host "2. Start the backend server:
" -ForegroundColor White
    Write-Host "   python app.py
" -ForegroundColor Yellow

    Write-Host "The backend will be available at: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "
Would you like to start the backend now? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host

    if ($response -eq "Y" -or $response -eq "y") {
        Write-Host "
Starting backend server..." -ForegroundColor Cyan
        Write-Host "Press Ctrl+C to stop the server
" -ForegroundColor Yellow
        python app.py
    } else {
        Write-Host "
To start the backend later, run:" -ForegroundColor Cyan
        Write-Host "  cd backend" -ForegroundColor White
        Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
        Write-Host "  python app.py
" -ForegroundColor White
    }

} catch {
    Write-Host "
✗ Setup failed: $_" -ForegroundColor Red
    Write-Host "
Please check the error above and try again." -ForegroundColor Yellow
    exit 1
}
