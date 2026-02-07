# Quantum Tracker Setup Script
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "QUANTUM TRACKER - AUTOMATED SETUP" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"

# Navigate to backend
Write-Host "`n[BACKEND SETUP]" -ForegroundColor Yellow
Write-Host "Setting up Python backend...`n" -ForegroundColor Cyan

Set-Location backend

# Check Python
Write-Host "Checking Python..." -ForegroundColor Cyan
python --version

# Create venv
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate and install
Write-Host "Installing dependencies (this takes 5-10 minutes)..." -ForegroundColor Cyan
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\python.exe -m pip install -r requirements.txt

# Create .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env file" -ForegroundColor Green
}

Write-Host "`n✓ Backend setup complete!`n" -ForegroundColor Green

# Return to root
Set-Location ..

# Frontend setup
Write-Host "`n[FRONTEND SETUP]" -ForegroundColor Yellow
Write-Host "Setting up React frontend...`n" -ForegroundColor Cyan

Set-Location frontend

# Check Node
Write-Host "Checking Node.js..." -ForegroundColor Cyan
node --version
npm --version

# Create .env
if (-not (Test-Path ".env")) {
    "VITE_API_URL=http://localhost:5000" | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "Created .env file" -ForegroundColor Green
}

# Install dependencies
Write-Host "Installing dependencies (this takes 3-5 minutes)..." -ForegroundColor Cyan
npm install

Write-Host "`n✓ Frontend setup complete!`n" -ForegroundColor Green

# Return to root
Set-Location ..

# Final message
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Green

Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "`n1. Start Backend (Terminal 1):" -ForegroundColor Yellow
Write-Host "   cd backend" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   python app.py" -ForegroundColor White

Write-Host "`n2. Start Frontend (Terminal 2):" -ForegroundColor Yellow
Write-Host "   cd frontend" -ForegroundColor White
Write-Host "   npm run dev" -ForegroundColor White

Write-Host "`n3. Open: http://localhost:5173`n" -ForegroundColor Cyan
