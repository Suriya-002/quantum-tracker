# Windows Setup Script for Quantum Tracker
Write-Host "Setting up Quantum Tracker..." -ForegroundColor Cyan

# Backend setup
Write-Host "
1. Setting up Backend..." -ForegroundColor Yellow
Set-Location backend

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r requirements.txt

Set-Location ..

# Frontend setup
Write-Host "
2. Setting up Frontend..." -ForegroundColor Yellow
Set-Location frontend

Write-Host "Installing Node dependencies..." -ForegroundColor Cyan
npm install

Set-Location ..

Write-Host "
✅ Setup complete!" -ForegroundColor Green
Write-Host "
To start the application:" -ForegroundColor Cyan
Write-Host "  Backend:  cd backend && .\venv\Scripts\Activate.ps1 && python app.py" -ForegroundColor White
Write-Host "  Frontend: cd frontend && npm run dev" -ForegroundColor White
