# Start Development Servers
Write-Host "Starting Quantum Tracker Development Servers..." -ForegroundColor Cyan

# Start backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Suriya\Documents\UF Class\Research\Philip Feng\quantum-tracker\backend'; .\venv\Scripts\Activate.ps1; python app.py"

# Wait a bit for backend to start
Start-Sleep -Seconds 3

# Start frontend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'C:\Users\Suriya\Documents\UF Class\Research\Philip Feng\quantum-tracker\frontend'; npm run dev"

Write-Host "
✅ Development servers starting..." -ForegroundColor Green
Write-Host "Backend:  http://localhost:5000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
