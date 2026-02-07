# Comprehensive Project Verification Script
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "QUANTUM TRACKER - PROJECT VERIFICATION" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$global:totalChecks = 0
$global:passedChecks = 0

function Test-PathExists {
    param([string]$Path, [string]$Description)
    $global:totalChecks++
    if (Test-Path $Path) {
        Write-Host "✓" -ForegroundColor Green -NoNewline
        Write-Host " $Description" -ForegroundColor White
        $global:passedChecks++
        return $true
    } else {
        Write-Host "✗" -ForegroundColor Red -NoNewline
        Write-Host " $Description" -ForegroundColor White
        return $false
    }
}

# Root Level
Write-Host "`n📦 ROOT LEVEL FILES:" -ForegroundColor Yellow
Test-PathExists "README.md" "README file"
Test-PathExists ".gitignore" "Git ignore file"
Test-PathExists "docker-compose.yml" "Docker compose file"

# Backend
Write-Host "`n🐍 BACKEND STRUCTURE:" -ForegroundColor Yellow
Test-PathExists "backend" "Backend directory"
Test-PathExists "backend/requirements.txt" "Requirements file"
Test-PathExists "backend/config.py" "Config file"
Test-PathExists "backend/app.py" "Main app file"
Test-PathExists "backend/.env.example" "Environment example"

Write-Host "`n🔧 BACKEND - Services:" -ForegroundColor Cyan
Test-PathExists "backend/src/services/rss_scraper.py" "RSS scraper service"
Test-PathExists "backend/src/services/stock_fetcher.py" "Stock fetcher service"
Test-PathExists "backend/src/services/sentiment_analyzer.py" "Sentiment analyzer"
Test-PathExists "backend/src/services/news_aggregator.py" "News aggregator"

Write-Host "`n🛣️  BACKEND - Routes:" -ForegroundColor Cyan
Test-PathExists "backend/src/routes/news.py" "News routes"
Test-PathExists "backend/src/routes/stocks.py" "Stocks routes"
Test-PathExists "backend/src/routes/companies.py" "Companies routes"
Test-PathExists "backend/src/routes/sentiment.py" "Sentiment routes"

Write-Host "`n🗄️  BACKEND - Database:" -ForegroundColor Cyan
Test-PathExists "backend/src/utils/database.py" "Database utility"
Test-PathExists "backend/src/utils/logger.py" "Logger utility"
Test-PathExists "backend/src/utils/scheduler.py" "Scheduler utility"

Write-Host "`n📊 BACKEND - Data Files:" -ForegroundColor Cyan
Test-PathExists "backend/src/data/quantum_companies.json" "Companies data"
Test-PathExists "backend/src/data/rss_feeds.json" "RSS feeds data"

# Frontend
Write-Host "`n⚛️  FRONTEND STRUCTURE:" -ForegroundColor Yellow
Test-PathExists "frontend" "Frontend directory"
Test-PathExists "frontend/package.json" "Package.json"
Test-PathExists "frontend/vite.config.js" "Vite config"
Test-PathExists "frontend/index.html" "Index HTML"

Write-Host "`n🎨 FRONTEND - Components:" -ForegroundColor Cyan
Test-PathExists "frontend/src/components/layout" "Layout components"
Test-PathExists "frontend/src/components/news" "News components"
Test-PathExists "frontend/src/components/stocks" "Stock components"
Test-PathExists "frontend/src/components/companies" "Company components"
Test-PathExists "frontend/src/components/common" "Common components"

Write-Host "`n📄 FRONTEND - Pages:" -ForegroundColor Cyan
Test-PathExists "frontend/src/pages/Dashboard.jsx" "Dashboard page"
Test-PathExists "frontend/src/pages/NewsPage.jsx" "News page"
Test-PathExists "frontend/src/pages/StocksPage.jsx" "Stocks page"
Test-PathExists "frontend/src/pages/CompaniesPage.jsx" "Companies page"

Write-Host "`n🔌 FRONTEND - Services:" -ForegroundColor Cyan
Test-PathExists "frontend/src/services/api.js" "API service"
Test-PathExists "frontend/src/services/newsService.js" "News service"
Test-PathExists "frontend/src/services/stockService.js" "Stock service"
Test-PathExists "frontend/src/services/companyService.js" "Company service"

# Database
Write-Host "`n🗃️  DATABASE:" -ForegroundColor Yellow
Test-PathExists "database" "Database directory"
Test-PathExists "database/schema.sql" "Database schema"

# Scripts
Write-Host "`n📜 SCRIPTS:" -ForegroundColor Yellow
Test-PathExists "scripts" "Scripts directory"

# Docs
Write-Host "`n📚 DOCUMENTATION:" -ForegroundColor Yellow
Test-PathExists "docs" "Docs directory"

# Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "VERIFICATION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$percentage = [math]::Round(($global:passedChecks / $global:totalChecks) * 100, 2)

Write-Host "`nTotal Checks: " -NoNewline
Write-Host $global:totalChecks -ForegroundColor White

Write-Host "Passed: " -NoNewline
Write-Host $global:passedChecks -ForegroundColor Green

Write-Host "Failed: " -NoNewline
Write-Host ($global:totalChecks - $global:passedChecks) -ForegroundColor Red

Write-Host "Success Rate: " -NoNewline
if ($percentage -eq 100) {
    Write-Host "$percentage%" -ForegroundColor Green
} elseif ($percentage -ge 80) {
    Write-Host "$percentage%" -ForegroundColor Yellow
} else {
    Write-Host "$percentage%" -ForegroundColor Red
}

if ($global:passedChecks -eq $global:totalChecks) {
    Write-Host "`n✅ ALL CHECKS PASSED! Project structure is complete." -ForegroundColor Green
} else {
    Write-Host "`n⚠️  SOME CHECKS FAILED! Please review missing files." -ForegroundColor Yellow
}

Write-Host "`n============================================================`n" -ForegroundColor Cyan
