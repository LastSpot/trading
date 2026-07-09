# Platform: Windows — run: .\setup.ps1  (or setup.cmd if scripts are blocked)
# If .\setup.ps1 fails with "running scripts is disabled on this system", run:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then re-run: .\setup.ps1
#
# Or run once without changing policy:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
#
#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Write-Info($Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn($Message) {
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Err($Message) {
    Write-Host "[X] $Message" -ForegroundColor Red
}

function Write-Step($Message) {
    Write-Host ""
    Write-Host "-- $Message --" -ForegroundColor White
}

function Get-StatusLabel($Status) {
    switch ($Status) {
        "ok" { return "done" }
        "partial" { return "partial (some steps skipped)" }
        default { return "skipped" }
    }
}

$pythonStatus = "skipped"

function Setup-Poetry {
    Write-Step "Python environment (Poetry)"

    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Err "python not found on PATH"
        return $false
    }
    Write-Info "Found python: $($python.Source)"

    $poetry = Get-Command poetry -ErrorAction SilentlyContinue
    if (-not $poetry) {
        Write-Info "Installing latest stable Poetry..."
        try {
            $installer = (Invoke-WebRequest -Uri "https://install.python-poetry.org" -UseBasicParsing).Content
            $installer | python -
        }
        catch {
            Write-Warn "Official Poetry installer failed; trying pipx..."
            $pipx = Get-Command pipx -ErrorAction SilentlyContinue
            if (-not $pipx) {
                Write-Err "Could not install Poetry. Install pipx or Poetry manually."
                return $false
            }
            pipx install poetry
        }

        $poetryScripts = Join-Path $env:APPDATA "Python\Scripts"
        if (Test-Path $poetryScripts) {
            $env:PATH = $poetryScripts + ";" + $env:PATH
        }
        $localBin = Join-Path $env:USERPROFILE ".local\bin"
        if (Test-Path $localBin) {
            $env:PATH = $localBin + ";" + $env:PATH
        }
    }
    else {
        Write-Info "Poetry already installed: $($poetry.Source)"
    }

    $poetry = Get-Command poetry -ErrorAction SilentlyContinue
    if (-not $poetry) {
        Write-Err "Poetry is not available on PATH after install"
        Write-Warn "Try adding %APPDATA%\Python\Scripts to PATH, then re-run."
        return $false
    }

    poetry config virtualenvs.in-project true --local

    $pyprojectPath = Join-Path $ScriptDir "pyproject.toml"
    if (-not (Test-Path $pyprojectPath)) {
        $defaultName = Split-Path -Leaf $ScriptDir
        $projectName = Read-Host "Project name [$defaultName]"
        if ([string]::IsNullOrWhiteSpace($projectName)) {
            $projectName = $defaultName
        }

        Write-Info "Initializing Poetry project: $projectName"
        poetry init `
            --name $projectName `
            --python "^3.11" `
            --dependency "numpy:*" `
            --dependency "pandas:*" `
            --dependency "yfinance:*" `
            --no-interaction

        if (Test-Path $pyprojectPath) {
            $content = Get-Content $pyprojectPath -Raw
            if ($content -notmatch 'package-mode\s*=\s*false') {
                if ($content -match '\[tool\.poetry\]') {
                    $replacement = '[tool.poetry]' + [Environment]::NewLine + 'package-mode = false'
                    $content = $content -replace '\[tool\.poetry\]', $replacement
                }
                else {
                    $toolPoetry = [Environment]::NewLine + '[tool.poetry]' + [Environment]::NewLine + 'package-mode = false' + [Environment]::NewLine
                    if ($content -match '\[build-system\]') {
                        $content = $content -replace '\[build-system\]', ($toolPoetry + '[build-system]')
                    }
                    else {
                        $content = $content.TrimEnd() + $toolPoetry
                    }
                }
                Set-Content $pyprojectPath -Value $content -NoNewline
                Write-Info "Set package-mode = false in pyproject.toml"
            }
            else {
                Write-Info "package-mode already set to false"
            }
        }
        else {
            Write-Err "poetry init did not create pyproject.toml"
            return $false
        }
    }
    else {
        Write-Info "Found existing pyproject.toml - skipping init"
    }

    Write-Info "Installing dependencies..."
    poetry install

    Write-Info "Activate with: poetry shell"
    Write-Info "Run backtest: poetry run python research\v2_trend_trio\v2_research.py"
    return $true
}

if (Setup-Poetry) {
    $pythonStatus = "ok"
}
else {
    Write-Warn "Poetry setup failed"
}

Write-Step "Setup complete"
Write-Host ""
Write-Host "  Poetry: $(Get-StatusLabel $pythonStatus)"
Write-Host ""
