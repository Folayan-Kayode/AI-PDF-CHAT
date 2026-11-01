<#
.SYNOPSIS
Start the API and the Streamlit UI with one shared API key.

.DESCRIPTION
Reads API_KEY from the repo-root .env, exports it as both API_KEY (for the
backend) and BACKEND_API_KEY (for the frontend), starts uvicorn with exactly
one worker, waits for /health, then runs Streamlit in the foreground. Stopping
Streamlit (Ctrl+C) stops the API too.

This exists because the two processes read the same secret from two different
variable names. Doing it by hand is easy to get wrong, and the symptom is a
401 that looks like a network or configuration failure.

.PARAMETER Python
Python executable to use. Defaults to "python" on PATH; pass the stable venv's
python.exe if that is where the dependencies live.

.PARAMETER ApiPort
Backend port. Default 8000.

.PARAMETER UiPort
Streamlit port. Default 8501.

.EXAMPLE
.\scripts\dev.ps1 -Python C:\Users\Davel\Documents\Python\.venvs\pdfchat\Scripts\python.exe
#>
param(
    [string]$Python = "python",
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repo ".env"

if (-not (Test-Path $envPath)) {
    throw "No .env at $envPath. Copy .env.example to .env and add the three keys."
}

# Parse only what we need. Never echo values.
$values = @{}
Get-Content $envPath | ForEach-Object {
    if ($_ -match '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $values[$matches[1]] = $matches[2].Trim().Trim('"').Trim("'")
    }
}

if (-not $values.ContainsKey("API_KEY") -or -not $values["API_KEY"]) {
    $hint = 'python -c "import secrets; print(secrets.token_urlsafe(32))"'
    throw "API_KEY is missing from $envPath. Generate one with: $hint"
}

# One value, both names: the backend reads API_KEY, the frontend BACKEND_API_KEY.
$env:API_KEY = $values["API_KEY"]
$env:BACKEND_API_KEY = $values["API_KEY"]
$env:PYTHONUNBUFFERED = "1"

$pythonExe = (Get-Command $Python -ErrorAction Stop).Source

Write-Host "Starting API on http://127.0.0.1:$ApiPort (workers=1) ..."

$api = Start-Process -FilePath $pythonExe `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", $ApiPort, "--workers", "1") `
    -WorkingDirectory (Join-Path $repo "backend") `
    -PassThru `
    -NoNewWindow

try {
    $deadline = (Get-Date).AddSeconds(45)
    $healthy = $false

    while ((Get-Date) -lt $deadline) {
        if ($api.HasExited) {
            throw "The API exited during startup (exit code $($api.ExitCode)). Read the output above."
        }

        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                $healthy = $true
                break
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $healthy) {
        throw "The API did not become healthy on port $ApiPort within 45 seconds."
    }

    Write-Host "API ready. Starting UI on http://127.0.0.1:$UiPort ..."

    & $pythonExe -m streamlit run (Join-Path $repo "frontend\app.py") `
        --server.address 127.0.0.1 `
        --server.port $UiPort
}
finally {
    if ($api -and -not $api.HasExited) {
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
}
