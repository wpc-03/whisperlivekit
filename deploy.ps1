# WhisperLiveKit Docker Deploy Script (Windows PowerShell)

Write-Host "=== WhisperLiveKit Docker Deploy Script ===" -ForegroundColor Green
Write-Host ""

# Check Docker installation
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Docker not installed" -ForegroundColor Red
    Write-Host "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
}

# Check Docker Compose installation
if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Docker Compose not installed" -ForegroundColor Red
    Write-Host "Please install Docker Compose"
    exit 1
}

# Show menu
Write-Host "Please select deploy option:" -ForegroundColor Yellow
Write-Host "1) Development (CPU)"
Write-Host "2) Production (GPU)"
Write-Host "3) Build image only"
Write-Host "4) Stop services"
Write-Host "5) View logs"
Write-Host "6) Clean data"
Write-Host ""

$choice = Read-Host "Enter option (1-6)"

switch ($choice) {
    "1" {
        Write-Host "Deploying development environment (CPU)..." -ForegroundColor Green
        docker-compose -f docker-compose.yml up -d --build
        Write-Host "Deploy completed!" -ForegroundColor Green
        Write-Host "Access: http://localhost:8000"
    }
    "2" {
        Write-Host "Deploying production environment (GPU)..." -ForegroundColor Green
        
        # Check NVIDIA Docker
        $nvidiaCheck = docker info 2>&1 | Select-String "nvidia"
        if (!$nvidiaCheck) {
            Write-Host "Warning: NVIDIA Docker runtime not detected" -ForegroundColor Yellow
            Write-Host "For GPU support, please install NVIDIA Container Toolkit"
            $continue = Read-Host "Continue with CPU mode? (y/n)"
            if ($continue -ne "y") {
                exit 1
            }
        }
        
        docker-compose -f docker-compose.prod.yml up -d --build
        Write-Host "Deploy completed!" -ForegroundColor Green
        Write-Host "Access: http://localhost"
    }
    "3" {
        Write-Host "Building image..." -ForegroundColor Green
        docker build -t whisperlivekit:latest .
        Write-Host "Image build completed!" -ForegroundColor Green
        docker images | Select-String "whisperlivekit"
    }
    "4" {
        Write-Host "Stopping services..." -ForegroundColor Yellow
        docker-compose -f docker-compose.yml down
        docker-compose -f docker-compose.prod.yml down
        Write-Host "Services stopped" -ForegroundColor Green
    }
    "5" {
        Write-Host "Viewing logs..." -ForegroundColor Green
        docker logs -f whisper-asr
    }
    "6" {
        Write-Host "Warning: This will delete all data including model cache!" -ForegroundColor Red
        $confirm = Read-Host "Are you sure? (y/n)"
        if ($confirm -eq "y") {
            docker-compose -f docker-compose.yml down -v
            docker-compose -f docker-compose.prod.yml down -v
            docker volume rm whisperlivekit_model-cache 2>$null
            Write-Host "Data cleaned" -ForegroundColor Green
        }
    }
    default {
        Write-Host "Invalid option" -ForegroundColor Red
        exit 1
    }
}
