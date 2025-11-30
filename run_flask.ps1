# Run Flask app with Python 3.12.10
Write-Host "Starting Flask Web Server..." -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install flask opencv-python werkzeug --quiet

# Run Flask app
Write-Host "Launching Flask app..." -ForegroundColor Green
python flask_app.py
