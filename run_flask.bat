@echo off
REM Run Flask app with Python 3.12.10
echo Starting Flask Web Server...
echo.
python -m pip install flask opencv-python --quiet
python flask_app.py
pause
