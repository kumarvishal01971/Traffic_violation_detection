"""
Flask Web Application for Two-Wheeler Traffic Violation Detection
Matching UI with Four-Wheeler System
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import os
import sys

# Add the current directory to path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from your existing app.py
from app import SimplifiedTrafficViolationDetector, TESSERACT_AVAILABLE, EASYOCR_AVAILABLE
import sqlite3
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'two-wheeler-traffic-system-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['EVIDENCE_FOLDER'] = 'violation_evidence'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Create folders
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['EVIDENCE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
def init_database():
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  plate_number TEXT,
                  violation_type TEXT,
                  confidence REAL,
                  details TEXT,
                  evidence_file TEXT,
                  video_source TEXT)''')
    conn.commit()
    conn.close()

init_database()

def save_violation_to_db(violation, video_source):
    """Save violation to database"""    
    conn = sqlite3.connect('violations.db')
    c = conn.cursor()
    c.execute('''INSERT INTO violations 
                 (timestamp, plate_number, violation_type, confidence, details, evidence_file, video_source)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               violation.get('plate_number', 'UNKNOWN'),
               violation['type'],
               violation.get('confidence', 0.0),
               violation.get('details', ''),
               violation.get('evidence_file', ''),
               video_source))
    conn.commit()
    conn.close()

def get_all_violations():
    """Get all violations from database"""
    conn = sqlite3.connect('violations.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM violations ORDER BY timestamp DESC')
    violations = [dict(row) for row in c.fetchall()]
    conn.close()
    return violations

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', 
                         TESSERACT_AVAILABLE=TESSERACT_AVAILABLE,
                         EASYOCR_AVAILABLE=EASYOCR_AVAILABLE,
                         MODELS_AVAILABLE=True)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        
        if not file.filename or file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the file
            try:
                detector = SimplifiedTrafficViolationDetector()
                
                # Determine if image or video
                ext = filename.rsplit('.', 1)[1].lower()
                
                if ext in ['jpg', 'jpeg', 'png']:
                    # Process image
                    output_filename = 'output_' + filename
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    report = detector.process_image(filepath, output_path)
                else:
                    # Process video
                    output_filename = 'output_' + filename.rsplit('.', 1)[0] + '.mp4'
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    report = detector.process_video(filepath, output_path)
                
                if report:
                    # Save violations to database
                    for violation in report['violations']:
                        save_violation_to_db(violation, filename)
                    
                    # Extract plates and violation breakdown
                    plates = list(set([v['plate_number'] for v in report['violations'] if v['plate_number'] != 'UNKNOWN']))
                    violation_breakdown = defaultdict(int)
                    for v in report['violations']:
                        violation_breakdown[v['type']] += 1
                    
                    flash('Processing completed successfully!')
                    return render_template('result.html',
                                         total_violations=report['total_violations'],
                                         plates=plates,
                                         output_filename=output_filename,
                                         violation_breakdown=dict(violation_breakdown))
                else:
                    flash('Error processing file')
                    return redirect(url_for('upload'))
                    
            except Exception as e:
                flash(f'Error: {str(e)}')
                return redirect(url_for('upload'))
        else:
            flash('Invalid file format')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/violations')
def violations():
    """View all violations"""
    all_violations = get_all_violations()
    return render_template('violations.html', violations=all_violations)

@app.route('/demo')
def demo():
    """Demo page"""
    return render_template('demo.html')

@app.route('/run_demo', methods=['POST'])
def run_demo():
    """Run demo with sample video"""
    # Check if demo video exists
    demo_video = 'demo_video.mp4'
    demo_path = os.path.join(app.config['UPLOAD_FOLDER'], demo_video)
    
    if not os.path.exists(demo_path):
        flash('Demo video not found. Please add demo_video.mp4 to uploads folder.')
        return redirect(url_for('demo'))
    
    try:
        detector = SimplifiedTrafficViolationDetector()
        output_filename = 'demo_output.mp4'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        report = detector.process_video(demo_path, output_path)
        
        if report:
            # Save violations to database
            for violation in report['violations']:
                save_violation_to_db(violation, demo_video)
            
            plates = list(set([v['plate_number'] for v in report['violations'] if v['plate_number'] != 'UNKNOWN']))
            violation_breakdown = defaultdict(int)
            for v in report['violations']:
                violation_breakdown[v['type']] += 1
            
            flash('Demo processing completed!')
            return render_template('result.html',
                                 total_violations=report['total_violations'],
                                 plates=plates,
                                 output_filename=output_filename,
                                 violation_breakdown=dict(violation_breakdown))
        else:
            flash('No violations detected in demo video.')
            return redirect(url_for('demo'))
    except Exception as e:
        flash(f'Error running demo: {str(e)}')
        return redirect(url_for('demo'))

@app.route('/download/<filename>')
def download(filename):
    """Download processed file"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    else:
        flash('File not found')
        return redirect(url_for('index'))

@app.route('/evidence/<filename>')
def evidence(filename):
    """View evidence image"""
    filepath = os.path.join(app.config['EVIDENCE_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg')
    else:
        flash('Evidence file not found')
        return redirect(url_for('violations'))

@app.route('/clear_database', methods=['POST'])
def clear_database():
    """Clear all violations from database"""
    try:
        conn = sqlite3.connect('violations.db')
        c = conn.cursor()
        c.execute('DELETE FROM violations')
        conn.commit()
        conn.close()
        flash('Database cleared successfully')
    except Exception as e:
        flash(f'Error clearing database: {str(e)}')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏍️  TWO-WHEELER TRAFFIC VIOLATION DETECTION SYSTEM")
    print("="*70)
    print("\n✅ Server starting...")
    print("📍 Access the application at: http://127.0.0.1:3000")
    print("📁 Make sure 'templates' folder contains all HTML files")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=3000)