import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
import cv2
import random
import string

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f'Face {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        face_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'face_{i+1}')
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
        
        face_image_path = os.path.join(face_dir, secure_filename(os.path.basename(image_path)))
        cv2.imwrite(face_image_path, image[y:y+h, x:x+w])
    
    cv2.imwrite(output_path, image)
    return faces

@app.route('/')
def index():
    uploaded_files = [f for f in os.listdir('uploads') if not f.startswith('processed_')]
    processed_files = [f for f in os.listdir('uploads') if f.startswith('processed_')]
    return render_template('index.html', uploaded_files=uploaded_files, processed_files=processed_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('file[]')
    for file in files:
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
            file.save(file_path)
            detect_faces(file_path, output_path)
    
    flash('Files successfully uploaded and processed.')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
