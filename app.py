from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from ultralytics import YOLO
import cv2
import threading
from flask import Response


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'crowd.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------- Database Model --------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Create database
with app.app_context():
    db.create_all()
    # -------- Real-Time YOLO Tracking --------
model = YOLO("yolov8n.pt")  # or your trained model

# Global variable for current frame
current_frame = None

def process_frames(video_path):
    global current_frame
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        # Draw boxes and IDs
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"ID:{cls} {conf:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        current_frame = frame.copy()
    
    cap.release()


# -------- Routes --------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    username = session['user']
    return render_template('dashboard.html', username=username)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))
@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame
        while True:
            if current_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -------- Run App --------
if __name__ == '__main__':
    video_path = 0  # 0 for webcam, or "video.mp4" for file
    threading.Thread(target=process_frames, args=(video_path,), daemon=True).start()
    app.run(debug=True)

