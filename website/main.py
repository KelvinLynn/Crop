from flask import Flask, redirect, url_for, render_template, request, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
import logging
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import bcrypt

# Cấu hình logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'Teams')

# Cấu hình Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

# Cấu hình kết nối MySQL
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Mặc định XAMPP để trống
    'database': 'crops_db'
}

# Hàm kết nối MySQL
def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        logger.error(f"Lỗi kết nối MySQL: {e}")
        return None

# Load model và các file liên quan
try:
    model_path = os.path.abspath('../data/model.pkl')
    label_encoder_path = os.path.abspath('../data/label_encoder.pkl')
    scaler_path = os.path.abspath('../data/scaler.pkl')
    crop_names_path = os.path.abspath('../data/crop_names.pkl')

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    scaler = joblib.load(scaler_path)
    crop_names = joblib.load(crop_names_path)

    CROP_NAMES = {i: name.capitalize() for i, name in enumerate(crop_names)}
    logger.info(f"CROP_NAMES mapping: {CROP_NAMES}")

except Exception as e:
    logger.error(f"Error loading files: {str(e)}")
    raise

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    connection = get_db_connection()
    if connection is None:
        return None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user:
            return User(user['username'])
        return None
    finally:
        cursor.close()
        connection.close()

def authenticate_user(username, password):
    connection = get_db_connection()
    if connection is None:
        return None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT username, password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return User(user['username'])
        return None
    finally:
        cursor.close()
        connection.close()

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        session['login_attempts'] = session.get('login_attempts', 0)
        if session['login_attempts'] >= 3:
            return render_template('index.html', error="Quá nhiều lần thử đăng nhập. Vui lòng thử lại sau 5 phút.")

        user = authenticate_user(username, password)
        if user:
            login_user(user)
            session['username'] = username
            session['isLoggedIn'] = True
            session['login_attempts'] = 0
            logger.info(f"Successful login - Username: {username}, Session: {session}")
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('home'))
        else:
            session['login_attempts'] += 1
            logger.warning(f"Failed login attempt - Username: {username}, Attempts: {session['login_attempts']}")
            return render_template('index.html', error="Tên đăng nhập hoặc mật khẩu không đúng!")
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validate input
        if not username or not password or not confirm_password:
            return render_template('register.html', error="Vui lòng điền đầy đủ thông tin.")

        if password != confirm_password:
            return render_template('register.html', error="Mật khẩu xác nhận không khớp.")

        # Check if username already exists
        connection = get_db_connection()
        if connection is None:
            return render_template('register.html', error="Không thể kết nối đến cơ sở dữ liệu.")
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT username FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return render_template('register.html', error="Tên đăng nhập đã tồn tại.")

            # Hash password and insert user into MySQL
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password.decode('utf-8')))
            connection.commit()
            logger.info(f"User registered - Username: {username}")
            flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
            return redirect(url_for('index'))
        except Error as e:
            logger.error(f"Error during registration: {e}")
            return render_template('register.html', error=f"Đã xảy ra lỗi: {e}")
        finally:
            cursor.close()
            connection.close()

    return render_template('register.html')

@app.route('/home')
@login_required
def home():
    logger.info(f"Accessing home - Session: {session}")
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        try:
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosphorous'])
            K = float(request.form['Potassium'])
            Temperature = float(request.form['Temperature'])
            Humidity = float(request.form['Humidity'])
            PH = float(request.form['PH'])
            Rainfall = float(request.form['Rainfall'])

            if not (1 <= N <= 200 and 1 <= P <= 200 and 1 <= K <= 200):
                return render_template('home.html', error="N, P, K phải nằm trong khoảng 1-200.")
            if not (1 <= Temperature <= 65):
                return render_template('home.html', error="Nhiệt độ phải nằm trong khoảng 1-65°C.")
            if not (1 <= Humidity <= 100):
                return render_template('home.html', error="Độ ẩm phải nằm trong khoảng 1-100%.")
            if not (1 <= PH <= 14):
                return render_template('home.html', error="pH phải nằm trong khoảng 1-14.")
            if Rainfall < 0:
                return render_template('home.html', error="Lượng mưa không thể âm.")

            input_data = pd.DataFrame(
                [[N, P, K, Temperature, Humidity, PH, Rainfall]],
                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            )
            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(input_data_scaled)
            predicted_class = int(prediction[0])
            crop_name = CROP_NAMES.get(predicted_class, f"Unknown label: {predicted_class}")

            probabilities = []
            if isinstance(model, KNeighborsClassifier):
                n_neighbors = min(200, len(model._y))
                distances, indices = model.kneighbors(input_data_scaled, n_neighbors=n_neighbors)
                neighbor_labels = model._y[indices[0]]
                distances = distances[0]

                inverse_distances = 1 / (distances + 1e-5)
                scores = np.zeros(len(CROP_NAMES))
                for label, inv_dist in zip(neighbor_labels, inverse_distances):
                    scores[label] += inv_dist

                baseline_scores = np.zeros(len(CROP_NAMES))
                for i in range(len(CROP_NAMES)):
                    class_indices = np.where(model._y == i)[0]
                    if len(class_indices) > 0:
                        class_distances = np.mean(model._fit_X[class_indices], axis=0)
                        baseline_scores[i] = 1 / (np.linalg.norm(input_data_scaled - class_distances) + 1e-5)
                baseline_scores = baseline_scores / np.max(baseline_scores) * 5
                scores += baseline_scores

                max_score = np.max(scores)
                if max_score > 0:
                    percentages = (scores / max_score) * 200
                else:
                    percentages = np.zeros(len(CROP_NAMES))
                percentages = np.clip(percentages, 0, 100)

                for i in range(len(CROP_NAMES)):
                    probabilities.append({
                        'label': i,
                        'name': CROP_NAMES.get(i, str(i)),
                        'probability': round(percentages[i], 2)
                    })
                probabilities = sorted(probabilities, key=lambda x: x['probability'], reverse=True)

            humidity_level = 'Low Humid' if 1 <= Humidity <= 33 else 'Medium Humid' if 34 <= Humidity <= 66 else 'High Humid'
            temperature_level = 'Cool' if 0 <= Temperature <= 15 else 'Warm' if 16 <= Temperature <= 30 else 'Hot'
            rainfall_level = 'Less' if 1 <= Rainfall <= 100 else 'Moderate' if 101 <= Rainfall <= 200 else 'Heavy Rain'
            N_level = 'Less' if 1 <= N <= 50 else 'Moderate' if 51 <= N <= 100 else 'High'
            P_level = 'Less' if 1 <= P <= 50 else 'Moderate' if 51 <= P <= 100 else 'High'
            potassium_level = 'Less' if 1 <= K <= 50 else 'Moderate' if 51 <= K <= 100 else 'High'
            phlevel = 'Acidic' if 0 <= PH <= 5 else 'Neutral' if 6 <= PH <= 8 else 'Alkaline'

            values = [N, P, K, Humidity, Temperature, Rainfall, PH]
            cont = [N_level, P_level, potassium_level, humidity_level, temperature_level, rainfall_level, phlevel]
            return render_template('Display.html', 
                                 cont=cont, 
                                 values=values, 
                                 cropName=crop_name, 
                                 predictedLabel=predicted_class, 
                                 probabilities=probabilities)
        except ValueError:
            return render_template('home.html', error="Vui lòng nhập số hợp lệ.")
        except Exception as e:
            logger.error(f"Đã xảy ra lỗi trong dự đoán: {str(e)}")
            return render_template('home.html', error=f"Đã xảy ra lỗi: {str(e)}")
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/crops_info')
def crops_info():
    connection = get_db_connection()
    if connection is None:
        return render_template('crops_info.html', error="Không thể kết nối đến cơ sở dữ liệu.")

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, name, N, P, K, temperature, humidity, ph, rainfall, image FROM crops")
        crops = cursor.fetchall()

        cursor.execute("SELECT id, name, type_id, image, description FROM crop_varieties")
        crops_info = cursor.fetchall()

        for variety in crops_info:
            for crop in crops:
                if crop['id'] == variety['type_id']:
                    variety['type'] = crop['name']
                    break

        selected_crop_id = request.args.get('crop_id', type=int)
        selected_crop = None
        if selected_crop_id:
            for crop in crops:
                if crop['id'] == selected_crop_id:
                    selected_crop = crop
                    break

        return render_template('crops_info.html', crops=crops, crops_info=crops_info, selected_crop=selected_crop)
    except Error as e:
        logger.error(f"Lỗi khi truy vấn dữ liệu: {e}")
        return render_template('crops_info.html', error=f"Lỗi khi truy vấn dữ liệu: {e}")
    finally:
        cursor.close()
        connection.close()

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    connection = get_db_connection()
    if connection is None:
        flash("Không thể kết nối đến cơ sở dữ liệu.", "danger")
        return render_template('contact.html')

    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            phone = request.form.get('phone') or None
            message = request.form['message']
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Thêm thời gian hiện tại

            cursor = connection.cursor()
            sql = "INSERT INTO contacts (name, email, phone, message, created_at) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql, (name, email, phone, message, created_at))
            connection.commit()

            flash("Gửi tin nhắn thành công!", "success")
            return redirect(url_for('contact'))
        except Error as e:
            logger.error(f"Contact form error: {e}")
            flash(f"Đã xảy ra lỗi: {e}", "danger")
            return redirect(url_for('contact'))
        finally:
            cursor.close()
            connection.close()
    return render_template('contact.html')

@app.route('/user/<usr>')
def user(usr):
    return f"<h1>Hi {usr}!</h1>"

@app.route('/detail/<int:id>')
def detail(id):
    connection = get_db_connection()
    if connection is None:
        return render_template('detail.html', error="Không thể kết nối đến cơ sở dữ liệu.")

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, name, type_id, image, description FROM crop_varieties WHERE id = %s", (id,))
        crop = cursor.fetchone()

        if not crop:
            return render_template('detail.html', error="Không tìm thấy giống cây trồng.")

        cursor.execute("SELECT name FROM crops WHERE id = %s", (crop['type_id'],))
        crop_type = cursor.fetchone()
        crop['type'] = crop_type['name'] if crop_type else "Unknown"

        cursor.execute("SELECT id, name, type_id, image, description FROM crop_varieties WHERE type_id = %s AND id != %s", (crop['type_id'], id))
        related_crops = cursor.fetchall()

        for related in related_crops:
            related['type'] = crop['type']

        return render_template('detail.html', 
                             namer=crop['name'], 
                             typerr=crop['type'], 
                             imager=crop['image'], 
                             descriptionr=crop['description'], 
                             listr=related_crops)
    except Error as e:
        logger.error(f"Lỗi khi truy vấn chi tiết: {e}")
        return render_template('detail.html', error=f"Lỗi khi truy vấn chi tiết: {e}")
    finally:
        cursor.close()
        connection.close()

@app.route('/logout')
@login_required
def logout():
    logger.info(f"Logging out - Session before: {session}")
    logout_user()
    session.pop('username', None)
    session.pop('login_attempts', None)
    logger.info(f"Logged out - Session after: {session}")
    flash('Bạn đã đăng xuất thành công!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)