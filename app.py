import os
import secrets
import sqlite3
import hashlib
import joblib
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, jsonify, session

app = Flask(__name__)
# Reads SECRET_KEY from the environment in production. Falls back to a
# random key generated at process start if it isn't set, so the app never
# ships with a hardcoded secret -- but note that admin sessions will be
# invalidated on every restart until SECRET_KEY is set in the environment.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
DB_NAME = "database.db"

# Cardholder Home Baseline (Accra, Ghana)
HOME_LAT = 5.6037
HOME_LNG = -0.1870

# Expanded Global Bank Issuer Coordinates Database (Lat, Lng)
ISSUER_LOCATIONS = {
    # Africa
    "Ecobank Ghana (Accra, GH)": (5.6037, -0.1870),
    "GCB Bank (Accra, GH)": (5.5471, -0.2012),
    "Fidelity Bank Ghana (Accra, GH)": (5.5560, -0.1969),
    "Stanbic Bank Ghana (Accra, GH)": (5.5600, -0.1920),
    "Access Bank (Lagos, NG)": (6.4549, 3.3887),
    "Guaranty Trust Bank (Lagos, NG)": (6.4474, 3.4233),
    "Standard Bank (Johannesburg, ZA)": (-26.2041, 28.0473),
    "Absa Bank (Johannesburg, ZA)": (-26.1952, 28.0341),
    "Equity Bank (Nairobi, KE)": (-1.2921, 36.8219),
    "KCB Bank (Nairobi, KE)": (-1.286389, 36.821944),
    
    # North America
    "JPMorgan Chase (New York, US)": (40.7128, -74.0060),
    "Bank of America (Charlotte, US)": (35.2271, -80.8431),
    "Wells Fargo (San Francisco, US)": (37.7749, -122.4194),
    "Citigroup (New York, US)": (40.7128, -74.0060),
    "Goldman Sachs (New York, US)": (40.7153, -74.0142),
    "Scotiabank (Toronto, CA)": (43.6532, -79.3832),
    "Royal Bank of Canada (Toronto, CA)": (43.6532, -79.3832),
    "TD Bank (Toronto, CA)": (43.6532, -79.3832),
    "National Bank of Canada (Montreal, CA)": (45.5017, -73.5673),

    # Europe & Nordics
    "HSBC (London, UK)": (51.5074, -0.1278),
    "Barclays Bank (London, UK)": (51.5074, -0.1278),
    "Lloyds Bank (London, UK)": (51.5155, -0.0922),
    "BNP Paribas (Paris, FR)": (48.8566, 2.3522),
    "Crédit Agricole (Paris, FR)": (48.8448, 2.3242),
    "Deutsche Bank (Frankfurt, DE)": (50.1109, 8.6821),
    "Commerzbank (Frankfurt, DE)": (50.1109, 8.6821),
    "Banco Santander (Madrid, ES)": (40.4168, -3.7038),
    "BBVA (Madrid, ES)": (40.4168, -3.7038),
    "UniCredit (Milan, IT)": (45.4642, 9.1900),
    "UBS Group (Zurich, CH)": (47.3769, 8.5417),
    "Nordea (Helsinki, FI)": (60.1699, 24.9384),
    "Danske Bank (Copenhagen, DK)": (55.6761, 12.5683),
    "SEB Group (Stockholm, SE)": (59.3293, 18.0686),

    # Asia-Pacific & South Asia
    "ICBC (Beijing, CN)": (39.9042, 116.4074),
    "China Construction Bank (Beijing, CN)": (39.9042, 116.4074),
    "Agricultural Bank of China (Beijing, CN)": (39.9042, 116.4074),
    "Bank of China (Beijing, CN)": (39.9042, 116.4074),
    "DBS Bank (Singapore, SG)": (1.3521, 103.8198),
    "OCBC Bank (Singapore, SG)": (1.2855, 103.8565),
    "MUFG Bank (Tokyo, JP)": (35.6762, 139.6503),
    "Sumitomo Mitsui Banking Corporation (Tokyo, JP)": (35.6895, 139.6917),
    "Commonwealth Bank (Sydney, AU)": (-33.8688, 151.2093),
    "National Australia Bank (Melbourne, AU)": (-37.8136, 144.9631),
    "State Bank of India (Mumbai, IN)": (18.9220, 72.8347),
    "HDFC Bank (Mumbai, IN)": (18.9389, 72.8258),
    "KB Financial Group (Seoul, KR)": (37.5665, 126.9780),

    # Latin America
    "Itaú Unibanco (São Paulo, BR)": (-23.5505, -46.6333),
    "Banco do Brasil (Brasília, BR)": (-15.7975, -47.8919),
    "Banco Bradesco (Osasco, BR)": (-23.5329, -46.7916),

    # Middle East
    "First Abu Dhabi Bank (Abu Dhabi, AE)": (24.4539, 54.3773),
    "Qatar National Bank (Doha, QA)": (25.2854, 51.5310)
}

def haversine(lat1, lon1, lat2, lon2):
    r = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * r * asin(sqrt(a))

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_masked TEXT NOT NULL,
            card_hash TEXT DEFAULT '',
            issuer_name TEXT NOT NULL,
            amount REAL NOT NULL,
            device_time TEXT NOT NULL,
            velocity_count INTEGER NOT NULL,
            home_distance_km REAL NOT NULL,
            issuer_distance_km REAL NOT NULL,
            device_ip TEXT DEFAULT '127.0.0.1',
            prediction TEXT NOT NULL,
            status TEXT DEFAULT 'Pending'
        )
    ''')
    # Migrate older databases that don't have card_hash yet
    cursor.execute("PRAGMA table_info(transactions)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if 'card_hash' not in existing_cols:
        cursor.execute("ALTER TABLE transactions ADD COLUMN card_hash TEXT DEFAULT ''")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flagged_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_value TEXT UNIQUE NOT NULL,
            reason TEXT NOT NULL,
            flagged_at TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    cursor.execute('SELECT COUNT(*) FROM admins')
    if cursor.fetchone()[0] == 0:
        default_pw = generate_password_hash('password123')
        cursor.execute('INSERT INTO admins (username, password_hash) VALUES (?, ?)', ('admin', default_pw))

    conn.commit()
    conn.close()

# Run once at import time so the tables exist no matter which route is hit
# first. Under gunicorn (or any WSGI server) the `if __name__ == '__main__'`
# block at the bottom never runs, so relying on that alone would leave the
# database uninitialized until someone happened to visit "/" first.
init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/', methods=['GET'])
def home():
    # Root URL now lands on the admin area: straight to the dashboard if
    # already signed in, otherwise the login page. The transaction
    # simulation form lives at /intake and is linked from there.
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('admin_login'))

@app.route('/intake', methods=['GET'])
def transaction_intake():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT id, card_masked, issuer_name, amount, device_time, velocity_count, home_distance_km, issuer_distance_km, device_ip, prediction, status FROM transactions ORDER BY id DESC')
    logs = cursor.fetchall()
    conn.close()
    return render_template('index.html', logs=logs, issuers=list(ISSUER_LOCATIONS.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model file missing. Run train_model.py first.", 500

    raw_card = request.form.get('card_number', '').replace(' ', '')
    issuer_name = request.form.get('issuer_name')
    amount = float(request.form.get('amount', 0))
    lat = float(request.form.get('latitude', HOME_LAT))
    lng = float(request.form.get('longitude', HOME_LNG))
    client_time_raw = request.form.get('client_time')

    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if client_ip and ',' in client_ip:
        client_ip = client_ip.split(',')[0].strip()

    last_four = raw_card[-4:] if len(raw_card) >= 4 else "0000"
    card_masked = f"•••• {last_four}"
    card_hash = hashlib.sha256(raw_card.encode()).hexdigest() if raw_card else ""

    try:
        dt = datetime.fromisoformat(client_time_raw.replace('Z', '+00:00'))
        formatted_device_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        transaction_hour = dt.hour
    except Exception:
        dt = datetime.now()
        formatted_device_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        transaction_hour = dt.hour

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM flagged_entities WHERE (entity_value = ? AND entity_type = "Card") OR (entity_value = ? AND entity_type = "IP")', (card_masked, client_ip))
    is_blacklisted = cursor.fetchone() is not None

    if is_blacklisted:
        fraud_proba = 1.0  
    else:
        cursor.execute('SELECT device_time FROM transactions WHERE card_hash = ?', (card_hash,))
        past_txns = cursor.fetchall()
        velocity_count = 1  
        dt_naive = dt.replace(tzinfo=None) if dt.tzinfo else dt

        for past in past_txns:
            try:
                p_time_str = past[0].strip().split('.')[0]
                p_dt = datetime.strptime(p_time_str, "%Y-%m-%d %H:%M:%S")
                if 0 <= (dt_naive - p_dt).total_seconds() <= 3600:
                    velocity_count += 1
            except Exception:
                pass

        home_distance_km = round(haversine(HOME_LAT, HOME_LNG, lat, lng), 1)
        issuer_lat, issuer_lng = ISSUER_LOCATIONS.get(issuer_name, (HOME_LAT, HOME_LNG))
        issuer_distance_km = round(haversine(issuer_lat, issuer_lng, lat, lng), 1)

        scaled_amount = np.log1p(amount)
        features = np.array([[scaled_amount, transaction_hour, velocity_count, home_distance_km, issuer_distance_km]])
        
        try:
            fraud_proba = model.predict_proba(features)[0][1]
        except Exception:
            fraud_proba = 0.05

        if home_distance_km > 1000 and amount > 500:
            fraud_proba = max(fraud_proba, 0.78)

        if amount >= 10000:
            fraud_proba = max(fraud_proba, 0.95)

    risk_percentage = math.floor(fraud_proba * 100)
    prediction_str = f"{risk_percentage}% Fraud Risk"
    tx_status = 'Flagged' if risk_percentage > 70 or is_blacklisted else 'Approved'

    cursor.execute('''
        INSERT INTO transactions (card_masked, card_hash, issuer_name, amount, device_time, velocity_count, home_distance_km, issuer_distance_km, device_ip, prediction, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (card_masked, card_hash, issuer_name, amount, formatted_device_time, velocity_count if not is_blacklisted else 1, 
          home_distance_km if not is_blacklisted else 0.0, issuer_distance_km if not is_blacklisted else 0.0, 
          client_ip, prediction_str, tx_status))
    conn.commit()
    conn.close()

    return redirect(url_for('transaction_intake'))

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    if not model:
        return jsonify({"error": "Model file missing."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    raw_card = str(data.get('card_number', '')).replace(' ', '')
    issuer_name = data.get('issuer_name', list(ISSUER_LOCATIONS.keys())[0])
    amount = float(data.get('amount', 0))
    lat = float(data.get('latitude', HOME_LAT))
    lng = float(data.get('longitude', HOME_LNG))
    
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if client_ip and ',' in client_ip:
        client_ip = client_ip.split(',')[0].strip()

    last_four = raw_card[-4:] if len(raw_card) >= 4 else "0000"
    card_masked = f"•••• {last_four}"
    card_hash = hashlib.sha256(raw_card.encode()).hexdigest() if raw_card else ""

    dt = datetime.now()
    formatted_device_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    transaction_hour = dt.hour

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id FROM flagged_entities WHERE (entity_value = ? AND entity_type = "Card") OR (entity_value = ? AND entity_type = "IP")', (card_masked, client_ip))
    is_blacklisted = cursor.fetchone() is not None

    if is_blacklisted:
        fraud_proba = 1.0
    else:
        cursor.execute('SELECT device_time FROM transactions WHERE card_hash = ?', (card_hash,))
        past_txns = cursor.fetchall()
        velocity_count = 1  

        for past in past_txns:
            try:
                p_time_str = past[0].strip().split('.')[0]
                p_dt = datetime.strptime(p_time_str, "%Y-%m-%d %H:%M:%S")
                if 0 <= (dt - p_dt).total_seconds() <= 3600:
                    velocity_count += 1
            except Exception:
                pass

        home_distance_km = round(haversine(HOME_LAT, HOME_LNG, lat, lng), 1)
        issuer_lat, issuer_lng = ISSUER_LOCATIONS.get(issuer_name, (HOME_LAT, HOME_LNG))
        issuer_distance_km = round(haversine(issuer_lat, issuer_lng, lat, lng), 1)

        scaled_amount = np.log1p(amount)
        features = np.array([[scaled_amount, transaction_hour, velocity_count, home_distance_km, issuer_distance_km]])
        
        try:
            fraud_proba = model.predict_proba(features)[0][1]
        except Exception:
            fraud_proba = 0.05

        if home_distance_km > 1000 and amount > 500:
            fraud_proba = max(fraud_proba, 0.78)

        if amount >= 10000:
            fraud_proba = max(fraud_proba, 0.95)

    risk_percentage = math.floor(fraud_proba * 100)
    prediction_str = f"{risk_percentage}% Fraud Risk"
    action_status = "BLOCK" if risk_percentage > 70 or is_blacklisted else "ALLOW"

    cursor.execute('''
        INSERT INTO transactions (card_masked, card_hash, issuer_name, amount, device_time, velocity_count, home_distance_km, issuer_distance_km, device_ip, prediction, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (card_masked, card_hash, issuer_name, amount, formatted_device_time, 1, 0.0, 0.0, client_ip, prediction_str, 'Flagged' if action_status == "BLOCK" else 'Approved'))
    conn.commit()
    conn.close()

    return jsonify({
        "card_masked": card_masked,
        "device_ip": client_ip,
        "risk_score_percentage": risk_percentage,
        "action": action_status,
        "status": "Blacklisted & Blocked" if is_blacklisted else "Evaluated successfully"
    })

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT password_hash FROM admins WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session['admin_logged_in'] = True
            session['admin_user'] = username
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('admin_user', None)
    return redirect(url_for('admin_login'))

@app.route('/admin')
@login_required
def admin_dashboard():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM transactions')
    total_txns = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE status = "Fraud Reported"')
    fraud_reports = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM flagged_entities')
    total_flagged = cursor.fetchone()[0]

    cursor.execute('SELECT id, card_masked, issuer_name, amount, device_time, device_ip, prediction, status FROM transactions ORDER BY id DESC LIMIT 50')
    transactions = cursor.fetchall()

    cursor.execute('SELECT id, entity_type, entity_value, reason, flagged_at FROM flagged_entities ORDER BY id DESC')
    flagged_list = cursor.fetchall()

    blacklisted_cards = {item[2] for item in flagged_list if item[1] == 'Card'}
    blacklisted_ips = {item[2] for item in flagged_list if item[1] == 'IP'}

    conn.close()
    return render_template('admin.html', total_txns=total_txns, fraud_reports=fraud_reports, total_flagged=total_flagged,
                           transactions=transactions, flagged_list=flagged_list, blacklisted_cards=blacklisted_cards, blacklisted_ips=blacklisted_ips)

@app.route('/admin/flag', methods=['POST'])
@login_required
def flag_entity():
    entity_type = request.form.get('entity_type')
    entity_value = request.form.get('entity_value')
    reason = request.form.get('reason', 'Suspicious activity detected')
    flagged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO flagged_entities (entity_type, entity_value, reason, flagged_at) VALUES (?, ?, ?, ?)', 
                       (entity_type, entity_value, reason, flagged_at))
        
        if entity_type == 'Card':
            cursor.execute('UPDATE transactions SET status = "Flagged", prediction = "100% Fraud Risk" WHERE card_masked = ?', (entity_value,))
        elif entity_type == 'IP':
            cursor.execute('UPDATE transactions SET status = "Flagged", prediction = "100% Fraud Risk" WHERE device_ip = ?', (entity_value,))
            
        conn.commit()
    except sqlite3.IntegrityError:
        pass 
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/flag_quick', methods=['POST'])
@login_required
def flag_entity_quick():
    card_masked = request.form.get('card_masked')
    reason = "Flagged from transaction triage"
    flagged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO flagged_entities (entity_type, entity_value, reason, flagged_at) VALUES (?, ?, ?, ?)', 
                       ('Card', card_masked, reason, flagged_at))
        
        cursor.execute('UPDATE transactions SET status = "Flagged", prediction = "100% Fraud Risk" WHERE card_masked = ?', (card_masked,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/flag_ip_quick', methods=['POST'])
@login_required
def flag_entity_ip_quick():
    device_ip = request.form.get('device_ip')
    reason = "Flagged from transaction triage (Suspicious IP)"
    flagged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO flagged_entities (entity_type, entity_value, reason, flagged_at) VALUES (?, ?, ?, ?)', 
                       ('IP', device_ip, reason, flagged_at))
        
        cursor.execute('UPDATE transactions SET status = "Flagged", prediction = "100% Fraud Risk" WHERE device_ip = ?', (device_ip,))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unflag/<int:flag_id>', methods=['POST'])
@login_required
def unflag_entity(flag_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('SELECT entity_type, entity_value FROM flagged_entities WHERE id = ?', (flag_id,))
    row = cursor.fetchone()
    
    cursor.execute('DELETE FROM flagged_entities WHERE id = ?', (flag_id,))
    
    if row:
        entity_type, entity_value = row
        if entity_type == 'Card':
            cursor.execute('UPDATE transactions SET status = "Approved", prediction = "5% Fraud Risk" WHERE card_masked = ? AND status = "Flagged"', (entity_value,))
        elif entity_type == 'IP':
            cursor.execute('UPDATE transactions SET status = "Approved", prediction = "5% Fraud Risk" WHERE device_ip = ? AND status = "Flagged"', (entity_value,))

    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/unflag_value', methods=['POST'])
@login_required
def unflag_entity_by_value():
    entity_value = request.form.get('entity_value')
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('SELECT entity_type FROM flagged_entities WHERE entity_value = ?', (entity_value,))
    row = cursor.fetchone()
    
    cursor.execute('DELETE FROM flagged_entities WHERE entity_value = ?', (entity_value,))
    
    if row:
        entity_type = row[0]
        if entity_type == 'Card':
            cursor.execute('UPDATE transactions SET status = "Approved", prediction = "5% Fraud Risk" WHERE card_masked = ? AND status = "Flagged"', (entity_value,))
        elif entity_type == 'IP':
            cursor.execute('UPDATE transactions SET status = "Approved", prediction = "5% Fraud Risk" WHERE device_ip = ? AND status = "Flagged"', (entity_value,))
    
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/report_fraud/<int:txn_id>', methods=['POST'])
@login_required
def report_fraud(txn_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE transactions SET status = "Fraud Reported" WHERE id = ?', (txn_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/clear', methods=['GET', 'POST'])
@login_required
def clear_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM transactions')
    cursor.execute('DELETE FROM flagged_entities')
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)