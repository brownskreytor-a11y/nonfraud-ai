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

try:
    import requests
except ImportError:
    # requests isn't installed -- the app still runs, it just can't send
    # real SMS and always falls back to the on-screen demo code below.
    requests = None

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
HOME_COUNTRY = "GH"

# A transaction on a card issued outside the cardholder's home country is a
# classic cross-border fraud signal on its own, independent of amount or
# velocity -- so it gets a risk floor and always triggers the OTP step-up,
# even for a small amount that would otherwise sail through untouched.
DIFFERENT_COUNTRY_RISK_FLOOR = 0.40


def _issuer_country(issuer_name):
    """Pulls the 2-letter country code out of an issuer's display name,
    e.g. "Ecobank Ghana (Accra, GH)" -> "GH". Every entry in
    ISSUER_LOCATIONS is formatted "<name> (<city>, <XX>)"."""
    if not issuer_name or ',' not in issuer_name or not issuer_name.endswith(')'):
        return None
    return issuer_name.rsplit(',', 1)[-1].strip(') ').strip()

# Every evaluated transaction now requires manual admin review before it is
# considered final -- the model's risk score is computed and shown, but it
# only informs the admin's Approve/Flag decision on the dashboard, it never
# decides the outcome by itself. A blacklisted card/IP is the one exception:
# that is a deterministic security rule, not a prediction, so it still
# auto-flags instantly and skips the review queue.

# A hard ceiling, not a risk judgment -- nothing above this amount is ever
# processed at all, regardless of the model's score, blacklist status, or
# OTP. It's checked before anything else touches the database, so a
# transaction over the limit produces no row and never reaches scoring,
# the same way a card network declines a charge that exceeds a hard limit
# before it even considers fraud risk.
MAX_TRANSACTION_AMOUNT = 5000

# Transactions at or above this amount can't slip through on the model's
# score alone (see train_model.py -- amount alone can never push the "true"
# fraud probability much past ~12%, so a lone large amount reads as low
# risk). Instead of just labeling it for the admin queue, the cardholder
# has to clear a one-time-code challenge before the transaction is even
# recorded -- an active control, not just a passive flag. A card that fails
# or abandons the challenge produces no transaction row at all.
OTP_REVIEW_THRESHOLD = 300
OTP_MAX_ATTEMPTS = 3
OTP_TTL_SECONDS = 300

# Clearing the one-time-code challenge no longer discounts the stored risk
# score -- a verified transaction still carries its real, pre-challenge risk
# percentage into admin review. Passing OTP only proves the cardholder holds
# the verification channel on file for this card; it doesn't by itself lower
# how risky the transaction actually is, so the admin sees the same number
# either way.

# Real SMS/voice delivery via Arkesel's OTP API -- set ARKESEL_API_KEY in
# the environment (never hardcode it) to send an actual code to the
# cardholder's phone. ARKESEL_SENDER_ID is optional (defaults below) and
# controls the sender name shown on the SMS -- SMS sender names require
# operator-level registration (a business certificate + authorization
# letter) in Ghana, which an individual/student account won't have.
# ARKESEL_OTP_MEDIUM lets that be worked around: set it to "voice" (Ghana
# only) to have Arkesel place a call reading the code aloud instead of
# texting it -- a voice call isn't a branded sender name, so it isn't
# gated by the same registration requirement. Defaults to "sms". When the
# key is missing, or a send/verify call fails (no credit, bad number, no
# network, unregistered sender), the app transparently falls back to
# generating its own code and showing it on screen -- the review step is
# never skipped, only the delivery channel changes.
ARKESEL_API_KEY = os.environ.get("ARKESEL_API_KEY")
ARKESEL_SENDER_ID = os.environ.get("ARKESEL_SENDER_ID", "NonFraud")
ARKESEL_OTP_MEDIUM = os.environ.get("ARKESEL_OTP_MEDIUM", "sms").strip().lower()
if ARKESEL_OTP_MEDIUM not in ("sms", "voice"):
    ARKESEL_OTP_MEDIUM = "sms"
ARKESEL_BASE_URL = "https://sms.arkesel.com/api/otp"

arkesel_configured = bool(requests and ARKESEL_API_KEY)


def _normalize_phone_for_arkesel(phone_number):
    """Arkesel expects international format with no leading '+' (e.g.
    233544919953). Accepts a local Ghana number typed with a leading 0
    (e.g. 0544919953) and converts it, since that's how most people will
    naturally type their own number."""
    cleaned = phone_number.strip().replace(' ', '').replace('-', '')
    if cleaned.startswith('+'):
        cleaned = cleaned[1:]
    if cleaned.startswith('0') and len(cleaned) == 10:
        cleaned = '233' + cleaned[1:]
    return cleaned


def _arkesel_send_otp(phone_number):
    """Generates and sends an OTP via Arkesel. Arkesel holds the code
    server-side (like Twilio Verify did) -- nothing to store locally beyond
    knowing a real send succeeded. Returns True on success, False on any
    failure (network, bad number, insufficient balance, etc.)."""
    try:
        resp = requests.post(
            f"{ARKESEL_BASE_URL}/generate",
            headers={"api-key": ARKESEL_API_KEY, "Content-Type": "application/json"},
            json={
                "expiry": max(1, min(10, OTP_TTL_SECONDS // 60)),
                "length": 6,
                "medium": ARKESEL_OTP_MEDIUM,
                "message": "Your NonFraud-AI verification code is: %otp_code%",
                "number": _normalize_phone_for_arkesel(phone_number),
                "sender_id": ARKESEL_SENDER_ID,
                "type": "numeric",
            },
            timeout=10,
        )
        data = resp.json()
        ok = data.get("code") == "1000"
        if not ok:
            # Logged (not swallowed) so a real send failure -- bad key,
            # unapproved sender ID, zero balance, wrong number format --
            # shows up in Render's Logs tab instead of just silently
            # falling back to demo mode with no way to tell why.
            app.logger.warning("Arkesel OTP generate failed: HTTP %s, response=%s", resp.status_code, data)
        return ok
    except Exception as e:
        app.logger.warning("Arkesel OTP generate error: %s", e)
        return False


def _arkesel_verify_otp(phone_number, code):
    try:
        resp = requests.post(
            f"{ARKESEL_BASE_URL}/verify",
            headers={"api-key": ARKESEL_API_KEY, "Content-Type": "application/json"},
            json={"code": code, "number": _normalize_phone_for_arkesel(phone_number)},
            timeout=10,
        )
        data = resp.json()
        ok = data.get("code") == "1100"
        if not ok:
            app.logger.warning("Arkesel OTP verify failed: HTTP %s, response=%s", resp.status_code, data)
        return ok
    except Exception as e:
        app.logger.warning("Arkesel OTP verify error: %s", e)
        return False

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
            status TEXT DEFAULT 'Pending',
            otp_verified INTEGER DEFAULT 0
        )
    ''')
    # Migrate older databases that don't have card_hash / otp_verified yet
    cursor.execute("PRAGMA table_info(transactions)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if 'card_hash' not in existing_cols:
        cursor.execute("ALTER TABLE transactions ADD COLUMN card_hash TEXT DEFAULT ''")
    if 'otp_verified' not in existing_cols:
        cursor.execute("ALTER TABLE transactions ADD COLUMN otp_verified INTEGER DEFAULT 0")
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
    return render_template('index.html', logs=logs, issuers=list(ISSUER_LOCATIONS.keys()),
                           notice=request.args.get('notice'), success=request.args.get('success'))

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
    phone_number = request.form.get('phone_number', '').strip()
    is_different_country = _issuer_country(issuer_name) not in (None, HOME_COUNTRY)

    if amount > MAX_TRANSACTION_AMOUNT:
        return redirect(url_for('transaction_intake',
                                notice=f'Transaction declined -- ${amount:,.2f} exceeds the ${MAX_TRANSACTION_AMOUNT:,.0f} per-transaction limit.'))

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

        # A card issued outside the cardholder's home country is checked
        # before the other override rules below -- it sets a 40% floor
        # first, so amount/distance can still push it higher (e.g. also
        # far from home + large amount still lands on the 78% floor), but
        # it never drops back below 40% just for being foreign-issued.
        if is_different_country:
            fraud_proba = max(fraud_proba, DIFFERENT_COUNTRY_RISK_FLOOR)

        if home_distance_km > 1000 and amount > 500:
            fraud_proba = max(fraud_proba, 0.78)

    risk_percentage = math.floor(fraud_proba * 100)
    prediction_str = f"{risk_percentage}% Fraud Risk"

    # Blacklisted entities still auto-flag; every other transaction waits
    # for the admin to Approve or Flag it from the dashboard, guided by the
    # risk_percentage/prediction_str computed above.
    tx_status = 'Flagged' if is_blacklisted else 'Pending'

    # Large amounts don't get inserted straight away -- they have to clear a
    # one-time-code challenge first. Nothing is written to the transactions
    # table until verify_otp() confirms the code (or the card is blacklisted,
    # which bypasses this and auto-flags instead). A different-country card
    # forces the same challenge regardless of amount -- cross-border use is
    # a strong enough signal on its own to warrant step-up verification.
    if not is_blacklisted and (amount >= OTP_REVIEW_THRESHOLD or is_different_country):
        conn.close()

        # Try a real SMS via Arkesel first. Arkesel generates and holds the
        # code itself (nothing to store locally), so a "delivery" flag is
        # all that's needed to remember which path to check against later.
        # Any reason it can't go out -- not configured, no network, bad
        # number, no credit -- falls back to a locally generated code shown
        # on screen, so the review step itself never breaks even if the SMS
        # side does.
        otp_delivery = 'demo'
        otp_code = None
        if not arkesel_configured:
            app.logger.info("Falling back to demo OTP: ARKESEL_API_KEY is not set.")
        elif not phone_number:
            app.logger.info("Falling back to demo OTP: no phone number was submitted with the transaction.")
        elif _arkesel_send_otp(phone_number):
            otp_delivery = 'arkesel'

        if otp_delivery == 'demo':
            otp_code = f"{secrets.randbelow(1000000):06d}"

        session['pending_otp'] = {
            'delivery': otp_delivery,
            'medium': ARKESEL_OTP_MEDIUM,
            'phone_number': phone_number,
            'code': otp_code,
            'attempts': 0,
            'expires_at': datetime.now().timestamp() + OTP_TTL_SECONDS,
            'txn': {
                'card_masked': card_masked,
                'card_hash': card_hash,
                'issuer_name': issuer_name,
                'amount': amount,
                'device_time': formatted_device_time,
                'velocity_count': velocity_count,
                'home_distance_km': home_distance_km,
                'issuer_distance_km': issuer_distance_km,
                'device_ip': client_ip,
                'prediction': prediction_str,
                'risk_percentage': risk_percentage,
            }
        }
        return render_template('otp_verify.html', amount=amount, card_masked=card_masked,
                               issuer_name=issuer_name, prediction_str=prediction_str, otp_code=otp_code,
                               delivery=otp_delivery, medium=ARKESEL_OTP_MEDIUM, phone_number=phone_number)

    cursor.execute('''
        INSERT INTO transactions (card_masked, card_hash, issuer_name, amount, device_time, velocity_count, home_distance_km, issuer_distance_km, device_ip, prediction, status, otp_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (card_masked, card_hash, issuer_name, amount, formatted_device_time, velocity_count if not is_blacklisted else 1,
          home_distance_km if not is_blacklisted else 0.0, issuer_distance_km if not is_blacklisted else 0.0,
          client_ip, prediction_str, tx_status, 0))
    conn.commit()
    conn.close()

    return redirect(url_for('transaction_intake'))

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    pending = session.get('pending_otp')
    if not pending:
        # Nothing to verify (session expired, or the page was reloaded
        # after already resolving) -- just send them back to submit again.
        return redirect(url_for('transaction_intake', notice='No verification in progress. Please resubmit the transaction.'))

    if request.form.get('cancel'):
        session.pop('pending_otp', None)
        return redirect(url_for('transaction_intake'))

    if datetime.now().timestamp() > pending['expires_at']:
        session.pop('pending_otp', None)
        return redirect(url_for('transaction_intake', notice='Verification code expired. Please resubmit the transaction.'))

    entered_code = request.form.get('otp_code', '').strip()
    txn = pending['txn']

    if pending['delivery'] == 'arkesel' and arkesel_configured:
        verified = _arkesel_verify_otp(pending['phone_number'], entered_code)
    else:
        verified = (entered_code == pending['code'])

    if not verified:
        pending['attempts'] += 1
        if pending['attempts'] >= OTP_MAX_ATTEMPTS:
            session.pop('pending_otp', None)
            return redirect(url_for('transaction_intake', notice='Transaction blocked -- verification code entered incorrectly too many times.'))
        session['pending_otp'] = pending
        return render_template('otp_verify.html', amount=txn['amount'], card_masked=txn['card_masked'],
                               issuer_name=txn['issuer_name'], prediction_str=txn['prediction'], otp_code=pending['code'],
                               delivery=pending['delivery'], medium=pending.get('medium', 'sms'), phone_number=pending['phone_number'],
                               error='Incorrect code.', attempts_left=OTP_MAX_ATTEMPTS - pending['attempts'])

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO transactions (card_masked, card_hash, issuer_name, amount, device_time, velocity_count, home_distance_km, issuer_distance_km, device_ip, prediction, status, otp_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (txn['card_masked'], txn['card_hash'], txn['issuer_name'], txn['amount'], txn['device_time'],
          txn['velocity_count'], txn['home_distance_km'], txn['issuer_distance_km'], txn['device_ip'],
          txn['prediction'], 'Pending', 1))
    conn.commit()
    conn.close()

    session.pop('pending_otp', None)
    return redirect(url_for('transaction_intake', success='Verification successful.'))

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
    is_different_country = _issuer_country(issuer_name) not in (None, HOME_COUNTRY)

    if amount > MAX_TRANSACTION_AMOUNT:
        return jsonify({
            "action": "BLOCK",
            "status": f"Declined -- exceeds the ${MAX_TRANSACTION_AMOUNT:,.0f} per-transaction limit"
        }), 200

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

        if is_different_country:
            fraud_proba = max(fraud_proba, DIFFERENT_COUNTRY_RISK_FLOOR)

        if home_distance_km > 1000 and amount > 500:
            fraud_proba = max(fraud_proba, 0.78)

    risk_percentage = math.floor(fraud_proba * 100)
    prediction_str = f"{risk_percentage}% Fraud Risk"

    # This endpoint is a machine-to-machine API call, not a browser session,
    # so it can't run the interactive one-time-code challenge that /predict
    # uses for the same threshold (see verify_otp()). It still has to tell
    # the caller the truth: a large amount or different-country card here
    # has NOT cleared step-up verification, so otp_verified stays 0 and the
    # caller is told to collect that verification on their end before
    # treating this as final.
    otp_required = (not is_blacklisted) and (amount >= OTP_REVIEW_THRESHOLD or is_different_country)

    if is_blacklisted:
        tx_status = 'Flagged'
        action_status = "BLOCK"
    elif otp_required:
        tx_status = 'Pending'
        action_status = "OTP_REQUIRED"
    else:
        tx_status = 'Pending'
        action_status = "REVIEW"

    stored_velocity = 1 if is_blacklisted else velocity_count
    stored_home_distance = 0.0 if is_blacklisted else home_distance_km
    stored_issuer_distance = 0.0 if is_blacklisted else issuer_distance_km

    cursor.execute('''
        INSERT INTO transactions (card_masked, card_hash, issuer_name, amount, device_time, velocity_count, home_distance_km, issuer_distance_km, device_ip, prediction, status, otp_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (card_masked, card_hash, issuer_name, amount, formatted_device_time, stored_velocity, stored_home_distance, stored_issuer_distance, client_ip, prediction_str, tx_status, 0))
    conn.commit()
    conn.close()

    if is_blacklisted:
        status_text = "Blacklisted & Blocked"
    elif otp_required:
        status_text = "Step-up verification (OTP) required before this can be finalized"
    elif tx_status == 'Pending':
        status_text = "Pending manual review"
    else:
        status_text = "Evaluated successfully"

    return jsonify({
        "card_masked": card_masked,
        "device_ip": client_ip,
        "risk_score_percentage": risk_percentage,
        "action": action_status,
        "otp_required": otp_required,
        "status": status_text
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

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE status = "Pending"')
    pending_review = cursor.fetchone()[0]

    # velocity_count/home_distance_km/issuer_distance_km are pulled in
    # alongside the prediction so the template can show an explicit "why
    # this needs a look" signal next to the risk score -- the model's
    # percentage alone dilutes velocity for small/local transactions (see
    # train_model.py's weighting), so admins shouldn't have to infer a
    # rapid-fire pattern purely from a number that can still read as low.
    cursor.execute('''SELECT id, card_masked, issuer_name, amount, device_time, device_ip, prediction, status,
                              velocity_count, home_distance_km, issuer_distance_km, otp_verified
                       FROM transactions ORDER BY id DESC LIMIT 50''')
    transactions = cursor.fetchall()

    cursor.execute('SELECT id, entity_type, entity_value, reason, flagged_at FROM flagged_entities ORDER BY id DESC')
    flagged_list = cursor.fetchall()

    blacklisted_cards = {item[2] for item in flagged_list if item[1] == 'Card'}
    blacklisted_ips = {item[2] for item in flagged_list if item[1] == 'IP'}

    # Cards specifically blacklisted through report_fraud() (its reason text
    # always starts with this prefix) get their own review list, separate
    # from the general Active Blacklist -- these are confirmed-fraud cards,
    # not just ad hoc suspicious flags, so an admin reviewing them needs to
    # find them without hunting through every blacklist reason. "Release"
    # reuses the same unflag path as the general list: it lifts the card's
    # blacklist and restores any other transactions it swept to Flagged, but
    # leaves the original reported transaction's own "Fraud Reported" status
    # alone as a permanent record of what was reported.
    fraud_reported_cards = [item for item in flagged_list
                             if item[1] == 'Card' and item[3].startswith('Reported as fraud from transaction #')]

    conn.close()
    return render_template('admin.html', total_txns=total_txns, fraud_reports=fraud_reports, total_flagged=total_flagged,
                           pending_review=pending_review, transactions=transactions, flagged_list=flagged_list,
                           fraud_reported_cards=fraud_reported_cards,
                           blacklisted_cards=blacklisted_cards, blacklisted_ips=blacklisted_ips)

@app.route('/admin/approve/<int:txn_id>', methods=['POST'])
@login_required
def approve_transaction(txn_id):
    # Resolves a transaction sitting in 'Pending' (large amount and/or high
    # velocity) after manual review clears it -- the counterpart to Flag,
    # which resolves it the other way.
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE transactions SET status = "Approved" WHERE id = ? AND status = "Pending"', (txn_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/cancel/<int:txn_id>', methods=['POST'])
@login_required
def cancel_transaction(txn_id):
    # A neutral decline -- distinct from Flag/Report, which accuse the card
    # of fraud and blacklist it. Cancel just closes the transaction out with
    # no accusation attached. Usable from two starting points: while it's
    # still Pending (admin declines it outright) or after it was already
    # Approved (the cardholder calls in asking to cancel, so it needs to be
    # unwound after the fact).
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('UPDATE transactions SET status = "Cancelled" WHERE id = ? AND status IN ("Pending", "Approved")', (txn_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

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
    # Reporting fraud used to just relabel this one row, which did nothing
    # to stop the same card being used again. It now actually blacklists
    # the card -- so it hits the auto-flag path in /predict and
    # /api/evaluate on its very next transaction -- and sweeps up any other
    # still-open transactions already on record from that same card, since
    # a confirmed-fraud card means their prior activity deserves a second
    # look too. The reported transaction itself keeps the distinct "Fraud
    # Reported" status as the audit trail marker for which row triggered it.
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('SELECT card_masked FROM transactions WHERE id = ?', (txn_id,))
    row = cursor.fetchone()

    cursor.execute('UPDATE transactions SET status = "Fraud Reported" WHERE id = ?', (txn_id,))

    if row:
        card_masked = row[0]
        reason = f"Reported as fraud from transaction #{txn_id}"
        flagged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute('INSERT INTO flagged_entities (entity_type, entity_value, reason, flagged_at) VALUES (?, ?, ?, ?)',
                           ('Card', card_masked, reason, flagged_at))
        except sqlite3.IntegrityError:
            pass  # card was already blacklisted

        cursor.execute('''UPDATE transactions SET status = "Flagged", prediction = "100% Fraud Risk"
                           WHERE card_masked = ? AND id != ? AND status != "Fraud Reported"''', (card_masked, txn_id))

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