import eventlet
eventlet.monkey_patch()

import os
import pickle
import time
import numpy as np
from collections import deque, Counter
from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit

# --- إعدادات الثبات ---
ADD_LETTER_DELAY = 3.0     
MAX_QUEUE_LEN = 20         
PREDICTION_THRESHOLD = 15  

app = Flask(__name__, template_folder='.', static_folder='.')
app.config['SECRET_KEY'] = 'secret!'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=False, engineio_logger=False)

# تحميل الموديل
MODEL_PATH = './rf.pkl'
model_en = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data:
                model_en = data['model']
            else:
                model_en = data
        print("Model loaded successfully.")
    else:
        print("!!! Error: Model file not found !!!")
except Exception as e:
    print(f"Error loading model: {e}")

english_letters = [chr(65 + i) for i in range(26)]
labels_dict_en = {i: english_letters[i] for i in range(len(english_letters))}
labels_dict_en[len(english_letters)] = "Space"

client_states = {}

def get_or_create_state(sid):
    if sid not in client_states:
        client_states[sid] = {
            'queue': deque(maxlen=MAX_QUEUE_LEN),
            'sentence': "",
            'last_char': "",
            'last_time': 0,
        }
    return client_states[sid]

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

@socketio.on('connect')
def handle_connect():
    get_or_create_state(request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    client_states.pop(request.sid, None)

# استقبال الإحداثيات فقط
@socketio.on('process_landmarks')
def handle_process_landmarks(data):
    sid = request.sid
    state = get_or_create_state(sid)
    current_time = time.time()
    
    if not model_en:
        return

    try:
        landmarks = data.get('landmarks', [])
        if not landmarks: return

        # تحويل البيانات (Normalization)
        x_ = [lm.get('x', 0) for lm in landmarks]
        y_ = [lm.get('y', 0) for lm in landmarks]
        
        min_x, min_y = min(x_), min(y_)
        
        data_aux = []
        for lm in landmarks:
            data_aux.append(lm.get('x', 0) - min_x)
            data_aux.append(lm.get('y', 0) - min_y)

        # التوقع
        prediction = model_en.predict([np.asarray(data_aux)])
        predicted_character = labels_dict_en.get(int(prediction[0]), "?")
        
        # منطق الجملة
        state['queue'].append(predicted_character)
        most_common_char, count = Counter(state['queue']).most_common(1)[0]
        
        stable_char = ""
        if count > PREDICTION_THRESHOLD:
            stable_char = most_common_char
            if (most_common_char != state['last_char'] or current_time - state['last_time'] > ADD_LETTER_DELAY):
                if most_common_char == "Space":
                    state['sentence'] += " "
                else:
                    state['sentence'] += most_common_char
                
                state['last_char'] = most_common_char
                state['last_time'] = current_time
                state['queue'].clear()
                emit('sentence_update', {'sentence': state['sentence']})

        emit('prediction', {'prediction': predicted_character})
        emit('stable_prediction', {'stable_prediction': stable_char})

    except Exception:
        pass

@socketio.on('clear_sentence')
def handle_clear():
    state = get_or_create_state(request.sid)
    state['sentence'] = ""
    state['last_char'] = ""
    state['queue'].clear()
    emit('sentence_update', {'sentence': ""})

@socketio.on('backspace')
def handle_backspace():
    state = get_or_create_state(request.sid)
    if state['sentence']:
        state['sentence'] = state['sentence'][:-1]
        state['last_char'] = "" 
        state['queue'].clear()
        emit('sentence_update', {'sentence': state['sentence']})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=7860, allow_unsafe_werkzeug=True)
