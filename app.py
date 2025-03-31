# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd

# app = Flask(__name__, template_folder="templates", static_folder="static")

# # Load trained models
# model_lstm = tf.keras.models.load_model('lstm_model_tuned.keras')
# model_bilstm = tf.keras.models.load_model('bilstm_model_tuned.keras')

# # Load dataset (For Scaling)
# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
# df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
# data = df[['Passengers']]

# # Initialize scaler and fit it on the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(data)

# # Define sequence length
# SEQ_LENGTH = 12

# # Function to preprocess input data
# def preprocess_input(input_data):
#     input_scaled = scaler.transform(np.array(input_data).reshape(-1, 1))
#     input_seq = np.array([input_scaled[-SEQ_LENGTH:]])  # Ensure correct shape (1, 12, 1)
#     return input_seq.reshape(1, SEQ_LENGTH, 1)

# # ✅ FIXED: Serve the HTML page when accessing "/"
# @app.route('/')
# def home():
#     return render_template("index.html")  # 🔥 This will serve your front-end HTML

# # API Prediction Endpoint
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         input_sequence = data.get('sequence')

#         if len(input_sequence) != 12:
#             return jsonify({"error": "Input must have 12 values"}), 400

#         processed_input = preprocess_input(input_sequence)

#         # Predictions
#         lstm_pred = model_lstm.predict(processed_input)[0][0]
#         bilstm_pred = model_bilstm.predict(processed_input)[0][0]

#         # Convert predictions back to original scale
#         lstm_pred_original = scaler.inverse_transform([[lstm_pred]])[0][0]
#         bilstm_pred_original = scaler.inverse_transform([[bilstm_pred]])[0][0]

#         return jsonify({
#             "LSTM Prediction": lstm_pred_original,
#             "BiLSTM Prediction": bilstm_pred_original
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)



from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load dataset (For Scaling)
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
data = df[['Passengers']]

# Initialize scaler and fit it on the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data)

# Define sequence length
SEQ_LENGTH = 12

# Global variables for lazy-loaded models
model_lstm = None
model_bilstm = None

def load_models():
    """Lazy load models only when needed to save memory."""
    global model_lstm, model_bilstm
    if model_lstm is None:
        print("Loading LSTM model...")
        model_lstm = tf.keras.models.load_model('lstm_model_tuned.keras')
    if model_bilstm is None:
        print("Loading BiLSTM model...")
        model_bilstm = tf.keras.models.load_model('bilstm_model_tuned.keras')

# Function to preprocess input data
def preprocess_input(input_data):
    input_scaled = scaler.transform(np.array(input_data).reshape(-1, 1))
    input_seq = np.array([input_scaled[-SEQ_LENGTH:]])  # Ensure correct shape (1, 12, 1)
    return input_seq.reshape(1, SEQ_LENGTH, 1)

@app.route('/')
def home():
    return render_template("index.html")

# API Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_sequence = data.get('sequence')

        if len(input_sequence) != 12:
            return jsonify({"error": "Input must have 12 values"}), 400

        # Load models dynamically (lazy loading)
        load_models()

        processed_input = preprocess_input(input_sequence)

        # Predictions
        lstm_pred = model_lstm.predict(processed_input)[0][0]
        bilstm_pred = model_bilstm.predict(processed_input)[0][0]

        # Convert predictions back to original scale
        lstm_pred_original = scaler.inverse_transform([[lstm_pred]])[0][0]
        bilstm_pred_original = scaler.inverse_transform([[bilstm_pred]])[0][0]

        return jsonify({
            "LSTM Prediction": lstm_pred_original,
            "BiLSTM Prediction": bilstm_pred_original
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
