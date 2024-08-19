import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import traceback
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Model
import torch.distributions as dist

app = Flask(__name__)

# Tải các mô hình LSTM và GRU
model_lstm = load_model('models/lstm_model.keras')
model_gru = load_model('models/gru_model.keras')
# Định nghĩa hàm dự đoán cho LSTM và GRU
def predict_future_value(target_date, model, df, features, scaler, time_steps=14):
    target_date = pd.to_datetime(target_date)
    
    if target_date in df['day'].values:
        actual_data = df[df['day'] == target_date][features].values
        return actual_data[0]  # Trả về giá trị có trong tập dữ liệu
    
    last_14_days = df[features].values[-time_steps:]
    last_14_days_scaled = scaler.transform(last_14_days)
    predicted_values = []
    
    while target_date not in df['day'].values:
        X_test = np.reshape(last_14_days_scaled, (1, time_steps, len(features)))
        predicted_value_scaled = model.predict(X_test)
        last_14_days_scaled = np.append(last_14_days_scaled[1:], predicted_value_scaled, axis=0)
        predicted_values.append(predicted_value_scaled[0])
        target_date -= pd.Timedelta(days=1)
    
    predicted_value = scaler.inverse_transform([predicted_values[-1]])
    return predicted_value[0]
#--------------------------------
# Load mô hình DeepAR
features = ['temperature_2m','apparent_temperature', 'soil_temperature_0_to_7cm','et0_fao_evapotranspiration']
# Định nghĩa lớp Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        attn_weights = self.softmax(self.attn(lstm_out))
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context

# Định nghĩa lớp DeepAR với LSTM hai chiều và Attention
class DeepAR(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(DeepAR, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size * 2, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc_mu = nn.Linear(hidden_size * 2, output_size)  # Bidirectional nên nhân với 2
        self.fc_presigma = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # *2 do bidirectional
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        attn_out = self.attention(out)
        mu = self.fc_mu(attn_out)
        sigma = torch.log(1 + torch.exp(self.fc_presigma(attn_out)))

        return mu, sigma

# Các siêu tham số
params = {
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.3,
}

# Định nghĩa mô hình DeepAR
model_deepar = DeepAR(
    input_size=(len(features)),
    hidden_size=params['hidden_size'],
    num_layers=params['num_layers'],
    output_size=len(features),
    dropout=params['dropout']
    )

model_deepar.load_state_dict(torch.load('models/deepar_model.pth'))
model_deepar.eval()

def predict_future_value_deepar(target_date, model, df, features, scaler, time_steps=14):
    target_date = pd.to_datetime(target_date)

    last_14_days = df[features].tail(time_steps)
    
    # Loại bỏ tên đặc trưng trước khi transform
    last_14_days_scaled = scaler.transform(last_14_days.values)

    initial_data = last_14_days_scaled.reshape(1, time_steps, len(features))

    def predict_future_days(model, initial_data, days_to_predict):
        predictions = []
        current_input = initial_data
        
        # Khởi tạo z_prev là giá trị của ngày cuối cùng trong chuỗi đầu vào
        z_prev = current_input[0, -1, :]

        for i in range(days_to_predict):
            z_prev_reshaped = z_prev.reshape(1, 1, -1)

            X_test = np.concatenate([current_input, z_prev_reshaped], axis=1)
            X_test = np.concatenate([X_test, X_test], axis=-1)
            X_test = torch.tensor(X_test).float()

            with torch.no_grad():
                mu, sigma = model(X_test)
                normal_dist = dist.Normal(mu, sigma)

                samples = normal_dist.sample((90000,))
                result = samples.mean(dim=0).numpy()

            predictions.append(result)
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, :] = result

        return np.array(predictions)

    if target_date in df['day'].values:
        actual_data = df[df['day'] == target_date][features].values
        return actual_data[0]
    else:
        days_to_predict = (target_date - df['day'].max()).days
        if days_to_predict > 0:
            future_predictions_scaled = predict_future_days(model, initial_data, days_to_predict)

            future_predictions_df = pd.DataFrame(future_predictions_scaled.reshape(-1, len(features)), columns=features)

            result_reshaped = scaler.inverse_transform(future_predictions_df)

            final_prediction = result_reshaped[-1]
            return final_prediction
        else:
            return None

#--------------------------------
# Lớp Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, model_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, model_dim)

    def get_angles(self, pos, i, model_dim):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(model_dim, tf.float32))
        return pos * angles

    def positional_encoding(self, sequence_length, model_dim):
        pos = tf.cast(tf.range(sequence_length)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(model_dim)[tf.newaxis, :], tf.float32)
        angle_rads = self.get_angles(pos, i, model_dim)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

# Lớp Transformer Encoder
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate, epsilon):
        super(TransformerEncoderLayer, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=model_dim)
        self.ffn = tf.keras.Sequential([
            Conv1D(filters=ff_dim, kernel_size=1, activation='relu'),
            Conv1D(filters=model_dim, kernel_size=1)
        ])
        self.layernorm1 = LayerNormalization(epsilon=epsilon)
        self.layernorm2 = LayerNormalization(epsilon=epsilon)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Mô hình Transformer
def build_transformer_model(sequence_length, feature_dim, model_dim, num_heads, ff_dim, num_layers, dropout_rate, epsilon, training=False):
    inputs = Input(shape=(sequence_length, feature_dim))
    x = Dense(model_dim)(inputs)
    x = PositionalEncoding(sequence_length, model_dim)(x)

    for _ in range(num_layers):
        x = TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout_rate, epsilon)(x, training=training)

    x = GlobalAveragePooling1D()(x)
    x = Dense(ff_dim, activation='relu')(x)
    outputs = Dense(feature_dim)(x)

    return Model(inputs=inputs, outputs=outputs)

# Các siêu tham số cho Transformer
sequence_length = 14
feature_dim = 4
model_dim = 64
num_heads = 4
ff_dim = model_dim*4
num_layers = 2
dropout_rate = 0.2
epsilon = 3.1455e-6

# Định nghĩa mô hình Transformer
transformer_model = build_transformer_model(
    sequence_length, feature_dim, model_dim, num_heads,
    ff_dim, num_layers, dropout_rate, epsilon
)

# Tải trọng số vào mô hình
transformer_model.load_weights('models/transformer_model.keras')

# Cập nhật hàm dự đoán cho Transformer
def predict_future_value_transformer(target_date, model, df, features, scaler, time_steps=14):
    target_date = pd.to_datetime(target_date)

    last_14_days = df[features].tail(time_steps)
    
    # Loại bỏ tên đặc trưng trước khi transform
    last_14_days_scaled = scaler.transform(last_14_days.values)

    initial_data = last_14_days_scaled.reshape(1, time_steps, len(features))

    def predict_future_days(model, initial_data, days_to_predict):
        predictions = []
        current_input = initial_data

        for _ in range(days_to_predict):
            predictions_transformer = model.predict(current_input)
            predictions.append(predictions_transformer.flatten())
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, :] = predictions_transformer.flatten()

        return np.array(predictions)

    if target_date in df['day'].values:
        actual_data = df[df['day'] == target_date][features].values
        return actual_data[0]
    else:
        days_to_predict = (target_date - df['day'].max()).days

        if days_to_predict > 0:
            future_predictions_scaled = predict_future_days(model, initial_data, days_to_predict)

            result_reshaped = future_predictions_scaled.reshape(-1, len(features))
            result_reshaped = scaler.inverse_transform(result_reshaped)

            final_prediction = result_reshaped[-1]
            return final_prediction
        else:
            return None

#--------------------------------
# Hàm dự đoán chính
def predict(date, model_type):
    try:
        df = pd.read_csv("data/initial_data.csv")
        df['day'] = pd.to_datetime(df['day'], format='%Y-%m-%d')
        
        features = ['temperature_2m', 'apparent_temperature', 'soil_temperature_0_to_7cm', 'et0_fao_evapotranspiration']
        hcm_df = df[['day'] + features]
        
        scaler = MinMaxScaler()
        hcm_data = hcm_df[features].values
        hcm_data_scaled = scaler.fit_transform(hcm_data)
        
        selected_date = pd.to_datetime(date, format='%Y-%m-%d')
        
        if model_type == 'lstm':
            prediction = predict_future_value(selected_date, model_lstm, hcm_df, features, scaler)
        elif model_type == 'gru':
            prediction = predict_future_value(selected_date, model_gru, hcm_df, features, scaler)
        elif model_type == 'deepar':
            prediction = predict_future_value_deepar(selected_date, model_deepar, hcm_df, features, scaler)
        elif model_type == 'transformer':
            prediction = predict_future_value_transformer(selected_date, transformer_model, hcm_df, features, scaler)
        return prediction.tolist()
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        raise e

# Route chính của ứng dụng
@app.route('/')
def index():
    return render_template('index.html')

# Route dự đoán
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()  # Đọc dữ liệu JSON
        date = data['date']
        model = data['model']
        
        # Dự đoán
        forecast = predict(date, model)
        
        # Đảm bảo dự đoán là danh sách số, nếu không thì đóng gói
        if isinstance(forecast, list) and all(isinstance(i, (int, float)) for i in forecast):
            return jsonify({'forecast': forecast})
        else:
            return jsonify({'error': 'Prediction result is not in expected format'}), 500
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in /predict route: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
