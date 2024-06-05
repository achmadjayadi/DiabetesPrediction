import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Tambahkan import scaler

# Membaca model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Inisialisasi scaler dengan skala yang sama yang digunakan saat training
scaler = StandardScaler()
# Contoh skala yang sama yang digunakan saat training, sesuaikan dengan skala yang benar
scaler.mean_ = [3.55, 121.69, 72.41, 29.15, 155.55, 33.09, 0.47, 33.24]
scaler.scale_ = [3.38, 30.53, 12.34, 10.48, 118.77, 7.28, 0.33, 11.76]

# Judul web
st.title('Prediksi Diabetes')

# Link untuk kembali sebagai tombol dengan latar belakang warna biru dan teks putih
st.markdown("""
    <style>
    .button {
        display: inline-block;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        text-align: center;
        color: #fff;  /* Warna teks putih */
        background-color: #79c8da;
        border: none;
        border-radius: 15px;
        padding: 0.25em 0.75em;
    }
    </style>
    <a href="https://dhcc.netlify.app/" class="button">Kembali ke halaman utama</a>
""", unsafe_allow_html=True)


# Inisialisasi session state
if 'Pregnancies' not in st.session_state:
    st.session_state['Pregnancies'] = ''
if 'BloodPressure' not in st.session_state:
    st.session_state['BloodPressure'] = ''
if 'SkinThickness' not in st.session_state:
    st.session_state['SkinThickness'] = ''
if 'DiabetesPedigreeFunction' not in st.session_state:
    st.session_state['DiabetesPedigreeFunction'] = ''
if 'Glucose' not in st.session_state:
    st.session_state['Glucose'] = ''
if 'Insulin' not in st.session_state:
    st.session_state['Insulin'] = ''
if 'BMI' not in st.session_state:
    st.session_state['BMI'] = ''
if 'Age' not in st.session_state:
    st.session_state['Age'] = ''

# Kolom
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Input Nilai Pregnancies', st.session_state['Pregnancies'], key='Pregnancies')
    Glucose = st.text_input('Input Nilai Glucose', st.session_state['Glucose'], key='Glucose')
    BloodPressure = st.text_input('Input Nilai BloodPressure', st.session_state['BloodPressure'], key='BloodPressure')
    SkinThickness = st.text_input('Input Nilai SkinThickness', st.session_state['SkinThickness'], key='SkinThickness')
with col2:
    Insulin = st.text_input('Input Nilai Insulin', st.session_state['Insulin'], key='Insulin')
    BMI = st.text_input('Input Nilai BMI', st.session_state['BMI'], key='BMI')
    DiabetesPedigreeFunction = st.text_input('Input Nilai DiabetesPedigreeFunction', st.session_state['DiabetesPedigreeFunction'], key='DiabetesPedigreeFunction')
    Age = st.text_input('Input Nilai Age', st.session_state['Age'], key='Age')

# Code
diab_diagnosis = ''

# Button Prediksi
if st.button('Tes Prediksi Diabetes'):
    # Standarisasi data input sebelum prediksi
    input_data = [[float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]]
    std_data = scaler.transform(input_data)
    
    diab_prediction = diabetes_model.predict(std_data)
    if diab_prediction[0] == 1:
        diab_diagnosis = 'Pasien Terkena Diabetes'
    else:
        diab_diagnosis = 'Pasien Tidak Terkena Diabetes'

# Tampilkan hasil prediksi
st.success(diab_diagnosis)

# Tampilkan penjelasan dari setiap inputan
st.markdown('### Penjelasan Input:')
st.markdown('**Pregnancies:** Jumlah kehamilan yang dialami oleh pasien.')
st.markdown('**Glucose:** Kadar glukosa plasma 2 jam dalam tes toleransi glukosa oral.')
st.markdown('**BloodPressure:** Tekanan darah diastolik (mm Hg).')
st.markdown('**SkinThickness:** Ketebalan lipatan kulit trisep (mm).')
st.markdown('**Insulin:** Kadar insulin serum 2 jam (mu U/ml).')
st.markdown('**BMI:** Indeks Massa Tubuh (berat dalam kg/(tinggi dalam meter)^2).')
st.markdown('**DiabetesPedigreeFunction:** Fungsi silsilah diabetes (menunjukkan riwayat diabetes dalam keluarga dan pengaruhnya terhadap risiko diabetes).')
st.markdown('**Age:** Usia pasien (tahun).')

# Tambahkan tabel contoh inputan
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70],
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45],
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df_contoh_inputan = pd.DataFrame(data)

st.markdown('### Contoh Inputan:')
st.write(df_contoh_inputan)
