import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Загрузка моделей и фичей  
model = joblib.load('/models/car_price_model.pkl')
with open('../models/feature_list.txt', 'r') as f:
    feature_list = f.read().split(',')

# Инициализация кодировщиков
label_encoders = {
    'fuel': LabelEncoder().fit(["Diesel", "Petrol", "CNG"]),
    'seller_type': LabelEncoder().fit(["Individual", "Dealer", "Trustmark Dealer"]),
    'transmission': LabelEncoder().fit(["Manual", "Automatic"]),
    'owner': LabelEncoder().fit(["First Owner", "Second Owner", "Third Owner"])
}

st.title('🚗 Предсказание стоимости автомобиля')

# Форма для ввода данных
with st.form("car_form"):
    # Числовые признаки
    year = st.slider("Год выпуска", 1990, 2023, 2018)
    km_driven = st.number_input("Пробег (км)", 0, 500000, 50000)
    engine = st.number_input("Объем двигателя (cc)", 500, 5000, 1500)
    max_power = st.number_input("Мощность (bhp)", 50, 500, 100)
    mileage = st.number_input("Пробег (kmpl)", 5, 50, 20)
    seats = st.number_input("Количество мест", 2, 10, 5)
    # Категориальные признаки
    fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG"])
    seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
    owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner"])
    
    submitted = st.form_submit_button("Предсказать цену")
    
    if submitted:
        # Кодируем категориальные признаки
        encoded_fuel = label_encoders['fuel'].transform([fuel])[0]
        encoded_seller = label_encoders['seller_type'].transform([seller_type])[0]
        encoded_trans = label_encoders['transmission'].transform([transmission])[0]
        encoded_owner = label_encoders['owner'].transform([owner])[0]

        # Преобразование ввода в формат модели
        input_data = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'fuel': [encoded_fuel],
            'seller_type': [encoded_seller],
            'transmission': [encoded_trans],
            'owner': [encoded_owner],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'seats': [seats]
        })
        # Убедимся, что порядок признаков соответствует ожиданиям модели
        input_data = input_data[feature_list]

        try:
            # Предсказание
            prediction = model.predict(input_data)[0]
            st.success(f"Предсказанная цена: ₹{prediction:,.2f}")

            # Визуализация
            st.subheader("Влияние параметров на цену")
            fig, ax = plt.subplots()
            features = ['year', 'mileage', 'engine_size', 'horsepower']
            importances = model.feature_importances_
            indices = np.argsort(importances)[-4:]
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_list[i] for i in indices])
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка предсказания: {str(e)}")
            st.write("Ожидаемые признаки:", feature_list)
            st.write("Переданные признаки:", input_data.columns.tolist())