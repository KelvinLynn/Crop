from flask import Flask, redirect, url_for, render_template, request, session, flash
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
import logging
from datetime import datetime

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'Teams')  # Sử dụng biến môi trường cho secret_key

# Danh sách người dùng (nên chuyển sang cơ sở dữ liệu trong thực tế)
USERS = [
    {"username": "nguoidung1", "password": "123456"},
    {"username": "phamductrung", "password": "admin123"}
]

# Load model và các file liên quan
try:
    model_path = os.path.abspath('../data/model.pkl')
    label_encoder_path = os.path.abspath('../data/label_encoder.pkl')
    scaler_path = os.path.abspath('../data/scaler.pkl')
    crop_names_path = os.path.abspath('../data/crop_names.pkl')

    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading label encoder from: {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)

    logger.info(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)

    logger.info(f"Loading crop names from: {crop_names_path}")
    crop_names = joblib.load(crop_names_path)

    CROP_NAMES = {i: name.capitalize() for i, name in enumerate(crop_names)}
    logger.info(f"CROP_NAMES mapping: {CROP_NAMES}")

except Exception as e:
    logger.error(f"Error loading files: {str(e)}")
    raise

CROPS_INFO = [
    {"id": 0, "name": "Apple", "N": "20-40 kg/ha", "P": "20-40 kg/ha", "K": "40-60 kg/ha", "temperature": "15-25°C", "humidity": "50-70%", "ph": "5.5-6.5", "rainfall": "100-150 mm", "image": "https://images.unsplash.com/photo-1560806887-1e4cd0b5cbd6"},
    {"id": 1, "name": "Banana", "N": "100-120 kg/ha", "P": "50-70 kg/ha", "K": "150-200 kg/ha", "temperature": "25-35°C", "humidity": "70-90%", "ph": "5.5-7.0", "rainfall": "150-250 mm", "image": "https://images.unsplash.com/photo-1603833665858-e61d17a0f716"},
    {"id": 2, "name": "Blackgram", "N": "20-40 kg/ha", "P": "40-60 kg/ha", "K": "20-40 kg/ha", "temperature": "25-35°C", "humidity": "60-80%", "ph": "5.5-7.0", "rainfall": "80-120 mm", "image": "https://via.placeholder.com/150?text=Blackgram"},
    {"id": 3, "name": "Chickpea", "N": "20-40 kg/ha", "P": "60-80 kg/ha", "K": "80-100 kg/ha", "temperature": "15-25°C", "humidity": "40-60%", "ph": "6.0-7.5", "rainfall": "40-60 mm", "image": "https://images.unsplash.com/photo-1606727396937-98e766876588"},
    {"id": 4, "name": "Coconut", "N": "50-80 kg/ha", "P": "30-50 kg/ha", "K": "100-150 kg/ha", "temperature": "25-35°C", "humidity": "60-80%", "ph": "5.0-8.0", "rainfall": "150-250 mm", "image": "https://images.unsplash.com/photo-1586209136933-656b765207e6"},
    {"id": 5, "name": "Coffee", "N": "80-100 kg/ha", "P": "20-40 kg/ha", "K": "20-40 kg/ha", "temperature": "15-25°C", "humidity": "50-70%", "ph": "5.0-6.0", "rainfall": "150-200 mm", "image": "https://images.unsplash.com/photo-1506619216599-9d16d0903dfd"},
    {"id": 6, "name": "Cotton", "N": "80-100 kg/ha", "P": "40-60 kg/ha", "K": "40-60 kg/ha", "temperature": "20-30°C", "humidity": "50-70%", "ph": "5.5-7.5", "rainfall": "80-120 mm", "image": "https://images.unsplash.com/photo-1599153504238-46618b2f71fc"},
    {"id": 7, "name": "Grapes", "N": "60-80 kg/ha", "P": "20-40 kg/ha", "K": "80-100 kg/ha", "temperature": "15-30°C", "humidity": "50-70%", "ph": "5.5-7.0", "rainfall": "50-100 mm", "image": "https://images.unsplash.com/photo-1590137870800-307d6b3e8336"},
    {"id": 8, "name": "Jute", "N": "80-100 kg/ha", "P": "20-40 kg/ha", "K": "20-40 kg/ha", "temperature": "25-35°C", "humidity": "70-90%", "ph": "5.5-7.0", "rainfall": "150-250 mm", "image": "https://via.placeholder.com/150?text=Jute"},
    {"id": 9, "name": "Kidneybeans", "N": "20-40 kg/ha", "P": "60-80 kg/ha", "K": "20-40 kg/ha", "temperature": "15-25°C", "humidity": "60-80%", "ph": "5.0-6.5", "rainfall": "80-120 mm", "image": "https://via.placeholder.com/150?text=Kidneybeans"},
    {"id": 10, "name": "Lentil", "N": "20-40 kg/ha", "P": "40-60 kg/ha", "K": "20-40 kg/ha", "temperature": "15-25°C", "humidity": "50-70%", "ph": "5.5-7.0", "rainfall": "40-80 mm", "image": "https://images.unsplash.com/photo-1606727396937-98e766876588"},
    {"id": 11, "name": "Maize", "N": "90-120 kg/ha", "P": "40-60 kg/ha", "K": "30-50 kg/ha", "temperature": "18-30°C", "humidity": "50-70%", "ph": "5.8-7.0", "rainfall": "50-100 mm", "image": "https://images.unsplash.com/photo-1604538986653-a1d07b72e8d3"},
    {"id": 12, "name": "Mango", "N": "50-80 kg/ha", "P": "20-40 kg/ha", "K": "40-60 kg/ha", "temperature": "25-35°C", "humidity": "50-70%", "ph": "5.5-7.0", "rainfall": "100-150 mm", "image": "https://images.unsplash.com/photo-1591070681808-5423ca454d98"},
    {"id": 13, "name": "Mothbeans", "N": "20-40 kg/ha", "P": "40-60 kg/ha", "K": "20-40 kg/ha", "temperature": "25-35°C", "humidity": "50-70%", "ph": "5.5-7.0", "rainfall": "40-80 mm", "image": "https://via.placeholder.com/150?text=Mothbeans"},
    {"id": 14, "name": "Mungbean", "N": "20-40 kg/ha", "P": "40-60 kg/ha", "K": "20-40 kg/ha", "temperature": "25-35°C", "humidity": "50-70%", "ph": "6.0-7.0", "rainfall": "80-120 mm", "image": "https://via.placeholder.com/150?text=Mungbean"},
    {"id": 15, "name": "Muskmelon", "N": "80-100 kg/ha", "P": "20-40 kg/ha", "K": "40-60 kg/ha", "temperature": "25-35°C", "humidity": "60-80%", "ph": "6.0-7.0", "rainfall": "50-100 mm", "image": "https://images.unsplash.com/photo-1594051700589-44e7b6e04e2c"},
    {"id": 16, "name": "Orange", "N": "50-80 kg/ha", "P": "20-40 kg/ha", "K": "40-60 kg/ha", "temperature": "15-30°C", "humidity": "50-70%", "ph": "5.5-7.0", "rainfall": "100-150 mm", "image": "https://images.unsplash.com/photo-1582979512210-99b6a53386f9"},
    {"id": 17, "name": "Papaya", "N": "80-100 kg/ha", "P": "40-60 kg/ha", "K": "40-60 kg/ha", "temperature": "25-35°C", "humidity": "60-80%", "ph": "5.5-7.0", "rainfall": "100-150 mm", "image": "https://images.unsplash.com/photo-1521093470119-b2d960e76d56"},
    {"id": 18, "name": "Pigeonpeas", "N": "20-40 kg/ha", "P": "40-60 kg/ha", "K": "20-40 kg/ha", "temperature": "25-35°C", "humidity": "50-70%", "ph": "5.5-7.0", "rainfall": "80-120 mm", "image": "https://via.placeholder.com/150?text=Pigeonpeas"},
    {"id": 19, "name": "Pomegranate", "N": "50-80 kg/ha", "P": "20-40 kg/ha", "K": "40-60 kg/ha", "temperature": "20-35°C", "humidity": "50-70%", "ph": "5.5-7.0", "rainfall": "50-100 mm", "image": "https://images.unsplash.com/photo-1603568291637-309f842b8e91"},
    {"id": 20, "name": "Rice", "N": "80-100 kg/ha", "P": "35-45 kg/ha", "K": "35-45 kg/ha", "temperature": "20-35°C", "humidity": "70-90%", "ph": "5.5-7.0", "rainfall": "180-250 mm", "image": "https://images.unsplash.com/photo-1592984548732-8b46f738ba18"},
    {"id": 21, "name": "Watermelon", "N": "80-100 kg/ha", "P": "20-40 kg/ha", "K": "40-60 kg/ha", "temperature": "25-35°C", "humidity": "60-80%", "ph": "6.0-7.0", "rainfall": "50-100 mm", "image": "https://images.unsplash.com/photo-1589985270826-4b7bb135bc7d"}
]

CROPS_INFO_HANDLER = [
    {
        "id": 1,
        "name": "Táo chua",
        "type": "Apple",
        "image": "https://thegioihatgiong.com/wp-content/uploads/2021/07/Hat-tao-chua.jpg",
        "description": "Apple is a sweet fruit that is often red, green, or yellow."
    },
    {
        "id": 2,
        "name": "Táo tàu",
        "type": "Apple",
        "image": "https://vnn-imgs-f.vgcloud.vn/2020/11/07/08/tao-tau-1.jpg",
        "description": "Apple is a sweet fruit that is often red, green, or yellow."
    },
    {
        "id": 3,
        "name": "Táo Fuji",
        "type": "Apple",
        "image": "https://file.hstatic.net/200000439247/article/tao_fuji_be77e8054173413494c97f613c8e82d3.jpg",
        "description": "Fuji apple is a crisp and juicy variety with a sweet flavor."
    },
    # Banana Varieties
    {
        "id": 4,
        "name": "Chuối tiêu",
        "type": "Banana",
        "image": "https://example.com/chuoi-tieu.jpg",
        "description": "Banana is a tropical fruit that is yellow when ripe."
    },
    {
        "id": 5,
        "name": "Chuối lùn",
        "type": "Banana",
        "image": "https://example.com/chuoi-lun.jpg",
        "description": "A dwarf variety of banana with sweet flavor."
    },
    {
        "id": 8,
        "name": "Chuối ngự",
        "type": "Banana",
        "image": "https://example.com/chuoi-ngu.jpg",
        "description": "A royal banana variety known for its aroma."
    },
    
    # Blackgram Varieties
    {
        "id": 11,
        "name": "Đậu đen nhỏ",
        "type": "Blackgram",
        "image": "https://example.com/dau-den-nho.jpg",
        "description": "A small black gram used in soups and stews."
    },
    {
        "id": 12,
        "name": "Đậu đen lớn",
        "type": "Blackgram",
        "image": "https://example.com/dau-den-lon.jpg",
        "description": "A larger black gram variety for cooking."
    },
    # ... (continuing with 3 more varieties for Blackgram and so on for other crops)
    {
        "id": 13,
        "name": "Đậu đen hạt mịn",
        "type": "Blackgram",
        "image": "https://example.com/dau-den-min.jpg",
        "description": "A smooth-textured black gram for grinding."
    },
    
    # Chickpea Varieties
    {
        "id": 16,
        "name": "Đậu gà trắng",
        "type": "Chickpea",
        "image": "https://example.com/dau-ga-trang.jpg",
        "description": "A white chickpea variety for hummus."
    },
    {
        "id": 17,
        "name": "Đậu gà đen",
        "type": "Chickpea",
        "image": "https://example.com/dau-ga-den.jpg",
        "description": "A black chickpea with a nutty flavor."
    },
    {
        "id": 18,
        "name": "Đậu gà lớn",
        "type": "Chickpea",
        "image": "https://example.com/dau-ga-lon.jpg",
        "description": "A large chickpea variety for roasting."
    },
    
    # Coconut Varieties
    {
        "id": 21,
        "name": "Dừa xiêm",
        "type": "Coconut",
        "image": "https://example.com/dua-xiem.jpg",
        "description": "A tall coconut variety with sweet water."
    },
    {
        "id": 22,
        "name": "Dừa lùn",
        "type": "Coconut",
        "image": "https://example.com/dua-lun.jpg",
        "description": "A dwarf coconut with creamy flesh."
    },
    {
        "id": 23,
        "name": "Dừa vàng",
        "type": "Coconut",
        "image": "https://example.com/dua-vang.jpg",
        "description": "A yellow coconut variety."
    },
    
    # Coffee Varieties
    {
        "id": 26,
        "name": "Cà phê robusta",
        "type": "Coffee",
        "image": "https://example.com/ca-phe-robusta.jpg",
        "description": "A robust coffee with strong flavor."
    },
    {
        "id": 27,
        "name": "Cà phê arabica",
        "type": "Coffee",
        "image": "https://example.com/ca-phe-arabica.jpg",
        "description": "An arabica coffee with mild taste."
    },
    {
        "id": 28,
        "name": "Cà phê liberica",
        "type": "Coffee",
        "image": "https://example.com/ca-phe-liberica.jpg",
        "description": "A liberica coffee with unique aroma."
    },
    
    # Cotton Varieties
    {
        "id": 31,
        "name": "Bông vải trắng",
        "type": "Cotton",
        "image": "https://example.com/bong-vai-trang.jpg",
        "description": "A white cotton variety."
    },
    {
        "id": 32,
        "name": "Bông vải màu",
        "type": "Cotton",
        "image": "https://example.com/bong-vai-mau.jpg",
        "description": "A colored cotton variety."
    },
    {
        "id": 33,
        "name": "Bông vải dài",
        "type": "Cotton",
        "image": "https://example.com/bong-vai-dai.jpg",
        "description": "A long-fiber cotton variety."
    },
   
    # Grape Varieties
    {
        "id": 36,
        "name": "Nho xanh",
        "type": "Grape",
        "image": "https://example.com/nho-xanh.jpg",
        "description": "A green grape with crisp taste."
    },
    {
        "id": 37,
        "name": "Nho đỏ",
        "type": "Grape",
        "image": "https://example.com/nho-do.jpg",
        "description": "A red grape with sweet flavor."
    },
    {
        "id": 38,
        "name": "Nho tím",
        "type": "Grape",
        "image": "https://example.com/nho-tim.jpg",
        "description": "A purple grape variety."
    },
    
    # Jute Varieties
    {
        "id": 41,
        "name": "Đay trắng",
        "type": "Jute",
        "image": "https://example.com/day-trang.jpg",
        "description": "A white jute variety for fiber."
    },
    {
        "id": 42,
        "name": "Đay đen",
        "type": "Jute",
        "image": "https://example.com/day-den.jpg",
        "description": "A black jute variety."
    },
    {
        "id": 43,
        "name": "Đay dài",
        "type": "Jute",
        "image": "https://example.com/day-dai.jpg",
        "description": "A long-fiber jute variety."
    },
    
    # Kidneybean Varieties
    {
        "id": 46,
        "name": "Đậu thận đỏ",
        "type": "Kidneybean",
        "image": "https://example.com/dau-than-do.jpg",
        "description": "A red kidney bean variety."
    },
    {
        "id": 47,
        "name": "Đậu thận trắng",
        "type": "Kidneybean",
        "image": "https://example.com/dau-than-trang.jpg",
        "description": "A white kidney bean variety."
    },
    {
        "id": 48,
        "name": "Đậu thận lớn",
        "type": "Kidneybean",
        "image": "https://example.com/dau-than-lon.jpg",
        "description": "A large kidney bean variety."
    },
    
    # Lentil Varieties
    {
        "id": 51,
        "name": "Đậu lăng xanh",
        "type": "Lentil",
        "image": "https://example.com/dau-lang-xanh.jpg",
        "description": "A green lentil variety."
    },
    {
        "id": 52,
        "name": "Đậu lăng đỏ",
        "type": "Lentil",
        "image": "https://example.com/dau-lang-do.jpg",
        "description": "A red lentil variety."
    },
    {
        "id": 53,
        "name": "Đậu lăng đen",
        "type": "Lentil",
        "image": "https://example.com/dau-lang-den.jpg",
        "description": "A black lentil variety."
    },
    
    # Maize Varieties
    {
        "id": 56,
        "name": "Ngô ngọt",
        "type": "Maize",
        "image": "https://example.com/ngo-ngot.jpg",
        "description": "A sweet maize variety."
    },
    {
        "id": 57,
        "name": "Ngô nếp",
        "type": "Maize",
        "image": "https://example.com/ngo-nep.jpg",
        "description": "A sticky maize variety."
    },
    {
        "id": 58,
        "name": "Ngô vàng",
        "type": "Maize",
        "image": "https://example.com/ngo-vang.jpg",
        "description": "A yellow maize variety."
    },
   
    # Mango Varieties
    {
        "id": 61,
        "name": "Xoài cát",
        "type": "Mango",
        "image": "https://example.com/xoai-cat.jpg",
        "description": "A sweet and juicy mango variety."
    },
    {
        "id": 62,
        "name": "Xoài tượng",
        "type": "Mango",
        "image": "https://example.com/xoai-tuong.jpg",
        "description": "A large mango variety."
    },
    {
        "id": 63,
        "name": "Xoài thái",
        "type": "Mango",
        "image": "https://example.com/xoai-thai.jpg",
        "description": "A Thai mango with small size."
    },
    
    # Mothbean Varieties
    {
        "id": 66,
        "name": "Đậu moth xanh",
        "type": "Mothbean",
        "image": "https://example.com/dau-moth-xanh.jpg",
        "description": "A green mothbean variety."
    },
    {
        "id": 67,
        "name": "Đậu moth vàng",
        "type": "Mothbean",
        "image": "https://example.com/dau-moth-vang.jpg",
        "description": "A yellow mothbean variety."
    },
    {
        "id": 68,
        "name": "Đậu moth lớn",
        "type": "Mothbean",
        "image": "https://example.com/dau-moth-lon.jpg",
        "description": "A large mothbean variety."
    },
    
    # Mungbean Varieties
    {
        "id": 71,
        "name": "Đậu xanh nhỏ",
        "type": "Mungbean",
        "image": "https://example.com/dau-xanh-nho.jpg",
        "description": "A small mungbean variety."
    },
    {
        "id": 72,
        "name": "Đậu xanh lớn",
        "type": "Mungbean",
        "image": "https://example.com/dau-xanh-lon.jpg",
        "description": "A large mungbean variety."
    },
    {
        "id": 73,
        "name": "Đậu xanh vàng",
        "type": "Mungbean",
        "image": "https://example.com/dau-xanh-vang.jpg",
        "description": "A yellow mungbean variety."
    },
   
    # Muskmelon Varieties
    {
        "id": 76,
        "name": "Dưa lưới ngọt",
        "type": "Muskmelon",
        "image": "https://example.com/dua-luoi-ngot.jpg",
        "description": "A sweet muskmelon variety."
    },
    {
        "id": 77,
        "name": "Dưa lưới vàng",
        "type": "Muskmelon",
        "image": "https://example.com/dua-luoi-vang.jpg",
        "description": "A yellow muskmelon variety."
    },
    {
        "id": 78,
        "name": "Dưa lưới xanh",
        "type": "Muskmelon",
        "image": "https://example.com/dua-luoi-xanh.jpg",
        "description": "A green muskmelon variety."
    },
    
    # Orange Varieties
    {
        "id": 81,
        "name": "Cam ngọt",
        "type": "Orange",
        "image": "https://example.com/cam-ngot.jpg",
        "description": "A sweet orange variety."
    },
    {
        "id": 82,
        "name": "Cam chua",
        "type": "Orange",
        "image": "https://example.com/cam-chua.jpg",
        "description": "A tart orange variety."
    },
    {
        "id": 83,
        "name": "Cam vàng",
        "type": "Orange",
        "image": "https://example.com/cam-vang.jpg",
        "description": "A yellow orange variety."
    },
    
    # Papaya Varieties
    {
        "id": 86,
        "name": "Đu đủ xanh",
        "type": "Papaya",
        "image": "https://example.com/du-du-xanh.jpg",
        "description": "A green papaya variety."
    },
    {
        "id": 87,
        "name": "Đu đủ vàng",
        "type": "Papaya",
        "image": "https://example.com/du-du-vang.jpg",
        "description": "A yellow papaya variety."
    },
    {
        "id": 88,
        "name": "Đu đủ đỏ",
        "type": "Papaya",
        "image": "https://example.com/du-du-do.jpg",
        "description": "A red papaya variety."
    },
    
    # Pigeonpea Varieties
    {
        "id": 91,
        "name": "Đậu pigeon xanh",
        "type": "Pigeonpea",
        "image": "https://example.com/dau-pigeon-xanh.jpg",
        "description": "A green pigeonpea variety."
    },
    {
        "id": 92,
        "name": "Đậu pigeon đỏ",
        "type": "Pigeonpea",
        "image": "https://example.com/dau-pigeon-do.jpg",
        "description": "A red pigeonpea variety."
    },
    {
        "id": 93,
        "name": "Đậu pigeon lớn",
        "type": "Pigeonpea",
        "image": "https://example.com/dau-pigeon-lon.jpg",
        "description": "A large pigeonpea variety."
    },
    
    # Pomegranate Varieties
    {
        "id": 96,
        "name": "Lựu đỏ",
        "type": "Pomegranate",
        "image": "https://example.com/luu-do.jpg",
        "description": "A red pomegranate variety."
    },
    {
        "id": 97,
        "name": "Lựu trắng",
        "type": "Pomegranate",
        "image": "https://example.com/luu-trang.jpg",
        "description": "A white pomegranate variety."
    },
    {
        "id": 98,
        "name": "Lựu lớn",
        "type": "Pomegranate",
        "image": "https://example.com/luu-lon.jpg",
        "description": "A large pomegranate variety."
    },
    
    # Rice Varieties
    {
        "id": 101,
        "name": "Gạo nếp",
        "type": "Rice",
        "image": "https://example.com/gao-nep.jpg",
        "description": "A sticky rice variety."
    },
    {
        "id": 102,
        "name": "Gạo tẻ",
        "type": "Rice",
        "image": "https://example.com/gao-te.jpg",
        "description": "A non-sticky rice variety."
    },
    {
        "id": 103,
        "name": "Gạo lứt",
        "type": "Rice",
        "image": "https://example.com/gao-lut.jpg",
        "description": "A brown rice variety."
    },
   
    # Watermelon Varieties
    {
        "id": 106,
        "name": "Dưa hấu đỏ",
        "type": "Watermelon",
        "image": "https://example.com/dua-hau-do.jpg",
        "description": "A red watermelon variety."
    },
    {
        "id": 107,
        "name": "Dưa hấu vàng",
        "type": "Watermelon",
        "image": "https://example.com/dua-hau-vang.jpg",
        "description": "A yellow watermelon variety."
    },
    {
        "id": 108,
        "name": "Dưa hấu đen",
        "type": "Watermelon",
        "image": "https://example.com/dua-hau-den.jpg",
        "description": "A black watermelon variety."
    }
]

def authenticate_user(username, password):
    """Xác thực thông tin đăng nhập của người dùng."""
    user = next((u for u in USERS if u["username"] == username and u["password"] == password), None)
    return user

@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    logger.info(f"Accessing index - Session: {session}")
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Kiểm tra số lần thử đăng nhập sai
        session['login_attempts'] = session.get('login_attempts', 0)
        if session['login_attempts'] >= 3:
            return render_template('index.html', error="Quá nhiều lần thử đăng nhập. Vui lòng thử lại sau 5 phút.")

        user = authenticate_user(username, password)
        if user:
            session['username'] = username
            session['isLoggedIn'] = True
            session['login_attempts'] = 0  # Reset số lần thử
            logger.info(f"Successful login - Username: {username}, Session: {session}")
            return redirect(url_for('home'))
        else:
            session['login_attempts'] += 1
            logger.warning(f"Failed login attempt - Username: {username}, Attempts: {session['login_attempts']}")
            return render_template('index.html', error="Tên đăng nhập hoặc mật khẩu không đúng!")
    return render_template('index.html')

@app.route('/home')
def home():
    logger.info(f"Accessing home - Session: {session}")
    if not session.get('isLoggedIn'):
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('isLoggedIn'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        try:
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosphorous'])
            K = float(request.form['Potassium'])
            Temperature = float(request.form['Temperature'])
            Humidity = float(request.form['Humidity'])
            PH = float(request.form['PH'])
            Rainfall = float(request.form['Rainfall'])

            # Kiểm tra giá trị đầu vào hợp lệ
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

            # Chuẩn hóa dữ liệu đầu vào với tên cột
            input_data = pd.DataFrame(
                [[N, P, K, Temperature, Humidity, PH, Rainfall]],
                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            )
            input_data_scaled = scaler.transform(input_data)

            # Dự đoán cây trồng
            prediction = model.predict(input_data_scaled)
            predicted_class = int(prediction[0])
            crop_name = CROP_NAMES.get(predicted_class, f"Unknown label: {predicted_class}")

            # Tính điểm số phù hợp (suitability scores) cho tất cả cây trồng
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
                total_percentage = sum(p['probability'] for p in probabilities)
                logger.info(f"Input data: {input_data.values[0]}")
                logger.info(f"Suitability scores (percentages): {probabilities}")
                logger.info(f"Sum of percentages: {total_percentage}")
                logger.info(f"Number of non-zero scores: {sum(1 for p in probabilities if p['probability'] > 0)}")

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
            return render_template('home.html', error=f"Đã xảy ra lỗi: {str(e)}")
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/crops_info')
def crops_info():
    return render_template('crops_info.html', crops=CROPS_INFO, crops_info=CROPS_INFO_HANDLER)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            message = request.form['message']
            logger.info(f"Contact Form: Name={name}, Email={email}, Message={message}")
            return render_template('contact.html', success="Gửi tin nhắn thành công!")
        except Exception as e:
            logger.error(f"Contact form error: {str(e)}")
            return render_template('contact.html', error=f"Đã xảy ra lỗi: {str(e)}")
    return render_template('contact.html')

@app.route('/user/<usr>')
def user(usr):
    return f"<h1>Hi {usr}!</h1>"

@app.route('/detail/<int:id>')
def detail(id):
    list = []
    name = ""
    typer = ""
    image = ""
    description = ""
    for crop in CROPS_INFO_HANDLER:
        iddata = crop['id']
        if int(iddata) == int(id):
            name = crop['name']
            typer = crop['type']
            image = crop['image']
            description = crop['description']
    
    for crop in CROPS_INFO_HANDLER:
        if crop['type'] == typer and crop['id'] != id:
            list.append(crop)
    return render_template('detail.html', namer=name, typerr=typer, imager=image, descriptionr=description, listr=list)

@app.route('/logout')
def logout():
    logger.info(f"Logging out - Session before: {session}")
    session.pop('isLoggedIn', None)
    session.pop('username', None)
    session.pop('login_attempts', None)
    logger.info(f"Logged out - Session after: {session}")
    flash('Bạn đã đăng xuất thành công!', 'success')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return render_template('register.html', error="Chức năng đăng ký chưa được triển khai.")
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)