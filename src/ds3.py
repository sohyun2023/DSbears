

import streamlit as st
import pandas as pd
import numpy as np
import autogluon
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import OneHotEncoder
import base64

# Set the page to wide mode by default
st.set_page_config(layout="wide")

# Function to load an image and encode it to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 배경경로
background_path = 'data/잠실구장배경사진.png'

# Encode the background image to base64
background_base64 = get_base64_image(background_path)

# Background image and transparency settings
background_html = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{background_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}
.block-container {{
    background-color: rgba(255, 255, 255, 0.7); /* Adjusted transparency */
    border-radius: 30px;
    padding: 30px;
    width: 90%; 
    margin: auto; 
}}
.bold-prediction {{
    font-size: 34px;
    font-weight: bold;
    color: #142CA8;
    text-align: center;
    margin-top: 20px;
}}
.center-logo {{
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 20px;
}}
.stButton > button {{
    font-size: 50px !important;
    font-weight: bold !important;
    padding: 20px 30px !important;
    border-radius: 30px !important;
    background-color: #3132AA !important;
    color: white !important;
    border: none !important;
}}
.stButton > button:hover {{
    background-color: #FF8347 !important;
}}
.centered-title {{
    text-align: center;
    font-size: 3.5em;
    font-weight: bold;
    margin-bottom: 20px;
}}
.stLabel {{
    font-size: 50px !important; /* Increased font size */
    font-weight: bold !important; /* Made font bold */
}}
.prediction-result {{
    position: fixed;
    bottom: 100px;
    left: 48%;
    transform: translateX(-48%);
    width: 50%;
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #3132AA; /* Changed color to make it stand out */
    background-color: rgba(255, 255, 255, 1);
    padding: 20px;
    border-radius: 30px;
}}
.button-container {{
    display: flex;
    justify-content: center;
    margin-top: 20px;
}}
</style>
"""

st.markdown(background_html, unsafe_allow_html=True)

# Centered title
st.markdown('<h1 class="centered-title">⚾️ KBO 리그 경기 예상 관중 수 예측 ⚾️</h1>', unsafe_allow_html=True)

# 로고 이미지 경로
logo_path = 'data/두산사데연로고.png'
st.markdown(f'<div class="center-logo"><img src="data:image/png;base64,{get_base64_image(logo_path)}" width="400"/></div>', unsafe_allow_html=True)

# 매핑 사전 정의
stadium_mapping = {
    '두산': '서울 잠실 야구장',
    'SSG': '인천SSG 랜더스필드',
    '키움': '서울고척스카이돔',
    'LG': '서울 잠실 야구장',
    'KIA': '광주-기아 챔피언스 필드',
    '한화': '대전 한화생명 이글스파크',
    '롯데': '부산 사직 야구장',
    '삼성': '대구 삼성 라이온즈 파크',
    'NC': '창원NC파크',
    'KT': '수원케이티위즈파크'
}

# 사용자로부터 데이터 입력 받기
col1, col2, col3 = st.columns(3)
with col1:
    week_x = st.selectbox('요일', ['월', '화', '수', '목', '금', '토', '일'])
    temp_x = st.number_input('온도')
    home = st.selectbox('홈팀', ['두산', 'SSG', '키움', 'LG', 'KIA', '한화', '롯데', '삼성', 'NC', 'KT'])
    away = st.selectbox('원정팀', ['두산', 'SSG', '키움', 'LG', 'KIA', '한화', '롯데', '삼성', 'NC', 'KT'])

with col2:
    date = st.date_input('날짜')
    weather_x = st.selectbox('날씨', ['맑음', '구름조금', '구름많음', '흐림', '비'])
    winloss_home = st.number_input('홈팀 승패')
    winloss_away = st.number_input('원정팀 승패')

with col3:
    holiday_x = st.selectbox('공휴일:1 / 비공휴일:0', [1, 0])
    rainfall = st.number_input('강수량(mm)')
    winrate_home = st.number_input('홈팀 승률')
    winrate_away = st.number_input('원정팀 승률')

# stadium_x 값 자동 할당
stadium_x = stadium_mapping[home]

# 입력된 데이터를 DataFrame으로 변환
input_data = pd.DataFrame({
    'week_x': [week_x],
    'home': [home],
    'away': [away],
    'stadium_x': [stadium_x],
    'weather_x': [weather_x],
    'temp_x': [temp_x],
    '강수량(mm)': [rainfall],
    'date': [date],
    'holiday_x': [holiday_x],
    'winloss_home': [winloss_home],
    'winloss_away': [winloss_away],
    'winrate_home': [winrate_home],
    'winrate_away': [winrate_away]
})

def preprocess_data_fixed(data):
    data['date'] = pd.to_datetime(data['date'])
    
    # Mapping weekday names to numbers
    day_name_num = {'월': 1, '화': 2, '수': 3, '목': 4, '금': 5, '토': 6, '일': 7}
    data['day_num'] = data['week_x'].map(day_name_num)
    data['day_sin'] = np.sin(2 * np.pi * data['day_num'] / 7.0)
    data['day_cos'] = np.cos(2 * np.pi * data['day_num'] / 7.0)
    data.drop(['week_x', 'day_num'], axis=1, inplace=True)

    # Encoding for weather_x
    all_weather = ['맑음', '구름조금', '구름많음', '흐림', '비']
    for weather in all_weather:
        data[f'weather_x_{weather}'] = (data['weather_x'] == weather).astype(int)
    data.drop(['weather_x'], axis=1, inplace=True)

    # Encoding for stadium_x
    all_stadiums = ['대구 삼성 라이온즈 파크', '대전 한화생명 이글스파크', '마산', '부산 사직 야구장', '서울 잠실 야구장', '울산문수야구장', '창원NC파크', '청주야구장', '포항야구장', '광주-기아 챔피언스 필드', '인천SSG 랜더스필드', '서울고척스카이돔', '수원케이티위즈파크']
    for stadium in all_stadiums:
        data[f'stadium_x_{stadium}'] = (data['stadium_x'] == stadium).astype(int)
    data.drop(['stadium_x'], axis=1, inplace=True)

    teams = ['두산', 'SSG', '키움', 'LG', 'KIA', '한화', '롯데', '삼성', 'NC', 'KT']
    for team in teams:
        data[team] = ((data['home'] == team) | (data['away'] == team)).astype(int)

    # Drop original home and away columns
    data.drop(['home', 'away'], axis=1, inplace=True)

    return data

# 전처리 수행
processed_data = preprocess_data_fixed(input_data)

# Autogluon 모델 로드
model = TabularPredictor.load('models/ag-20240521_003925')
# Get the best model name
best_model = model.get_model_best()
# Extract the best model
best_model = predictor._trainer.load_model(best_model)
# '예측' 버튼 클릭
if st.button('GO⚾️'):
    # spec 값을 예측
    prediction = model.predict(processed_data,model=best_model)

    # 사용자로부터 입력 받은 경기장 이름
    stadium_name = stadium_x

    # 해당 경기장의 실제 관중석 규모
    stadium_capacity = {
        '대구 삼성 라이온즈 파크': 24000,
        '대전 한화생명 이글스파크': 13000,
        '마산': 11000,
        '부산 사직 야구장': 23500,
        '서울 잠실 야구장': 25000,
        '울산문수야구장': 12050,
        '창원NC파크': 22000,
        '청주야구장': 9580,
        '포항야구장': 12000,
        '광주-기아 챔피언스 필드': 20500,
        '인천SSG 랜더스필드': 23000,
        '서울고척스카이돔': 17000,
        '수원케이티위즈파크': 20000
    }

    # 실제 관중 수 계산
    if stadium_name in stadium_capacity:
        actual_attendance = prediction.iloc[0] * stadium_capacity[stadium_name]
        result_text = f'{stadium_name} 경기장의 예상 관중 수:<br>{actual_attendance:.0f}'
    else:
        result_text = '해당하는 경기장 이름을 찾을 수 없습니다.'

    # 결과 표시
    st.markdown(f'<div class="prediction-result">{result_text}</div>', unsafe_allow_html=True)
