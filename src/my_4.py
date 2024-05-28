# src/my_4_copy.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('data/all.csv')
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = df[~df['year'].isin([2020, 2021, 2022])]
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)

    df_rain = pd.read_csv('data/강수량.csv', encoding='cp949')
    df_rain = df_rain.astype({'일시': 'datetime64[ns]'})
    df_rain = df_rain.drop(columns=['지점명', '1시간최다강수량시각'])
    df_rain = df_rain.fillna({'강수량(mm)': 0, '1시간최다강수량(mm)': 0})

    df['date'] = pd.to_datetime(df['date'])
    df = pd.merge(df, df_rain, left_on='date', right_on='일시', how='left')
    df['연'] = df['date'].dt.year
    df['월'] = df['date'].dt.month
    df['일'] = df['date'].dt.day
    df = df.drop(columns=['일시', '1시간최다강수량(mm)'])

    df['weekday'] = df['date'].dt.day_of_week
    import holidays
    holiday_list = holidays.KR(years=[2016, 2017, 2018, 2019, 2023])
    df['holiday'] = 0
    df.loc[df['weekday'].isin(range(5, 7)), 'holiday'] = 1
    df.loc[df['date'].isin(holiday_list), 'holiday'] = 1
    df.drop(['weekday'], axis=1, inplace=True)

    df['temp'] = df['temp'].str.replace('℃', '').str.strip()
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['spec'] = pd.to_numeric(df['spec'], errors='coerce')

    df_sc = pd.read_csv('data/score.csv')

    def clean_data(df_sc):
        df_sc = df_sc.rename(columns={'year': '연', 'month': '월', 'day': '일'})
        return df_sc

    df_sc = clean_data(df_sc.copy())
    merged_df = pd.merge(df, df_sc, on=['연', '월', '일', 'home', 'away'], how='left')

    def clean_data(merged_df):
        merged_df = merged_df.drop(columns=['ID', 'week_y', 'weekend', 'streak_home', 'streak_away', 'score_home', 'score_away', 'winner', 'loser', 'stadium_y', 'weather_y', 'weather_int', 'is_rain', 'precip', 'day_sin', 'day_cos', 'temp_y', 'spec_y', 'weather_encoded', 'holiday_y', 'time'])
        merged_df = merged_df.rename(columns={'spec_x': 'spec'})
        return merged_df

    merged_df = clean_data(merged_df.copy())

    stadium_spec = {
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

    merged_df['adjusted_spec'] = merged_df.apply(lambda row: row['spec'] / stadium_spec[row['stadium_x']], axis=1)
    merged_df.drop(columns=['spec'], inplace=True)

    day_name_num = {'월': 1, '화': 2, '수': 3, '목': 4, '금': 5, '토': 6, '일': 7}
    merged_df['day_num'] = merged_df['week_x'].map(day_name_num)
    merged_df['day_sin'] = np.sin(2 * np.pi * merged_df['day_num'] / 7.0)
    merged_df['day_cos'] = np.cos(2 * np.pi * merged_df['day_num'] / 7.0)
    merged_df.drop(['week_x'], axis=1, inplace=True)

    teams = ['두산', 'SSG', '키움', 'LG', 'KIA', '한화', '롯데', '삼성', 'NC', 'KT']
    for team in teams:
        merged_df[team] = ((merged_df['home'] == team) | (merged_df['away'] == team)).astype(int)

    merged_df.drop(['home', 'away'], axis=1, inplace=True)

    all_weather = ['맑음', '구름조금', '구름많음', '흐림', '비']
    for weather in all_weather:
        merged_df[f'weather_x_{weather}'] = (merged_df['weather_x'] == weather).astype(int)
    merged_df.drop(['weather_x'], axis=1, inplace=True)

    encoded_df = pd.get_dummies(merged_df['stadium_x'], prefix='stadium_x')
    merged_df = pd.concat([merged_df, encoded_df], axis=1)
    merged_df.drop('stadium_x', axis=1, inplace=True)

    def clean_data(merged_df):
        merged_df = merged_df.astype({
            'stadium_x_광주-기아 챔피언스 필드': 'int8',
            'stadium_x_대구 삼성 라이온즈 파크': 'int8',
            'stadium_x_대전 한화생명 이글스파크': 'int8',
            'stadium_x_마산': 'int8',
            'stadium_x_부산 사직 야구장': 'int8',
            'stadium_x_서울 잠실 야구장': 'int8',
            'stadium_x_서울고척스카이돔': 'int8',
            'stadium_x_수원케이티위즈파크': 'int8',
            'stadium_x_울산문수야구장': 'int8',
            'stadium_x_청주야구장': 'int8',
            'stadium_x_인천SSG 랜더스필드': 'int8',
            'stadium_x_창원NC파크': 'int8',
            'stadium_x_포항야구장': 'int8'
        })
        return merged_df

    merged_df = clean_data(merged_df.copy())
    merged_df.drop(columns=['연', '월', '일'], inplace=True)

    spec2024 = pd.read_csv('data/2024spec.csv')
    spec2024.drop(['Unnamed: 0'], axis=1, inplace=True)

    all_df = pd.concat([merged_df, spec2024], ignore_index=True)

    spec2024_5 = pd.read_csv('data/2024_5_spec.csv')
    spec2024_5.drop(['Unnamed: 0'], axis=1, inplace=True)

    all_df = pd.concat([all_df, spec2024_5], ignore_index=True)

    all_df['date'] = pd.to_datetime(all_df['date'])

    train_data, test_data = train_test_split(all_df, test_size=0.1, random_state=42)
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)

    hyperparameters = {
        'presets': 'best_quality',
        'time_limit': 3600 * 10,
        'num_stack_levels': 4,
        'feature_prune_kwargs': {'method': 'rf_importance', 'threshold': 0.005},
        'calibrate': True,
        'verbosity': 3,
        'holdout_frac': 0.1
    }

    # 모델 학습
    predictor = TabularPredictor(label='adjusted_spec', eval_metric='mean_absolute_error').fit(train_data=train_data, **hyperparameters)
    

    # 모델 로드 및 평가
    predictor = TabularPredictor.load('models/ag-20240521_003925')
    
    best_model = predictor.get_model_best()
    print(f"The best model is: {best_model}")
    # Extract the best model
    best_model = predictor._trainer.load_model(best_model)

    y_true = test_data['adjusted_spec']
    pred = predictor.predict(test_data,model=best_model)

    # MAPE 계산
    mape = np.mean(np.abs((y_true - pred) / y_true)) * 100
    print('MAPE:', mape)

    # 시각화
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, pred, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Scatter plot of Actual vs. Predicted\nMAPE: {:.2f}%'.format(mape))
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')  # 대각선 추가
    plt.xlim(min(y_true) - 0.1, max(y_true) + 0.1)  # 가로축 범위 설정
    plt.show()

if __name__ == "__main__":
    main()
