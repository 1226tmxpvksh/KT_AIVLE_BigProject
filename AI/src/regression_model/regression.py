import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import warnings
import json

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

MIN_MONTHS_FOR_TRAINING = 6 # 특정 제품 모델을 학습시키기 위한 최소 데이터 기간 (월)

class RegressionModel:
    def __init__(self, data_path=None):
        self.models = {}
        self.raw_df = None
        self.monthly_subcategory_df = None
        self.product_to_subcategory_map = {}

        if data_path is None:
            base_path = Path(__file__).resolve().parents[2]
            data_path = base_path / 'data' / 'sales_data.csv'

        try:
            print(f"--- 데이터 로드 시작: {data_path} ---")
            try:
                self.raw_df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8 디코딩 실패. latin1으로 재시도합니다.")
                self.raw_df = pd.read_csv(data_path, encoding='latin1')
            
            self._initialize_data()
            print("--- 데이터 로드 및 전처리 완료 ---")
        except FileNotFoundError:
            print(f"!!! 에러: 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
        except Exception as e:
            print(f"!!! 에러: 데이터 처리 중 오류 발생: {e}")

    def _initialize_data(self):
        """데이터를 전처리하고, 월별 집계 및 맵을 생성합니다."""
        df = self.raw_df
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df.dropna(subset=['Order Date', 'Product Name', 'Sub-Category'], inplace=True)
        
        # Product Name -> Sub-Category 맵 생성
        self.product_to_subcategory_map = df[['Product Name', 'Sub-Category']].drop_duplicates().set_index('Product Name')['Sub-Category'].to_dict()

        # Sub-Category 기준으로 월별 데이터 집계
        df['Month'] = df['Order Date'].dt.to_period('M')
        self.monthly_subcategory_df = df.groupby(['Sub-Category', 'Month']).agg(
            Total_Sales=('Sales', 'sum')
        ).reset_index()
        self.monthly_subcategory_df['Month'] = self.monthly_subcategory_df['Month'].dt.to_timestamp()
        self.monthly_subcategory_df.rename(columns={'Sub-Category': 'Name'}, inplace=True)
        self.monthly_subcategory_df = self.monthly_subcategory_df.set_index(['Name', 'Month']).sort_index()

    def _create_features(self, df, target_col='Total_Sales'):
        df = df.copy()
        df['lag_1'] = df[target_col].shift(1)
        df['lag_2'] = df[target_col].shift(2)
        df['rolling_mean_3'] = df[target_col].shift(1).rolling(window=3, min_periods=1).mean()
        return df.dropna()

    def _train_and_predict(self, name, df, target_col='Total_Sales'):
        """공통 학습 및 예측 로직"""
        print(f"--- '{name}'의 '{target_col}' 모델 학습 및 예측 ---")
        
        if name not in self.models:
            featured_df = self._create_features(df, target_col)
            if len(featured_df) < 3:
                return {'value': None, 'type': 'error', 'reason': f"'{name}'의 학습 데이터가 부족합니다."}
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(featured_df[['lag_1', 'lag_2', 'rolling_mean_3']], featured_df[target_col])
            self.models[name] = model
        
        model = self.models[name]
        last_data = df[target_col].tail(3)
        if len(last_data) < 2:
            return {'value': None, 'type': 'error', 'reason': f"'{name}'의 예측용 최근 데이터가 부족합니다."}

        future_features = pd.DataFrame([[last_data.iloc[-1], last_data.iloc[-2], last_data.rolling(window=3, min_periods=1).mean().iloc[-1]]], columns=['lag_1', 'lag_2', 'rolling_mean_3'])
        return model.predict(future_features)[0]

    def predict(self, product_input):
        if self.raw_df is None: return json.dumps({'value': None, 'type': 'error', 'reason': "데이터가 로드되지 않았습니다."})

        product_name = product_input.split(',')[0].strip()
        
        if product_name not in self.product_to_subcategory_map:
            # Sub-Category 이름으로 직접 들어온 경우 처리
            if product_name in self.monthly_subcategory_df.index.get_level_values('Name'):
                 sub_category_name = product_name
                 print(f"'{sub_category_name}' 카테고리 모델로 직접 예측합니다.")
                 prediction = self._train_and_predict(sub_category_name, self.monthly_subcategory_df.loc[sub_category_name])
                 if isinstance(prediction, dict): return json.dumps(prediction)
                 return json.dumps({'value': f"{prediction:.2f}", 'type': 'sub_category_specific', 'name': sub_category_name})
            else:
                return json.dumps({'value': None, 'type': 'error', 'reason': f"제품 또는 카테고리 '{product_name}'을(를) 찾을 수 없습니다."})

        # 특정 제품(Product Name)에 대한 데이터 집계 및 확인
        product_df_monthly = self.raw_df[self.raw_df['Product Name'] == product_name].copy()
        product_df_monthly['Month'] = product_df_monthly['Order Date'].dt.to_period('M')
        product_df_monthly = product_df_monthly.groupby('Month').agg(Total_Sales=('Sales', 'sum')).sort_index()

        # 데이터가 충분하면 특정 제품 모델로 예측
        if len(product_df_monthly) >= MIN_MONTHS_FOR_TRAINING:
            print(f"'{product_name}'의 데이터가 충분하여, 제품별 모델로 예측합니다.")
            prediction = self._train_and_predict(product_name, product_df_monthly)
            if isinstance(prediction, dict): return json.dumps(prediction)
            return json.dumps({'value': f"{prediction:.2f}", 'type': 'product_specific', 'name': product_name})
            
        # 데이터가 부족하면 Sub-Category 모델로 예측 (Fallback)
        else:
            sub_category_name = self.product_to_subcategory_map[product_name]
            print(f"'{product_name}'의 데이터가 부족({len(product_df_monthly)}개월), 상위 카테고리 '{sub_category_name}' 모델로 예측합니다.")
            
            prediction = self._train_and_predict(sub_category_name, self.monthly_subcategory_df.loc[sub_category_name])
            if isinstance(prediction, dict): return json.dumps(prediction)

            reason = f"'{product_name}'의 데이터가 부족({len(product_df_monthly)}개월)하여 상위 카테고리 '{sub_category_name}'의 예측치로 대체합니다."
            return json.dumps({'value': f"{prediction:.2f}", 'type': 'sub_category_fallback', 'name': sub_category_name, 'reason': reason})

if __name__ == '__main__':
    reg_model = RegressionModel()
    if reg_model.raw_df is not None:
        print("\n--- [CASE 1] 데이터가 충분한 제품 예측 (임의의 제품) ---")
        print(reg_model.predict('Bretford CR4500 Series Slim Rectangular Table'))
        
        print("\n--- [CASE 2] 데이터가 부족한 제품 예측 (상위 카테고리 대체) ---")
        print(reg_model.predict('Bush Somerset Collection Bookcase'))

        print("\n--- [CASE 3] 상위 카테고리로 직접 예측 ---")
        print(reg_model.predict('Chairs'))