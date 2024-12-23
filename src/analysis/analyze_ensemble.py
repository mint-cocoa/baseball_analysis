import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import joblib

# 상수 정의
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned/player_stats"
OUTPUT_DIR = BASE_DIR / "output/model_analysis"
MODEL_DIR = OUTPUT_DIR / "models"

# 분석할 지표 정의
TARGET_METRICS = ['avg', 'ops', 'war']
FEATURE_GROUPS = {
    'hitting': [
        'hits', 'at_bats', 'plate_appearances'
    ],
    'power': [
        'singles', 'doubles', 'triples', 'home_runs'
    ],
    'discipline': [
        'walks', 'strikeouts', 'babip'
    ]
}

# 모델 하이퍼파라미터 설정
MODEL_PARAMS = {
    'rf': {
        'n_estimators': 500,
        'max_depth': 8,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'gb': {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'min_samples_split': 4,
        'random_state': 42
    },
    'xgb': {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'random_state': 42
    }
}

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_player_data() -> pd.DataFrame:
    """선수 데이터 로드 및 전처리"""
    all_data = []
    for file_path in CLEANED_DIR.glob("player_*.csv"):
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 필요한 피처만 선택
    features = sum(FEATURE_GROUPS.values(), [])
    selected_columns = ['player_id', 'team', 'category', 'subcategory'] + features + TARGET_METRICS
    
    return combined_data[selected_columns].copy()

class ModelAnalyzer:
    """각 ���델별 독립적인 성과 분석"""
    
    def __init__(self, model_type: str, target_metric: str):
        self.model_type = model_type
        self.target_metric = target_metric
        
        # 모델 초기화
        if model_type == 'rf':
            self.model = RandomForestRegressor(**MODEL_PARAMS['rf'])
        elif model_type == 'gb':
            self.model = GradientBoostingRegressor(**MODEL_PARAMS['gb'])
        elif model_type == 'xgb':
            self.model = xgb.XGBRegressor(**MODEL_PARAMS['xgb'])
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """데이터 전처리 및 분할"""
        # 전체 시즌 데이터만 사용
        train_data = data[data['category'] == 'total'].copy()
        
        # 최소 타석 수 필터링 (20타석 이상)
        train_data = train_data[train_data['plate_appearances'] >= 20]
        
        # 피처 선택
        features = sum(FEATURE_GROUPS.values(), [])
        X = train_data[features]
        y = train_data[self.target_metric]
        
        # 파생 피처 생성
        X = self.create_derived_features(X)
        
        # 데이터 분할
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """파생 피처 생성"""
        X = X.copy()
        
        if self.target_metric == 'avg':
            X['hit_rate'] = X['hits'] / X['at_bats']
            X['contact_rate'] = 1 - (X['strikeouts'] / X['plate_appearances'])
        elif self.target_metric == 'ops':
            X['hit_rate'] = X['hits'] / X['at_bats']
            X['power_rate'] = (X['doubles'] + 2*X['triples'] + 3*X['home_runs']) / X['at_bats']
        
        return X
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """모델 학습"""
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 모델 학습
        self.model.fit(X_train_scaled, y_train)
        
        # 피처 중요도 저장
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        # 파생 피처 생성
        X = self.create_derived_features(X)
        
        # 스케일링 및 예측
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """모델 평가"""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        return {
            'mse': mean_squared_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
    
    def save_model(self, model_dir: Path):
        """모델 저장"""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.model, model_dir / f'{self.model_type}_{self.target_metric}_model.joblib')
        
        # 스케일러 저장
        joblib.dump(self.scaler, model_dir / f'{self.model_type}_{self.target_metric}_scaler.joblib')
        
        # 피처 중요도 저장
        if self.feature_importance:
            importance_df = pd.DataFrame([self.feature_importance])
            importance_df.to_csv(model_dir / f'{self.model_type}_{self.target_metric}_importance.csv')

def create_feature_importance_plot(analyzer: ModelAnalyzer) -> go.Figure:
    """피처 중요도 시각화"""
    if not analyzer.feature_importance:
        return None
    
    sorted_importance = dict(sorted(analyzer.feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(sorted_importance.keys()),
               y=list(sorted_importance.values()))
    )
    
    fig.update_layout(
        title=f"{analyzer.model_type.upper()} - Feature Importance for {analyzer.target_metric}",
        xaxis_title="Features",
        yaxis_title="Importance",
        height=600
    )
    
    return fig

def analyze_player_performance(data: pd.DataFrame, player_id: int, analyzers: dict) -> dict:
    """선수별 성과 분석"""
    player_data = data[data['player_id'] == player_id].copy()
    
    if player_data.empty:
        return None
    
    analysis = {
        'player_id': player_id,
        'team': player_data['team'].iloc[0],
        'model_predictions': {},
        'actual': {}
    }
    
    # 전체 시즌 데이터
    total_stats = player_data[player_data['category'] == 'total'].iloc[0]
    features = sum(FEATURE_GROUPS.values(), [])
    X = total_stats[features].to_frame().T
    
    # 각 모델별 예측
    for metric in TARGET_METRICS:
        analysis['actual'][metric] = total_stats[metric]
        analysis['model_predictions'][metric] = {}
        
        for model_type, analyzer in analyzers[metric].items():
            prediction = analyzer.predict(X)[0]
            analysis['model_predictions'][metric][model_type] = prediction
    
    return analysis

def create_performance_comparison_plot(analysis: dict) -> go.Figure:
    """성과 비교 시각화"""
    fig = go.Figure()
    
    metrics = list(analysis['model_predictions'].keys())
    actual_values = [analysis['actual'][m] for m in metrics]
    
    # 실제 값 추가
    fig.add_trace(go.Bar(
        name='실제 값',
        x=metrics,
        y=actual_values
    ))
    
    # 각 모델별 예측값 추가
    model_types = list(analysis['model_predictions'][metrics[0]].keys())
    for model_type in model_types:
        predicted_values = [analysis['model_predictions'][m][model_type] for m in metrics]
        fig.add_trace(go.Bar(
            name=f'{model_type.upper()} 예측',
            x=metrics,
            y=predicted_values
        ))
    
    fig.update_layout(
        title=f"선수 {analysis['player_id']} 성과 비교",
        barmode='group',
        height=600
    )
    
    return fig

def main():
    logger = setup_logging()
    
    try:
        # 데이터 로드
        logger.info("선수 데이터 로드 중...")
        data = load_player_data()
        
        # 각 지표별, ���델별 분석
        analyzers = {}
        model_types = ['rf', 'gb', 'xgb']
        
        for metric in TARGET_METRICS:
            logger.info(f"{metric} 분석 중...")
            analyzers[metric] = {}
            
            for model_type in model_types:
                logger.info(f"- {model_type.upper()} 모델 학습 중...")
                analyzer = ModelAnalyzer(model_type, metric)
                X_train, X_test, y_train, y_test = analyzer.prepare_data(data)
                
                # 모델 학습
                analyzer.train(X_train, y_train)
                
                # 모델 평가
                scores = analyzer.evaluate(X_test, y_test)
                logger.info(f"- {model_type.upper()} 성능: MSE={scores['mse']:.4f}, R2={scores['r2']:.4f}")
                
                # 피처 중요도 시각화
                fig = create_feature_importance_plot(analyzer)
                if fig:
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    fig.write_html(OUTPUT_DIR / f'{model_type}_{metric}_importance.html')
                
                # 모델 저장
                analyzer.save_model(MODEL_DIR)
                
                analyzers[metric][model_type] = analyzer
        
        # 선수별 성과 분석
        logger.info("선수별 성과 분석 중...")
        for player_id in data['player_id'].unique():
            analysis = analyze_player_performance(data, player_id, analyzers)
            
            if analysis:
                # 성과 비교 시각화
                fig = create_performance_comparison_plot(analysis)
                fig.write_html(OUTPUT_DIR / f'player_{player_id}_performance.html')
        
        logger.info("모든 분석 완료")
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 