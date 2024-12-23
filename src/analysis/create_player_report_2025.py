import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
import datetime

# 상수 정의
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned/player_stats"
MODEL_DIR = BASE_DIR / "output/model_analysis/models"
REPORT_DIR = BASE_DIR / "output/player_reports_2025"

# HTML 템플릿
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>선수 ID: {{ player_id }} - 2025년 성과 예측 리포트</title>
    <style>
        body { font-family: 'Malgun Gothic', sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f5f5f5; }
        .chart { margin: 20px 0; }
        .highlight { background-color: #ffffcc; }
        .model-comparison { margin: 20px 0; }
        .prediction-summary { font-size: 1.1em; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>선수 ID: {{ player_id }} - 2025년 성과 예측 리포트</h1>
        <p>팀: {{ team }} | 생성일: {{ report_date }}</p>
    </div>

    <div class="section">
        <h2>2024년 실제 성과</h2>
        {{ actual_performance | safe }}
    </div>

    <div class="section">
        <h2>2025년 예측 성과</h2>
        {{ predicted_performance | safe }}
        <div class="prediction-summary">
            {{ prediction_summary | safe }}
        </div>
    </div>

    <div class="section">
        <h2>모델별 예측 비교</h2>
        <div class="model-comparison">
            {{ model_comparison | safe }}
        </div>
    </div>

    <div class="section">
        <h2>성과 시각화</h2>
        <div class="chart">
            {{ performance_chart | safe }}
        </div>
    </div>
</body>
</html>
"""

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_models(model_dir: Path):
    """저장된 모델 로드"""
    models = {
        'avg': {},
        'ops': {},
        'war': {}
    }
    
    for metric in models.keys():
        for model_type in ['rf', 'gb', 'xgb']:
            model_path = model_dir / f'{model_type}_{metric}_model.joblib'
            scaler_path = model_dir / f'{model_type}_{metric}_scaler.joblib'
            
            if model_path.exists() and scaler_path.exists():
                models[metric][model_type] = {
                    'model': joblib.load(model_path),
                    'scaler': joblib.load(scaler_path)
                }
    
    return models

def create_performance_summary(actual: dict, predictions: dict, show_actual: bool = True) -> str:
    """성과 요약 테이블 생성"""
    html = '<table>'
    html += '<tr><th>지표</th><th>2024 실제</th><th>2025 예측</th><th>변화율</th></tr>'
    
    for metric in ['avg', 'ops', 'war']:
        actual_value = actual.get(metric, 0)
        pred_value = predictions.get(metric, 0)
        
        if show_actual:
            # 실제 성과 표시
            html += f'''
            <tr>
                <td>{metric.upper()}</td>
                <td>{actual_value:.3f}</td>
                <td>-</td>
                <td>-</td>
            </tr>
            '''
        else:
            # 예측 성과 표시
            change = pred_value - actual_value
            change_pct = (change / actual_value * 100) if actual_value != 0 else 0
            
            html += f'''
            <tr>
                <td>{metric.upper()}</td>
                <td>{actual_value:.3f}</td>
                <td>{pred_value:.3f}</td>
                <td>{change_pct:+.1f}%</td>
            </tr>
            '''
    
    html += '</table>'
    return html

def create_model_comparison(predictions: dict) -> str:
    """모델별 예측 비교 테이블 생성"""
    html = '<table>'
    html += '<tr><th>지표</th><th>RF</th><th>GB</th><th>XGB</th><th>앙상블</th></tr>'
    
    for metric in predictions.keys():
        html += f'<tr><td>{metric.upper()}</td>'
        for model in ['rf', 'gb', 'xgb']:
            html += f'<td>{predictions[metric][model]:.3f}</td>'
        # 앙상블 예측 (평균)
        ensemble = np.mean([predictions[metric][m] for m in ['rf', 'gb', 'xgb']])
        html += f'<td>{ensemble:.3f}</td></tr>'
    
    html += '</table>'
    return html

def create_prediction_summary(actual: dict, predictions: dict) -> str:
    """예측 결과 요약 생성"""
    summary = []
    
    # 타율 분석
    avg_change = predictions['avg']['ensemble'] - actual['avg']
    if abs(avg_change) > 0.020:
        direction = "상승" if avg_change > 0 else "하락"
        summary.append(f"타율이 전년 대비 큰 폭으로 {direction}할 것으로 예측됩니다.")
    
    # OPS 분석
    ops_change = predictions['ops']['ensemble'] - actual['ops']
    if abs(ops_change) > 0.050:
        direction = "상승" if ops_change > 0 else "하락"
        summary.append(f"OPS가 전년 대비 큰 폭으로 {direction}할 것으로 예측됩니다.")
    
    # WAR 분석
    war_change = predictions['war']['ensemble'] - actual['war']
    if abs(war_change) > 1.0:
        direction = "상승" if war_change > 0 else "하락"
        summary.append(f"WAR이 전년 대비 큰 폭으로 {direction}할 것으로 예측됩니다.")
    
    if not summary:
        summary.append("전반적으로 2024년과 비슷한 수준의 성과를 보일 것으로 예측됩니다.")
    
    return "<br>".join(summary)

def create_derived_features(features: pd.DataFrame, target_metric: str) -> pd.DataFrame:
    """파생 피처 생성"""
    features = features.copy()
    
    if target_metric == 'avg':
        features['hit_rate'] = features['hits'] / features['at_bats']
        features['contact_rate'] = 1 - (features['strikeouts'] / features['plate_appearances'])
    elif target_metric == 'ops':
        features['hit_rate'] = features['hits'] / features['at_bats']
        features['power_rate'] = (features['doubles'] + 2*features['triples'] + 3*features['home_runs']) / features['at_bats']
    
    return features

def create_performance_chart(actual_data: pd.DataFrame, predictions: dict) -> str:
    """선수 성과 시각화"""
    # 시즌별 데이터 준비
    season_data = actual_data[actual_data['category'] == 'total'].copy()
    season_data = season_data.sort_values('season')
    
    # 2025년 예측값 추가
    next_season = season_data['season'].max() + 1
    pred_data = pd.DataFrame({
        'season': [next_season],
        'avg': [predictions['avg']['ensemble']],
        'ops': [predictions['ops']['ensemble']],
        'war': [predictions['war']['ensemble']]
    })
    
    # 서브플롯 생성
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=('타율 (AVG) 추이', 'OPS 추이', 'WAR 추이'),
                       vertical_spacing=0.15)
    
    metrics = ['avg', 'ops', 'war']
    colors = {'actual': 'rgb(31, 119, 180)', 'predicted': 'rgb(255, 127, 14)'}
    
    for i, metric in enumerate(metrics, 1):
        # 실제 데이터
        fig.add_trace(
            go.Scatter(x=season_data['season'], y=season_data[metric],
                      mode='lines+markers',
                      name=f'실제 {metric.upper()}',
                      line=dict(color=colors['actual']),
                      showlegend=(i==1)),
            row=i, col=1
        )
        
        # 예측 데이터
        fig.add_trace(
            go.Scatter(x=[season_data['season'].max(), next_season],
                      y=[season_data[metric].iloc[-1], pred_data[metric].iloc[0]],
                      mode='lines+markers',
                      name=f'예측 {metric.upper()}',
                      line=dict(color=colors['predicted'], dash='dash'),
                      showlegend=(i==1)),
            row=i, col=1
        )
        
        # 신뢰 구간 추가
        std_dev = season_data[metric].std()
        upper = pred_data[metric].iloc[0] + std_dev
        lower = pred_data[metric].iloc[0] - std_dev
        
        fig.add_trace(
            go.Scatter(x=[next_season, next_season],
                      y=[lower, upper],
                      mode='lines',
                      name='예측 범위',
                      line=dict(color=colors['predicted'], width=1),
                      showlegend=(i==1)),
            row=i, col=1
        )
    
    # 레이아웃 설정
    fig.update_layout(
        height=800,
        title_text="시즌별 성과 추이 및 예측",
        title_x=0.5,
        font=dict(family='Malgun Gothic'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # x축 시즌 표시 설정
    fig.update_xaxes(tickmode='linear', dtick=1)
    
    # y축 범위 및 그리드 설정
    for i, metric in enumerate(metrics, 1):
        y_range = season_data[metric].max() - season_data[metric].min()
        fig.update_yaxes(
            row=i, col=1,
            gridcolor='lightgray',
            range=[
                season_data[metric].min() - y_range * 0.1,
                season_data[metric].max() + y_range * 0.1
            ]
        )
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def create_player_report(player_data: pd.DataFrame, models: dict, player_id: int) -> bool:
    """선수별 2025년 예측 리포트 생성"""
    logger = logging.getLogger(__name__)
    
    try:
        # 2024년 실제 데이터 추출
        actual_data = player_data[player_data['player_id'] == player_id].copy()
        if actual_data.empty:
            logger.warning(f"선수 ID {player_id}의 데이터가 없습니다.")
            return False
        
        total_stats = actual_data[actual_data['category'] == 'total'].iloc[0]
        
        # 실제 성과
        actual_performance = {
            'avg': total_stats['avg'],
            'ops': total_stats['ops'],
            'war': total_stats['war']
        }
        
        # 2025년 예측
        predictions = {metric: {} for metric in ['avg', 'ops', 'war']}
        base_features = total_stats[['hits', 'at_bats', 'plate_appearances', 'singles', 'doubles', 
                                   'triples', 'home_runs', 'walks', 'strikeouts', 'babip']].to_frame().T
        
        for metric in predictions.keys():
            # 파생 피처 생성
            features = create_derived_features(base_features, metric)
            
            for model_type, model_info in models[metric].items():
                X_scaled = model_info['scaler'].transform(features)
                pred = model_info['model'].predict(X_scaled)[0]
                predictions[metric][model_type] = pred
            
            # 앙상블 예측 추가
            predictions[metric]['ensemble'] = np.mean([predictions[metric][m] for m in ['rf', 'gb', 'xgb']])
        
        # 앙상블 예측값 추출
        ensemble_predictions = {metric: predictions[metric]['ensemble'] for metric in predictions.keys()}
        
        # 성과 차트 생성
        performance_chart = create_performance_chart(actual_data, predictions)
        
        # 리포트 생성
        template = jinja2.Template(REPORT_TEMPLATE)
        html_content = template.render(
            player_id=player_id,
            team=total_stats['team'],
            report_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            actual_performance=create_performance_summary(actual_performance, {}, show_actual=True),
            predicted_performance=create_performance_summary(actual_performance, ensemble_predictions, show_actual=False),
            model_comparison=create_model_comparison(predictions),
            prediction_summary=create_prediction_summary(actual_performance, predictions),
            performance_chart=performance_chart
        )
        
        # 리포트 저장
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"player_{player_id}_2025_prediction.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"선수 ID {player_id}의 2025년 예측 리포트가 생성되었습니다: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"선수 ID {player_id} 리포트 생성 중 오류 발생: {str(e)}")
        return False

def load_player_data() -> pd.DataFrame:
    """선수 데이터 로드 및 전처리"""
    all_data = []
    for file_path in CLEANED_DIR.glob("player_*.csv"):
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 필요한 피처만 선택
    features = ['hits', 'at_bats', 'plate_appearances', 'singles', 'doubles', 
               'triples', 'home_runs', 'walks', 'strikeouts', 'babip']
    selected_columns = ['player_id', 'team', 'season', 'category', 'subcategory'] + features + ['avg', 'ops', 'war']
    
    return combined_data[selected_columns].copy()

def main():
    logger = setup_logging()
    
    try:
        # 데이터 로드
        logger.info("선수 데이터 로드 중...")
        player_data = load_player_data()
        
        # 모델 로드
        logger.info("저장된 모델 로드 중...")
        models = load_models(MODEL_DIR)
        
        # 선수별 리포트 생성
        success_count = 0
        total_count = len(player_data['player_id'].unique())
        
        for player_id in player_data['player_id'].unique():
            if create_player_report(player_data, models, player_id):
                success_count += 1
        
        logger.info(f"리포트 생성 완료: {success_count}/{total_count} 성공")
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 