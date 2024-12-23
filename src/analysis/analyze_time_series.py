import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jinja2
import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 상수 정의
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned/player_stats"
OUTPUT_DIR = BASE_DIR / "output/time_series"
PLAYER_REPORTS_DIR = OUTPUT_DIR / "player_reports"

# 분석할 지표 정의
METRICS = {
    'performance': ['avg', 'obp', 'slg', 'ops', 'war'],
    'power': ['home_runs', 'doubles', 'triples', 'iso'],
    'discipline': ['bb_pct', 'k_pct', 'bb_k_ratio'],
    'run_production': ['runs', 'rbis', 'rc27', 'wrc']
}

# HTML 템플릿
PLAYER_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>선수 ID: {{ player_id }} - 선수 분석 레포트</title>
    <style>
        body { font-family: 'Malgun Gothic', sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 40px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f5f5f5; }
        .chart { margin: 20px 0; }
        .highlight { background-color: #ffffcc; }
    </style>
</head>
<body>
    <div class="header">
        <h1>선수 ID: {{ player_id }} - 선수 분석 레포트</h1>
        <p>팀: {{ team }} | 생성일: {{ report_date }}</p>
    </div>

    <div class="section">
        <h2>주요 성과 요약</h2>
        {{ performance_summary | safe }}
    </div>

    <div class="section">
        <h2>월별 성과 분석</h2>
        {{ monthly_analysis | safe }}
        <div class="chart">
            {{ monthly_charts | safe }}
        </div>
    </div>

    <div class="section">
        <h2>상황별 성과 분석</h2>
        {{ situation_analysis | safe }}
        <div class="chart">
            {{ situation_charts | safe }}
        </div>
    </div>

    <div class="section">
        <h2>주요 지표 추세</h2>
        {{ metrics_trends | safe }}
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

def load_player_data() -> pd.DataFrame:
    """선수 데이터 로드"""
    all_data = []
    for file_path in CLEANED_DIR.glob("player_*.csv"):
        df = pd.read_csv(file_path)
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 데이터 전처리
    logger = logging.getLogger(__name__)
    logger.info(f"원본 데이터 형태: {combined_data.shape}")
    
    # NaN 값이 있는 행 제거
    combined_data = combined_data.dropna(subset=['player_id', 'team', 'category'])
    
    # 수치형 컬럼의 NaN을 0으로 대체
    numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
    combined_data[numeric_columns] = combined_data[numeric_columns].fillna(0)
    
    return combined_data

def analyze_monthly_trends(data: pd.DataFrame):
    """월별 추세 분석"""
    logger = logging.getLogger(__name__)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 월별 데이터 필터링
    monthly_data = data[data['category'] == 'month'].copy()
    
    # 각 지표 그룹별 분석
    for group_name, metrics in METRICS.items():
        logger.info(f"{group_name} 월별 추세 분석 중...")
        
        # 선수별 월별 추세 그래프
        fig = make_subplots(rows=len(metrics), cols=1,
                           subplot_titles=metrics,
                           vertical_spacing=0.05)
        
        for i, metric in enumerate(metrics, 1):
            for player_id in monthly_data['player_id'].unique():
                player_data = monthly_data[monthly_data['player_id'] == player_id]
                player_data = player_data.sort_values('subcategory')
                
                fig.add_trace(
                    go.Scatter(x=player_data['subcategory'],
                              y=player_data[metric],
                              name=f"Player {player_id}",
                              mode='lines+markers',
                              showlegend=(i == 1)),
                    row=i, col=1
                )
            
            fig.update_xaxes(title_text="Month", row=i, col=1)
            fig.update_yaxes(title_text=metric, row=i, col=1)
        
        fig.update_layout(height=300*len(metrics),
                         title_text=f"월별 {group_name} 추세",
                         showlegend=True)
        
        fig.write_html(OUTPUT_DIR / f'monthly_trends_{group_name}.html')
        
        # 히트맵으로 월별 평균 표현
        plt.figure(figsize=(12, 8))
        monthly_avg = monthly_data.groupby('subcategory')[metrics].mean()
        sns.heatmap(monthly_avg, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'월별 {group_name} 평균')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'monthly_heatmap_{group_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

def analyze_situational_performance(data: pd.DataFrame):
    """상황별 성과 분석"""
    logger = logging.getLogger(__name__)
    
    # 상황별 데이터 필터링
    situation_data = data[data['category'] == 'base'].copy()
    
    # 각 지표 그룹별 분석
    for group_name, metrics in METRICS.items():
        logger.info(f"{group_name} 상황별 분석 중...")
        
        # 상황별 평균 성과
        plt.figure(figsize=(15, 8))
        situation_avg = situation_data.groupby('subcategory')[metrics].mean()
        
        # 히트맵
        sns.heatmap(situation_avg, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'상황별 {group_name} 평균')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'situation_heatmap_{group_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 상황별 박스플롯
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='subcategory', y=metric, data=situation_data)
            plt.title(f'상황별 {metric} 분포')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'situation_boxplot_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()

def create_summary_report(data: pd.DataFrame):
    """분석 결과 요약 리포트 생성"""
    logger = logging.getLogger(__name__)
    
    # 월별 요약
    monthly_summary = data[data['category'] == 'month'].groupby('subcategory')[list(sum(METRICS.values(), []))].agg(['mean', 'std', 'min', 'max'])
    
    # 상황별 요약
    situation_summary = data[data['category'] == 'base'].groupby('subcategory')[list(sum(METRICS.values(), []))].agg(['mean', 'std', 'min', 'max'])
    
    # Excel 파일로 저장
    with pd.ExcelWriter(OUTPUT_DIR / 'time_series_analysis_summary.xlsx') as writer:
        monthly_summary.to_excel(writer, sheet_name='Monthly_Summary')
        situation_summary.to_excel(writer, sheet_name='Situation_Summary')
        
        # 선수별 월별 최고 성과
        for metric in sum(METRICS.values(), []):
            best_monthly = data[data['category'] == 'month'].sort_values(metric, ascending=False).groupby('player_id').first()
            best_monthly[['team', 'subcategory', metric]].to_excel(writer, sheet_name=f'Best_{metric}')

def create_player_report(data: pd.DataFrame, player_id: int):
    """선수별 상세 레포트 생성"""
    logger = logging.getLogger(__name__)
    logger.info(f"선수 {player_id}의 상세 레포트 생성 중...")

    # 선수 데이터 필터링
    player_data = data[data['player_id'] == player_id].copy()
    
    if player_data.empty:
        logger.warning(f"선수 ID {player_id}에 대한 데이터가 없습니다.")
        return
    
    # 기본 정보
    team = player_data['team'].iloc[0]
    
    # 성과 요약
    total_stats = player_data[player_data['category'] == 'total'].iloc[0]
    
    # 월별 성과
    monthly_data = player_data[player_data['category'] == 'month'].sort_values('subcategory')
    
    # 상황별 성과
    situation_data = player_data[player_data['category'] == 'base']
    
    # 차트 생성
    monthly_fig = create_monthly_charts(monthly_data)
    situation_fig = create_situation_charts(situation_data)
    
    # HTML 컨텐츠 생성
    template = jinja2.Template(PLAYER_REPORT_TEMPLATE)
    html_content = template.render(
        player_id=player_id,
        team=team,
        report_date=datetime.datetime.now().strftime("%Y-%m-%d"),
        performance_summary=create_performance_summary_table(total_stats),
        monthly_analysis=create_monthly_analysis_table(monthly_data),
        situation_analysis=create_situation_analysis_table(situation_data),
        monthly_charts=monthly_fig.to_html(full_html=False),
        situation_charts=situation_fig.to_html(full_html=False),
        metrics_trends=create_metrics_trends_table(monthly_data)
    )
    
    # 레포트 저장
    PLAYER_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = PLAYER_REPORTS_DIR / f"player_{player_id}_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"선수 {player_id}의 레포트가 생성되었습니다: {report_path}")

def create_monthly_charts(monthly_data: pd.DataFrame) -> go.Figure:
    """월별 성과 차트 생성"""
    fig = make_subplots(rows=len(METRICS), cols=1,
                       subplot_titles=list(METRICS.keys()),
                       vertical_spacing=0.05)
    
    for i, (group_name, metrics) in enumerate(METRICS.items(), 1):
        for metric in metrics:
            fig.add_trace(
                go.Scatter(x=monthly_data['subcategory'],
                          y=monthly_data[metric],
                          name=metric,
                          mode='lines+markers'),
                row=i, col=1
            )
        
        fig.update_xaxes(title_text="월", row=i, col=1)
        fig.update_yaxes(title_text=group_name, row=i, col=1)
    
    fig.update_layout(height=300*len(METRICS),
                     showlegend=True,
                     title_text="월별 성과 추이")
    
    return fig

def create_situation_charts(situation_data: pd.DataFrame) -> go.Figure:
    """상황별 성과 차트 생성"""
    fig = make_subplots(rows=len(METRICS), cols=1,
                       subplot_titles=list(METRICS.keys()),
                       vertical_spacing=0.05)
    
    for i, (group_name, metrics) in enumerate(METRICS.items(), 1):
        for metric in metrics:
            fig.add_trace(
                go.Bar(x=situation_data['subcategory'],
                      y=situation_data[metric],
                      name=metric),
                row=i, col=1
            )
        
        fig.update_xaxes(title_text="상황", row=i, col=1)
        fig.update_yaxes(title_text=group_name, row=i, col=1)
    
    fig.update_layout(height=300*len(METRICS),
                     showlegend=True,
                     title_text="상황별 ��과 분석")
    
    return fig

def create_performance_summary_table(total_stats: pd.Series) -> str:
    """성과 요약 테이블 생성"""
    summary_metrics = {
        '타율': 'avg',
        '출루율': 'obp',
        '장타율': 'slg',
        'OPS': 'ops',
        'WAR': 'war',
        '홈런': 'home_runs',
        '타점': 'rbis',
        '득점': 'runs'
    }
    
    html = '<table>'
    html += '<tr><th>지표</th><th>값</th></tr>'
    
    for label, metric in summary_metrics.items():
        value = total_stats[metric]
        html += f'<tr><td>{label}</td><td>{value:.3f}</td></tr>'
    
    html += '</table>'
    return html

def create_monthly_analysis_table(monthly_data: pd.DataFrame) -> str:
    """월별 성과 분석 테이블 생성"""
    html = '<table>'
    html += '<tr><th>월</th>'
    
    for metric in ['avg', 'obp', 'slg', 'ops', 'home_runs']:
        html += f'<th>{metric}</th>'
    html += '</tr>'
    
    for _, row in monthly_data.iterrows():
        html += f'<tr><td>{row["subcategory"]}월</td>'
        for metric in ['avg', 'obp', 'slg', 'ops', 'home_runs']:
            html += f'<td>{row[metric]:.3f}</td>'
        html += '</tr>'
    
    html += '</table>'
    return html

def create_situation_analysis_table(situation_data: pd.DataFrame) -> str:
    """상황별 성과 분석 테이블 생성"""
    html = '<table>'
    html += '<tr><th>상황</th>'
    
    for metric in ['avg', 'obp', 'slg', 'ops']:
        html += f'<th>{metric}</th>'
    html += '</tr>'
    
    for _, row in situation_data.iterrows():
        html += f'<tr><td>{row["subcategory"]}</td>'
        for metric in ['avg', 'obp', 'slg', 'ops']:
            html += f'<td>{row[metric]:.3f}</td>'
        html += '</tr>'
    
    html += '</table>'
    return html

def create_metrics_trends_table(monthly_data: pd.DataFrame) -> str:
    """주요 지표 추세 테이블 생성"""
    trends = {}
    
    for metric in ['avg', 'obp', 'slg', 'ops', 'war']:
        values = monthly_data[metric].tolist()
        if len(values) >= 2:
            trend = '상승' if values[-1] > values[0] else '하락'
            max_month = monthly_data.loc[monthly_data[metric].idxmax(), 'subcategory']
            min_month = monthly_data.loc[monthly_data[metric].idxmin(), 'subcategory']
            
            trends[metric] = {
                'trend': trend,
                'max_value': max(values),
                'max_month': max_month,
                'min_value': min(values),
                'min_month': min_month
            }
    
    html = '<table>'
    html += '<tr><th>지표</th><th>추세</th><th>최고값 (월)</th><th>최저값 (월)</th></tr>'
    
    for metric, info in trends.items():
        html += f'''
        <tr>
            <td>{metric}</td>
            <td>{info['trend']}</td>
            <td>{info['max_value']:.3f} ({info['max_month']}월)</td>
            <td>{info['min_value']:.3f} ({info['min_month']}월)</td>
        </tr>
        '''
    
    html += '</table>'
    return html

def main():
    logger = setup_logging()
    
    try:
        # 데이터 로드
        logger.info("선수 데이터 로드 중...")
        data = load_player_data()
        
        # 월별 추세 분석
        logger.info("월별 추세 분석 중...")
        analyze_monthly_trends(data)
        
        # 상황별 성과 분석
        logger.info("상황별 성과 분석 중...")
        analyze_situational_performance(data)
        
        # 요약 리포트 생성
        logger.info("요약 리포트 생성 중...")
        create_summary_report(data)
        
        # 선수별 상세 레포트 생성
        logger.info("선수별 상세 레포트 생성 중...")
        for player_id in data['player_id'].unique():
            create_player_report(data, player_id)
        
        logger.info("모든 분석 완료")
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 