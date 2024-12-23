import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import sys
import logging
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리의 모듈을 import하기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

try:
    from baseball_analysis.src.preprocessing.analyze_team import load_data, CORE_STATS, CORE_STAT_INDICES, CORE_STAT_DESCRIPTIONS
except ImportError as e:
    logger.error(f"Import 오류: {e}")
    logger.error(f"현재 디렉토리: {os.getcwd()}")
    logger.error(f"Python 경로: {sys.path}")
    logger.error(traceback.format_exc())
    raise

def get_situation_data(stat_array, dimension_info, analysis_type, detail_name=None):
    """특정 상황에 대한 선���들의 데이터를 추출"""
    try:
        analysis_index = dimension_info['dimensions']['analysis'].index(analysis_type)
        
        if detail_name:
            detail_names = dimension_info['dimensions']['detail'][analysis_index]
            logger.info(f"분석 유형 '{analysis_type}'의 세부 기준들: {detail_names}")
            detail_index = detail_names.index(detail_name)
        else:
            detail_index = 0
        
        situation_data = stat_array[:, analysis_index, detail_index, :]
        core_indices = [CORE_STAT_INDICES[stat] for stat in CORE_STATS]
        data = situation_data[:, core_indices]
        
        # NaN 값을 0으로 대체
        data = np.nan_to_num(data, nan=0.0)
        
        return data
    except Exception as e:
        logger.error(f"데이터 추출 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def perform_clustering(data, n_clusters=3):
    """데이터 클러스터링 수행"""
    try:
        # 데이터 정규화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        # NaN이나 무한대 값이 있는지 확인
        if np.any(np.isnan(scaled_data)) or np.any(np.isinf(scaled_data)):
            logger.warning("정규화된 데이터에 NaN 또는 무한대 값이 있습니다.")
            scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        return clusters, scaled_data
    except Exception as e:
        logger.error(f"클러스터링 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def create_cluster_visualization(data, clusters, player_ids, title, filename):
    """클러스터링 결과 시각화"""
    try:
        # 저장 디렉토리 확인 및 생성
        save_dir = os.path.join(current_dir, "results")
        os.makedirs(save_dir, exist_ok=True)
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=[
                               f'{CORE_STAT_DESCRIPTIONS["war"]} vs {CORE_STAT_DESCRIPTIONS["ops"]}',
                               f'{CORE_STAT_DESCRIPTIONS["woba"]} vs {CORE_STAT_DESCRIPTIONS["bb_k_ratio"]}',
                               f'{CORE_STAT_DESCRIPTIONS["rc27"]} vs {CORE_STAT_DESCRIPTIONS["spd"]}',
                               '클러스터별 분포'
                           ])
        
        colors = px.colors.qualitative.Set3
        
        # WAR vs OPS
        fig.add_trace(
            go.Scatter(x=data[:, 0], y=data[:, 1],
                      mode='markers+text',
                      marker=dict(color=[colors[c] for c in clusters]),
                      text=player_ids,
                      textposition="top center",
                      name='WAR vs OPS'),
            row=1, col=1
        )
        
        # WOBA vs BB/K
        fig.add_trace(
            go.Scatter(x=data[:, 2], y=data[:, 3],
                      mode='markers+text',
                      marker=dict(color=[colors[c] for c in clusters]),
                      text=player_ids,
                      textposition="top center",
                      name='WOBA vs BB/K'),
            row=1, col=2
        )
        
        # RC27 vs SPD
        fig.add_trace(
            go.Scatter(x=data[:, 4], y=data[:, 5],
                      mode='markers+text',
                      marker=dict(color=[colors[c] for c in clusters]),
                      text=player_ids,
                      textposition="top center",
                      name='RC27 vs SPD'),
            row=2, col=1
        )
        
        # 클러스터별 선수 수 분포
        cluster_counts = np.bincount(clusters)
        fig.add_trace(
            go.Bar(x=[f'클러스터 {i}' for i in range(len(cluster_counts))],
                   y=cluster_counts,
                   marker_color=colors[:len(cluster_counts)],
                   name='클러스터 분포'),
            row=2, col=2
        )
        
        # 축 레이블 추가
        fig.update_xaxes(title_text=CORE_STAT_DESCRIPTIONS["war"], row=1, col=1)
        fig.update_yaxes(title_text=CORE_STAT_DESCRIPTIONS["ops"], row=1, col=1)
        
        fig.update_xaxes(title_text=CORE_STAT_DESCRIPTIONS["woba"], row=1, col=2)
        fig.update_yaxes(title_text=CORE_STAT_DESCRIPTIONS["bb_k_ratio"], row=1, col=2)
        
        fig.update_xaxes(title_text=CORE_STAT_DESCRIPTIONS["rc27"], row=2, col=1)
        fig.update_yaxes(title_text=CORE_STAT_DESCRIPTIONS["spd"], row=2, col=1)
        
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=False
        )
        
        # HTML 파일로 저장
        save_path = os.path.join(save_dir, f"{filename}.html")
        fig.write_html(save_path)
        logger.info(f"시각화 파일 저장 완료: {save_path}")
        
    except Exception as e:
        logger.error(f"시각화 생성 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        logger.info("데이터 로딩 시작")
        stat_array, dimension_info = load_data("2024")
        player_ids = dimension_info['dimensions']['player']
        logger.info("데이터 로딩 완료")
        
        logger.info("차원 정보:")
        for key, value in dimension_info['dimensions'].items():
            logger.info(f"{key}: {value}")
        
        situations = [
            ('pitchside', 'left'),  # 좌투수
            ('pitchside', 'right'),  # 우투수
            ('base', 'empty_base'),  # 주자없음
            ('base', 'scoring_position'),  # 득점권
            ('homeaway', 'home'),  # 홈
            ('homeaway', 'away')  # 원정
        ]
        
        for analysis_type, detail_name in situations:
            logger.info(f"\n{analysis_type} - {detail_name} 분석 시작")
            
            situation_data = get_situation_data(stat_array, dimension_info, analysis_type, detail_name)
            logger.info("상황별 데이터 추출 완료")
            
            clusters, scaled_data = perform_clustering(situation_data)
            logger.info("클러스터링 완료")
            
            title = f"선수 클러스터링 분석: {detail_name} 상황"
            filename = f"clustering_{analysis_type}_{detail_name}"
            create_cluster_visualization(scaled_data, clusters, player_ids, title, filename)
            logger.info(f"{filename} 분석 완료")
            
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 