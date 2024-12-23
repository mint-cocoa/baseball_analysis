import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import logging
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리의 모듈을 import하기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# baseball_analysis 패키지를 Python 경로에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from baseball_analysis.src.preprocessing.analyze_team import load_data, CORE_STATS, CORE_STAT_INDICES, CORE_STAT_DESCRIPTIONS

def get_correlation_data(stat_array, dimension_info):
    """전체 성적 데이터 추출"""
    try:
        # 전체 성적 분석 인덱스 찾기
        analysis_index = dimension_info['dimensions']['analysis'].index('total')
        
        # 전체 성적 데이터 추출 (첫 번째 세부 기준 사용)
        total_data = stat_array[:, analysis_index, 0, :]
        
        # 핵심 지표 선택
        core_indices = [CORE_STAT_INDICES[stat] for stat in CORE_STATS]
        core_data = total_data[:, core_indices]
        
        return core_data
    except Exception as e:
        logger.error(f"데이터 추출 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def create_correlation_visualization(core_data, title):
    """상관관계 시각화 생성"""
    try:
        # 저장 디렉토리 확인 및 생성
        save_dir = os.path.join(os.path.dirname(current_dir), "visualization", "results")
        os.makedirs(save_dir, exist_ok=True)
        
        # 상관계수 행렬 계산
        corr_matrix = np.corrcoef(core_data.T)
        
        # 히트맵 생성
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=[CORE_STAT_DESCRIPTIONS[stat] for stat in CORE_STATS],
            y=[CORE_STAT_DESCRIPTIONS[stat] for stat in CORE_STATS],
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=title,
            width=800,
            height=800
        )
        
        # HTML 파일로 저장
        save_path = os.path.join(save_dir, "correlation_analysis.html")
        fig.write_html(save_path)
        logger.info(f"시각화 파일 저장 완료: {save_path}")
        
        # 강한 상관관계 찾기
        strong_correlations = []
        for i in range(len(CORE_STATS)):
            for j in range(i + 1, len(CORE_STATS)):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.5:  # 상관계수의 절대값이 0.5보다 큰 경우
                    strong_correlations.append({
                        'stat1': CORE_STAT_DESCRIPTIONS[CORE_STATS[i]],
                        'stat2': CORE_STAT_DESCRIPTIONS[CORE_STATS[j]],
                        'correlation': corr
                    })
        
        # 강한 상관관계 출력
        logger.info("\n강한 상관관계가 있는 지표들:")
        for corr in sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True):
            logger.info(f"\n{corr['stat1']} vs {corr['stat2']}")
            logger.info(f"상관계수: {corr['correlation']:.3f}")
            if corr['correlation'] > 0:
                logger.info("양의 상관관계: 한 지표가 증가하면 다른 지표도 증가하는 경향")
            else:
                logger.info("음의 상관관계: 한 지표가 증가하면 다른 지표는 감소하는 경향")
        
    except Exception as e:
        logger.error(f"시각화 생성 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def create_scatter_matrix(core_data, player_ids, title):
    """산점도 행렬 시각화 생성"""
    try:
        # 저장 디렉토리 확인 및 생성
        save_dir = os.path.join(os.path.dirname(current_dir), "visualization", "results")
        os.makedirs(save_dir, exist_ok=True)
        
        # 데이터프레임 생성
        import pandas as pd
        df = pd.DataFrame(core_data, columns=[CORE_STAT_DESCRIPTIONS[stat] for stat in CORE_STATS])
        df['선수ID'] = player_ids
        
        # 산점도 행렬 생성
        fig = px.scatter_matrix(
            df,
            dimensions=[CORE_STAT_DESCRIPTIONS[stat] for stat in CORE_STATS],
            color='선수ID',
            title=title
        )
        
        # 레이아웃 설정
        fig.update_layout(
            width=1200,
            height=1200
        )
        
        # HTML 파일로 저장
        save_path = os.path.join(save_dir, "scatter_matrix.html")
        fig.write_html(save_path)
        logger.info(f"산점도 행렬 저장 완료: {save_path}")
        
    except Exception as e:
        logger.error(f"산점도 행렬 생성 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        logger.info("데이터 로딩 시작")
        stat_array, dimension_info = load_data("2024")
        player_ids = dimension_info['dimensions']['player']
        logger.info("데이터 로딩 완료")
        
        # 상관관계 분석을 위한 데이터 추출
        core_data = get_correlation_data(stat_array, dimension_info)
        logger.info("핵심 지표 데이터 추출 완료")
        
        # 상관관계 히트맵 생성
        title = "핵심 지표 간 상관관계 분석"
        create_correlation_visualization(core_data, title)
        logger.info("상관관계 히트맵 생성 완료")
        
        # 산점도 행렬 생성
        scatter_title = "핵심 지표 산점도 행렬"
        create_scatter_matrix(core_data, player_ids, scatter_title)
        logger.info("산점도 행렬 생성 완료")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 