import numpy as np
import plotly.graph_objects as go
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

def get_monthly_data(stat_array, dimension_info):
    """월별 성적 데이터 추출"""
    try:
        # 월별 분석 인덱스 찾기
        analysis_index = dimension_info['dimensions']['analysis'].index('month')
        detail_names = dimension_info['dimensions']['detail'][analysis_index]
        
        # 핵심 지표 인덱스
        core_indices = [CORE_STAT_INDICES[stat] for stat in CORE_STATS]
        
        # 월별 데이터 저장을 위한 딕셔너리
        monthly_stats = {}
        valid_months = []
        
        # 각 월별 데이터 추출
        for month_idx, month_name in enumerate(detail_names):
            if month_name is not None:
                month_data = stat_array[:, analysis_index, month_idx, :]
                monthly_stats[month_name] = month_data[:, core_indices]
                valid_months.append(month_name)
        
        return monthly_stats, valid_months
    except Exception as e:
        logger.error(f"데이터 추출 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def create_time_series_visualization(monthly_stats, months, player_ids, title):
    """시계열 시각화 생성"""
    try:
        # 저장 디렉토리 확인 및 생성
        save_dir = os.path.join(os.path.dirname(current_dir), "visualization", "results")
        os.makedirs(save_dir, exist_ok=True)
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                f'{CORE_STAT_DESCRIPTIONS[stat]} 추이' for stat in CORE_STATS
            ]
        )
        
        # 각 선수별로 다른 색상 사용
        colors = [f'hsl({h},50%,50%)' for h in np.linspace(0, 360, len(player_ids))]
        
        # 각 지표별 시계열 플롯
        for stat_idx, stat_name in enumerate(CORE_STATS):
            row = (stat_idx // 2) + 1
            col = (stat_idx % 2) + 1
            
            for player_idx, player_id in enumerate(player_ids):
                # 해당 선수의 월별 데이터 추출
                monthly_values = []
                for month in months:
                    monthly_values.append(monthly_stats[month][player_idx, stat_idx])
                
                # 시계열 플롯 추가
                fig.add_trace(
                    go.Scatter(
                        x=months,
                        y=monthly_values,
                        name=f'선수 {player_id}',
                        line=dict(color=colors[player_idx]),
                        showlegend=(stat_idx == 0)  # 첫 번째 지표에서만 범례 표시
                    ),
                    row=row, col=col
                )
        
        # 레이아웃 설정
        fig.update_layout(
            height=1000,
            title_text=title,
            showlegend=True
        )
        
        # HTML 파일로 저장
        save_path = os.path.join(save_dir, "time_series_analysis.html")
        fig.write_html(save_path)
        logger.info(f"시각화 파일 저장 완료: {save_path}")
        
        # 성적이 가장 많이 향상된 선수들 찾기
        improvement_stats = {}
        for player_idx, player_id in enumerate(player_ids):
            improvements = []
            for stat_idx, stat_name in enumerate(CORE_STATS):
                first_month_value = monthly_stats[months[0]][player_idx, stat_idx]
                last_month_value = monthly_stats[months[-1]][player_idx, stat_idx]
                improvement = last_month_value - first_month_value
                improvements.append(improvement)
            
            # 전체 향상도 계산 (정규화된 값의 평균)
            normalized_improvements = [imp / max(abs(imp), 1e-10) for imp in improvements]
            improvement_stats[player_id] = np.mean(normalized_improvements)
        
        # 상위 5명의 향상된 선수들 출력
        top_improved = sorted(improvement_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        logger.info("\n성적이 가장 많이 향상된 선수들:")
        for player_id, improvement in top_improved:
            logger.info(f"\n선수 ID: {player_id}")
            logger.info(f"전체 향상도: {improvement:.3f}")
            for stat_idx, stat_name in enumerate(CORE_STATS):
                first_value = monthly_stats[months[0]][player_ids.index(player_id), stat_idx]
                last_value = monthly_stats[months[-1]][player_ids.index(player_id), stat_idx]
                logger.info(f"{CORE_STAT_DESCRIPTIONS[stat_name]}: {first_value:.3f} → {last_value:.3f}")
        
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
        
        # 월별 성적 데이터 추출
        monthly_stats, months = get_monthly_data(stat_array, dimension_info)
        logger.info("월별 성적 데이터 추출 완료")
        
        # 시계열 시각화 생성
        title = "선수별 월간 성적 추이 분석"
        create_time_series_visualization(monthly_stats, months, player_ids, title)
        logger.info("시계열 시각화 생성 완료")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 