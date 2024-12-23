import numpy as np
import pandas as pd
import os
import sys
import logging
import traceback
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리의 모듈을 import하기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# baseball_analysis 패키지를 Python 경로에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analyze_team import load_data, CORE_STATS, CORE_STAT_INDICES, CORE_STAT_DESCRIPTIONS

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

def create_monthly_stats_excel(monthly_stats, months, player_ids):
    """월별 성적 데이터를 엑셀 파일로 저장"""
    try:
        # 저장 디렉토리 생성
        output_dir = Path(project_root) / "output" / "stats"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터프레임 생성을 위한 데이터 준비
        data = []
        for player_idx, player_id in enumerate(player_ids):
            for month in months:
                row_data = {
                    '선수ID': player_id,
                    '월': month
                }
                # 각 지표별 데이터 추가
                for stat_idx, stat_name in enumerate(CORE_STATS):
                    stat_value = monthly_stats[month][player_idx, stat_idx]
                    row_data[CORE_STAT_DESCRIPTIONS[stat_name]] = stat_value
                data.append(row_data)
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 선수별 성장률 계산
        growth_data = []
        for player_id in player_ids:
            player_data = df[df['선수ID'] == player_id]
            if len(player_data) >= 2:
                first_month = player_data.iloc[0]
                last_month = player_data.iloc[-1]
                
                growth_row = {'선수ID': player_id}
                for stat in CORE_STATS:
                    stat_name = CORE_STAT_DESCRIPTIONS[stat]
                    if first_month[stat_name] != 0:
                        growth_rate = ((last_month[stat_name] - first_month[stat_name]) / abs(first_month[stat_name])) * 100
                    else:
                        growth_rate = 0
                    growth_row[f'{stat_name} 성장률(%)'] = growth_rate
                growth_data.append(growth_row)
        
        # 성장률 DataFrame 생성
        growth_df = pd.DataFrame(growth_data)
        
        # Excel 파일로 저장
        excel_path = output_dir / "monthly_stats.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='월별 지표', index=False)
            growth_df.to_excel(writer, sheet_name='성장률', index=False)
        
        logger.info(f"월별 성적 데이터가 저장되었습니다: {excel_path}")
        
        # 성장률 상위 5명 출력
        for stat in CORE_STATS:
            stat_name = CORE_STAT_DESCRIPTIONS[stat]
            logger.info(f"\n{stat_name} 성장률 상위 5명:")
            top_5 = growth_df.nlargest(5, f'{stat_name} 성장률(%)')
            for _, row in top_5.iterrows():
                logger.info(f"선수 ID: {row['선수ID']}, 성장률: {row[f'{stat_name} 성장률(%)']:.1f}%")
        
    except Exception as e:
        logger.error(f"엑셀 파일 생성 중 오류 발생: {e}")
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
        
        # 엑셀 파일 생성
        create_monthly_stats_excel(monthly_stats, months, player_ids)
        logger.info("엑셀 파일 생성 완료")
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 