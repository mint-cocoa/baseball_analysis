import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

STAT_FIELDS = [
    'games', 'plate_appearances', 'at_bats', 'hits', 'home_runs', 'runs', 'rbis',
    'walks', 'strikeouts', 'stolen_bases', 'singles', 'doubles', 'triples',
    'intentional_walks', 'hit_by_pitch', 'sacrifice_flies', 'sacrifice_bunts',
    'gidp', 'caught_stealing', 'babip', 'avg', 'obp', 'slg', 'ops', 'woba',
    'war', 'bb_pct', 'k_pct', 'bb_k_ratio', 'iso', 'ab_hr_ratio', 'rc',
    'rc27', 'wrc', 'spd', 'wsb', 'wraa'
]

CORE_STAT_INDICES = {
    'war': STAT_FIELDS.index('war'),
    'ops': STAT_FIELDS.index('ops'),
    'woba': STAT_FIELDS.index('woba'),
    'bb_k_ratio': STAT_FIELDS.index('bb_k_ratio'),
    'rc27': STAT_FIELDS.index('rc27'),
    'spd': STAT_FIELDS.index('spd')
}

CORE_STATS = ['war', 'ops', 'woba', 'bb_k_ratio', 'rc27', 'spd']
CORE_STAT_DESCRIPTIONS = {
    'war': '승리 기여도',
    'ops': '출루율+장타율',
    'woba': '가중 출루율',
    'bb_k_ratio': '볼넷/삼진 비율',
    'rc27': '27아웃당 득점 생산력',
    'spd': '주루 능력'
}

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_data(season):
    """데이터 로드"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stat_array = np.load(os.path.join(script_dir, "processed", f"stat_array_{season}.npy"))
    dimension_info = np.load(os.path.join(script_dir, "processed", f"dimension_info_{season}.npy"), allow_pickle=True).item()
    return stat_array, dimension_info

def display_player_data(stat_array, dimension_info, player_id):
    """특정 선수의 데이터를 분석 유형 및 세부 기준별로 출력"""
    try:
        player_index = dimension_info['dimensions']['player'].index(player_id)
    except ValueError:
        print(f"오류: 선수 ID '{player_id}'를 찾을 수 없습니다.")
        return

    player_data = stat_array[player_index]
    analysis_names = dimension_info['dimensions']['analysis']
    stat_names = dimension_info['stat_fields']

    print(f"\n{player_id} 선수의 데이터:")

    for analysis_index, analysis_name in enumerate(analysis_names):
        print(f"\n## {analysis_name}:")
        detail_names = dimension_info['dimensions']['detail'][analysis_index]

        for detail_index, detail_name in enumerate(detail_names):
            if detail_name is None:
                continue

            print(f"\n### {detail_name}:")
            detail_data = player_data[analysis_index, detail_index]

            for stat_index, stat_name in enumerate(stat_names):
                print(f"- {stat_name}: {detail_data[stat_index]}")

# 핵심 성과 지표 정의



def main():
    logger = setup_logging()
    season = "2024"
    logger.info(f"{season}시즌 데이터 로딩 시작")
    
    stat_array, dimension_info = load_data(season)
    logger.info("데이터 로딩 완료")
    
    print("\n=== 차원 정보 구조 ===")
    for key, value in dimension_info.items():
        if key == 'dimensions':
            print(f"\n{key}:")
            for dim_key, dim_value in value.items():
                print(f"  {dim_key}: {len(dim_value)} 개")
                if dim_key == 'player':
                    print(f"    처음 5개 선수: {dim_value[:5]}")
                elif dim_key == 'analysis':
                    print(f"    분석 유형들: {dim_value}")
                elif dim_key == 'detail':
                    print(f"    첫 번째 분석 유형의 세부 기준들: {dim_value[0]}")
        else:
            print(f"\n{key}: {len(value)} 개의 통계 지표")
            print(f"  처음 5개 지표: {value[:5]}")

if __name__ == "__main__":
    main() 