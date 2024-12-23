import os
import numpy as np
import pandas as pd

# 분석 기준 매핑 테이블 (역매핑용)
ANALYSIS_TYPES = {
    "month": "월별",
    "weekday": "요일별",
    "outcount": "아웃카운트별",
    "hitside": "타격위치별",
    "homeaway": "홈원정별",
    "opponent": "상대팀별",
    "base": "주자상황",
    "winlose": "승패별",
    "pitchside": "투수유형별"
}

# 각 분석 기준별 차원 정보
DIMENSION_INFO = {
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "weekday": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    "outcount": ["zero", "one", "two"],
    "hitside": ["left", "right"],
    "homeaway": ["home", "away"],
    "opponent": [f"team{i:02d}" for i in range(1, 11)],
    "base": ["on_base", "empty_base", "scoring_position", "full_base", 
             "first", "second", "third", "first_second", "first_third", "second_third"],
    "winlose": ["win", "draw", "lose"],
    "pitchside": ["left", "right", "unknown"]
}

# 통계 필드 정의
STAT_FIELDS = [
    'games', 'plate_appearances', 'at_bats', 'hits', 'home_runs', 'runs', 'rbis',
    'walks', 'strikeouts', 'stolen_bases', 'singles', 'doubles', 'triples',
    'intentional_walks', 'hit_by_pitch', 'sacrifice_flies', 'sacrifice_bunts',
    'gidp', 'caught_stealing', 'babip', 'avg', 'obp', 'slg', 'ops', 'woba',
    'war', 'bb_pct', 'k_pct', 'bb_k_ratio', 'iso', 'ab_hr_ratio', 'rc',
    'rc27', 'wrc', 'spd', 'wsb', 'wraa'
]

def safe_float_convert(value):
    """안전하게 문자열을 float로 변환"""
    if pd.isna(value) or value == '-' or value == '':
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def create_stat_array(base_dir, season):
    """주어진 시즌의 모든 선수 데이터를 다차원 배열로 변환"""

    # 선수 ID 목록 수집
    players = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    players.sort()

    # 분석 유형 및 세부 기준 수집
    analysis_types = []
    detail_types = {}
    max_detail_length = 0

    for player_id in players:
        player_season_dir = os.path.join(base_dir, player_id, season)
        if not os.path.exists(player_season_dir):
            continue

        for analysis_dir_name in os.listdir(player_season_dir):
            analysis_path = os.path.join(player_season_dir, analysis_dir_name)
            if os.path.isdir(analysis_path):
                if analysis_dir_name not in analysis_types:
                    analysis_types.append(analysis_dir_name)

                detail_dirs = [d for d in os.listdir(analysis_path) if os.path.isdir(os.path.join(analysis_path, d))]
                if detail_dirs:
                    if analysis_dir_name not in detail_types:
                        detail_types[analysis_dir_name] = []
                    for detail_dir_name in detail_dirs:
                        if detail_dir_name not in detail_types[analysis_dir_name]:
                            detail_types[analysis_dir_name].append(detail_dir_name)
                    max_detail_length = max(max_detail_length, len(detail_types[analysis_dir_name]))

    # 세부 기준 패딩
    for analysis_type in detail_types:
        detail_types[analysis_type].sort()
        while len(detail_types[analysis_type]) < max_detail_length:
            detail_types[analysis_type].append(None)

    # 배열 차원 결정
    dimensions = {
        "player": players,
        "analysis": analysis_types,
        "detail": [detail_types.get(analysis_type, [None] * max_detail_length) for analysis_type in analysis_types],
        "stat": STAT_FIELDS
    }

    # 배열 생성
    array_shape = (
        len(dimensions["player"]),
        len(dimensions["analysis"]),
        max_detail_length,
        len(dimensions["stat"])
    )
    stat_array = np.full(array_shape, np.nan)

    # 데이터 채우기
    for player_idx, player_id in enumerate(dimensions["player"]):
        player_season_dir = os.path.join(base_dir, player_id, season)
        if not os.path.exists(player_season_dir):
            continue

        for analysis_idx, analysis_type in enumerate(dimensions["analysis"]):
            analysis_dir = os.path.join(player_season_dir, analysis_type)
            if not os.path.exists(analysis_dir):
                continue

            # 세부 기준 폴더 처리
            detail_dirs = [d for d in os.listdir(analysis_dir) if os.path.isdir(os.path.join(analysis_dir, d))]

            if not detail_dirs:  # 세부 기준이 없는 경우 (예: total)
                stats_file = os.path.join(analysis_dir, "stats.csv")
                if os.path.exists(stats_file):
                    df = pd.read_csv(stats_file)
                    if not df.empty:
                        for stat_idx, stat_field in enumerate(dimensions["stat"]):
                            if stat_field in df.columns:
                                stat_array[player_idx, analysis_idx, 0, stat_idx] = safe_float_convert(df.iloc[0][stat_field])
            else:  # 세부 기준이 있는 경우
                detail_mapping = {v: i for i, v in enumerate(dimensions["detail"][analysis_idx])}
                for detail_dir in detail_dirs:
                    if detail_dir in detail_mapping:
                        detail_idx = detail_mapping[detail_dir]
                        stats_file = os.path.join(analysis_dir, detail_dir, "stats.csv")
                        if os.path.exists(stats_file):
                            df = pd.read_csv(stats_file)
                            if not df.empty:
                                for stat_idx, stat_field in enumerate(dimensions["stat"]):
                                    if stat_field in df.columns:
                                        stat_array[player_idx, analysis_idx, detail_idx, stat_idx] = safe_float_convert(df.iloc[0][stat_field])

    return stat_array, dimensions

def save_array(stat_array, dimensions, season):
    """배열과 차원 정보를 파일로 저장"""
    # 배열 저장
    np.save(f"stat_array_{season}.npy", stat_array)

    # 차원 정보 저장
    dimension_info = {
        "shape": stat_array.shape,
        "dimensions": dimensions,
        "stat_fields": STAT_FIELDS
    }
    np.save(f"dimension_info_{season}.npy", dimension_info)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "processed")

    season = "2024"  # 원하는 시즌 지정

    print(f"Creating stat array for season {season}...")
    stat_array, dimensions = create_stat_array(base_dir, season)

    print(f"Array shape: {stat_array.shape}")
    print(f"Number of players: {len(dimensions['player'])}")
    print(f"Number of analysis types: {len(dimensions['analysis'])}")
    print(f"Number of detail categories: {len(dimensions['detail'])}")
    print(f"Number of statistics: {len(dimensions['stat'])}")

    print("\nSaving array and dimension info...")
    save_array(stat_array, dimensions, season)
    print("Done!")

if __name__ == "__main__":
    main() 