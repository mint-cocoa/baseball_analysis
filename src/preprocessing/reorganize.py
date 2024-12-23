import os
import shutil
import pandas as pd
import re

# 분석 기준 매핑 테이블
ANALYSIS_TYPES = {
    "전체": "total",
    "월별": "month",
    "요일별": "weekday",
    "이닝별": "inning",
    "아웃카운트별": "outcount",
    "볼카운트별": "ballcount",
    "타격위치별": "hitside",
    "홈_원정별": "homeaway",
    "구장별": "stadium",
    "상대팀별": "opponent",
    "주자상황": "base",
    "주자상황별": "base",
    "승패별": "winlose",
    "투수유형별": "pitchside"
}

# 세부 지표 매핑 테이블
DETAIL_TYPES = {
    "month": {str(i): f"{i:02d}" for i in range(1, 13)},  # 01-12
    "weekday": {
        "0": "mon", "1": "tue", "2": "wed",
        "3": "thu", "4": "fri", "5": "sat", "6": "sun"
    },
    "outcount": {
        "0": "zero", "1": "one", "2": "two"
    },
    "hitside": {
        "hitR": "right", "hitL": "left"
    },
    "homeaway": {
        "home": "home", "away": "away"
    },
    "opponent": {str(i): f"team{i:02d}" for i in range(1, 11)},  # team01-team10
    "base": {
        "onBase": "on_base",
        "emptyBase": "empty_base",
        "onScoring": "scoring_position",
        "fullBase": "full_base",
        "base1": "first",
        "base2": "second",
        "base3": "third",
        "base12": "first_second",
        "base13": "first_third",
        "base23": "second_third"
    },
    "winlose": {
        "win": "win", "draw": "draw", "lose": "lose"
    },
    "pitchside": {
        "pitchL": "left", "pitchR": "right", "pitchU": "unknown"
    }
}

# 컬럼명 매핑 테이블
COLUMN_TYPES = {
    "선수ID": "player_id",
    "기록유형": "record_type",
    "시즌": "season",
    "팀명": "team",
    "경기": "games",
    "타석": "plate_appearances",
    "타수": "at_bats",
    "안타": "hits",
    "홈런": "home_runs",
    "득점": "runs",
    "타점": "rbis",
    "볼넷": "walks",
    "삼진": "strikeouts",
    "도루": "stolen_bases",
    "BABIP": "babip",
    "타율": "avg",
    "출루율": "obp",
    "장타율": "slg",
    "OPS": "ops",
    "wOBA": "woba",
    "WAR": "war",
    "단타": "singles",
    "2루타": "doubles",
    "3��타": "triples",
    "고4": "intentional_walks",
    "HBP": "hit_by_pitch",
    "희플": "sacrifice_flies",
    "희타": "sacrifice_bunts",
    "병살": "gidp",
    "도실": "caught_stealing",
    "볼넷%": "bb_pct",
    "삼진%": "k_pct",
    "볼/삼": "bb_k_ratio",
    "ISO": "iso",
    "타수/홈런": "ab_hr_ratio",
    "RC": "rc",
    "RC/27": "rc27",
    "wRC": "wrc",
    "SPD": "spd",
    "wSB": "wsb",
    "wRAA": "wraa"
}

def get_seasons_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if '시즌' in df.columns:
            return df['시즌'].unique().astype(str).tolist()
    except:
        pass
    return []

def get_analysis_type(file_name):
    # 정규식 패턴 - 파일명에서 분석 기준과 매개변수를 추출
    pattern = r'^([^_]+(?:_원정)?별?)(?:_{1,3}([^_]+))?(?:_([^_]+))?_기록\.csv$'
    match = re.match(pattern, file_name)
    
    if match:
        base_type = match.group(1)  # 기본 분석 기준
        param1 = match.group(2)     # 첫 번째 매개변수
        param2 = match.group(3)     # 두 번째 매개변수
        
        # 기본 분석 기준을 영어로 변환
        if base_type not in ANALYSIS_TYPES:
            return None
        eng_base_type = ANALYSIS_TYPES[base_type]
        
        # 세부 지표 변환
        if param1 and param2:
            # 두 개의 매개변수가 있는 경우 (예: 월별_3-3)
            if eng_base_type in DETAIL_TYPES:
                if param1 in DETAIL_TYPES[eng_base_type]:
                    param1 = DETAIL_TYPES[eng_base_type][param1]
                if param2 in DETAIL_TYPES[eng_base_type]:
                    param2 = DETAIL_TYPES[eng_base_type][param2]
                return f"{eng_base_type}/{param1}-{param2}"
            return f"{eng_base_type}/{param1}-{param2}"  # 매핑이 없는 경우 원래 값 사용
        elif param1:
            # 하나의 매개변수만 있는 경우 (예: 홈_원정별_home)
            if eng_base_type in DETAIL_TYPES and param1 in DETAIL_TYPES[eng_base_type]:
                param1 = DETAIL_TYPES[eng_base_type][param1]
            return f"{eng_base_type}/{param1}"
        else:
            # 매개변수가 없는 경우 (예: 전체)
            return eng_base_type
    
    return None

def merge_record_types(df):
    # 기록 유형별로 데이터프레임 분리
    main_record = df[df['record_type'] == '메인기록'].iloc[0] if not df[df['record_type'] == '메인기록'].empty else pd.Series()
    basic_record = df[df['record_type'] == '기본기록'].iloc[0] if not df[df['record_type'] == '기본기록'].empty else pd.Series()
    detail_record = df[df['record_type'] == '세부기록'].iloc[0] if not df[df['record_type'] == '세부기록'].empty else pd.Series()
    
    # 새로운 통합 레코드 생성
    merged_record = pd.Series()
    
    # 공통 필드는 메인 기록에서 가져오기
    common_fields = ['player_id', 'season', 'team', 'games', 'plate_appearances', 'at_bats', 
                    'hits', 'home_runs', 'runs', 'rbis', 'walks', 'strikeouts', 'stolen_bases']
    for field in common_fields:
        if field in main_record:
            merged_record[field] = main_record[field]
    
    # 기본 기록에서만 있는 필드 추가
    basic_only_fields = ['singles', 'doubles', 'triples', 'intentional_walks', 'hit_by_pitch',
                        'sacrifice_flies', 'sacrifice_bunts', 'gidp', 'caught_stealing']
    for field in basic_only_fields:
        if field in basic_record:
            merged_record[field] = basic_record[field]
    
    # 세부 기록에서만 있는 필드 추가
    detail_only_fields = ['babip', 'avg', 'obp', 'slg', 'ops', 'woba', 'war',
                         'bb_pct', 'k_pct', 'bb_k_ratio', 'iso', 'ab_hr_ratio',
                         'rc', 'rc27', 'wrc', 'spd', 'wsb', 'wraa']
    for field in detail_only_fields:
        if field in detail_record:
            merged_record[field] = detail_record[field]
    
    return pd.DataFrame([merged_record])

def process_csv_file(src_path, dst_path, target_season):
    # CSV 파일 읽기
    df = pd.read_csv(src_path)
    
    # 컬럼명 변경
    renamed_columns = {}
    for col in df.columns:
        if col in COLUMN_TYPES:
            renamed_columns[col] = COLUMN_TYPES[col]
    
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
    
    # 해당 시즌의 데이터만 필터링
    if 'season' in df.columns:
        df = df[df['season'] == int(target_season)]
    
    # 기록 유형 통합
    if not df.empty and 'record_type' in df.columns:
        df = merge_record_types(df)
    
    # 변경된 CSV 파일 저장
    df.to_csv(dst_path, index=False)

def reorganize_data():
    base_dir = 'data'
    new_base_dir = 'data_reorganized'
    
    # 새로운 기본 디렉토리 생성
    if os.path.exists(new_base_dir):
        shutil.rmtree(new_base_dir)
    os.makedirs(new_base_dir)
    
    # 각 선수 폴더 처리
    for player_dir in os.listdir(base_dir):
        if not player_dir.startswith('player_'):
            continue
            
        player_path = os.path.join(base_dir, player_dir)
        if not os.path.isdir(player_path):
            continue
            
        # 선수 ID 추출
        player_id = player_dir.split('_')[1]
        
        # 시즌 정보 가져오기
        total_record_path = os.path.join(player_path, '전체_기록.csv')
        if not os.path.exists(total_record_path):
            print(f"Warning: Could not find total record for {player_dir}")
            continue
            
        seasons = get_seasons_from_csv(total_record_path)
        if not seasons:
            print(f"Warning: Could not determine seasons for {player_dir}")
            continue
        
        # 각 시즌별로 처리
        for season in seasons:
            # 각 분석 기준별 파일 처리
            for file_name in os.listdir(player_path):
                if not file_name.endswith('_기록.csv'):
                    continue
                    
                # 분석 기준 추출
                analysis_path = get_analysis_type(file_name)
                if not analysis_path:
                    print(f"Warning: Could not parse analysis type from {file_name}")
                    continue
                
                # 새로운 경로 생성
                new_dir = os.path.join(new_base_dir, f"{player_id}", season, analysis_path)
                os.makedirs(new_dir, exist_ok=True)
                
                # 파일 복사 및 이름 변경, CSV 처리
                src_path = os.path.join(player_path, file_name)
                new_file_name = f"stats.csv"  # 모든 파일을 stats.csv로 통일
                dst_path = os.path.join(new_dir, new_file_name)
                process_csv_file(src_path, dst_path, season)
            
            print(f"Processed player {player_id} for season {season}")

if __name__ == "__main__":
    reorganize_data() 