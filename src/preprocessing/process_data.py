import os
import pandas as pd
import re
from pathlib import Path

# 기본 경로 설정
base_dir = 'combined_record'
output_dir = 'processed_data'

# 파일명 패턴 정의
pattern = r'player_(\d+)_\d+_([^_]+(?:_[^_]+)?(?:_{3})?)[_]([^_]+)_통합\.csv'

# 모든 파일 목록 가져오기
files = os.listdir(base_dir)

# 파일들을 분류별로 그룹화
file_groups = {}
for file in files:
    match = re.match(pattern, file)
    if match:
        player_id, base_category, subcategory = match.groups()
        
        if player_id not in file_groups:
            file_groups[player_id] = {}
        if base_category not in file_groups[player_id]:
            file_groups[player_id][base_category] = []
        file_groups[player_id][base_category].append((subcategory, file))

# 각 그룹별로 데이터 처리
for player_id in file_groups:
    for category in file_groups[player_id]:
        # 해당 카테고리의 모든 파일의 데이터를 시즌별로 수집
        season_data = {}
        
        for subcategory, filename in file_groups[player_id][category]:
            df = pd.read_csv(os.path.join(base_dir, filename))
            
            # 각 시즌별로 데이터 처리
            for _, row in df.iterrows():
                season = str(int(row['시즌']))  # 시즌을 문자열로 변환
                
                if season not in season_data:
                    season_data[season] = {}
                
                # 수치형 컬럼만 선택
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                data = row[numeric_cols].to_frame().T
                data.index = [subcategory]
                
                if subcategory not in season_data[season]:
                    season_data[season] = data
                else:
                    season_data[season] = pd.concat([season_data[season], data])
        
        # 각 시즌별로 결과 저장
        for season in season_data:
            # 출력 디렉토리 생성
            output_path = os.path.join(output_dir, player_id, season, category)
            os.makedirs(output_path, exist_ok=True)
            
            # 결과 저장
            if not season_data[season].empty:
                season_data[season].to_csv(os.path.join(output_path, 'stats.csv'))

print("데이터 처리가 완료되었습니다.") 