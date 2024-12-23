import os
import numpy as np
import pandas as pd

def create_player_csv(base_dir, player_id):
    """특정 선수의 모든 시즌 데이터를 통합하여 CSV 파일로 생성"""
    
    player_records = []
    player_dir = os.path.join(base_dir, player_id)
    
    if not os.path.exists(player_dir):
        print(f"Warning: {player_id}의 데이터가 없습니다.")
        return None
    
    # 시즌 목록 가져오기
    seasons = [d for d in os.listdir(player_dir) 
              if os.path.isdir(os.path.join(player_dir, d))]
    seasons.sort()
    
    for season in seasons:
        player_season_dir = os.path.join(player_dir, season)
        
        # 선수의 팀 정보 가져오기
        total_stats_path = os.path.join(player_season_dir, "total", "stats.csv")
        if not os.path.exists(total_stats_path):
            print(f"Warning: {player_id}의 {season} 시즌 total 데이터가 없습니다.")
            continue
            
        team = pd.read_csv(total_stats_path)['team'].iloc[0]
        
        # 분석 유형별 데이터 수집
        for category in os.listdir(player_season_dir):
            category_path = os.path.join(player_season_dir, category)
            if not os.path.isdir(category_path):
                continue
                
            # 세부 카테고리 확인
            subcategories = [d for d in os.listdir(category_path) 
                           if os.path.isdir(os.path.join(category_path, d))]
            
            if not subcategories:  # 세부 카테고리가 없는 경우 (예: total)
                stats_file = os.path.join(category_path, "stats.csv")
                if os.path.exists(stats_file):
                    df = pd.read_csv(stats_file)
                    if not df.empty:
                        record = df.iloc[0].to_dict()
                        record.update({
                            'player_id': player_id,
                            'season': season,
                            'team': team,
                            'category': category,
                            'subcategory': None
                        })
                        player_records.append(record)
            
            else:  # 세부 카테고리가 있는 경우
                for subcategory in subcategories:
                    stats_file = os.path.join(category_path, subcategory, "stats.csv")
                    if os.path.exists(stats_file):
                        df = pd.read_csv(stats_file)
                        if not df.empty:
                            record = df.iloc[0].to_dict()
                            record.update({
                                'player_id': player_id,
                                'season': season,
                                'team': team,
                                'category': category,
                                'subcategory': subcategory
                            })
                            player_records.append(record)
    
    if not player_records:
        print(f"Warning: {player_id}의 데이터가 없습니다.")
        return None
    
    # DataFrame 생성
    df = pd.DataFrame(player_records)
    
    # 컬럼 순서 정리
    first_columns = ['player_id', 'season', 'team', 'category', 'subcategory']
    other_columns = [col for col in df.columns if col not in first_columns]
    df = df[first_columns + other_columns]
    
    # 시즌 순으로 정렬
    df = df.sort_values(['season', 'category', 'subcategory'])
    
    # CSV 파일로 저장
    output_dir = os.path.join(base_dir, "player_stats")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"player_{player_id}.csv")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"{player_id}의 데이터가 {output_file}에 저장되었습니다.")
    
    return df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "processed")
    
    # 선수 목록 가져오기
    players = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d)) and d != "player_stats"]
    players.sort()
    
    print("선수별 통합 데이터 생성 중...")
    for player_id in players:
        df = create_player_csv(base_dir, player_id)
        if df is not None:
            seasons = df['season'].unique()
            print(f"{player_id}: {len(df)} 개의 레코드 생성 (시즌: {', '.join(sorted(seasons))})")
    
    print("모든 선수의 데이터 처리가 완료되었습니다.")

if __name__ == "__main__":
    main() 