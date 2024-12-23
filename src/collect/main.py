import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

def get_player_ids():
    print("선수 ID를 입력하세요 (여러 명인 경우 쉼표로 구분):")
    ids_input = input().strip()
    return [int(id.strip()) for id in ids_input.split(',')]

def get_split_settings():
    print("\n분석 기준을 선택하세요:")
    print("0. 전체")
    print("1. 월별 (month)")
    print("2. 날짜별 (day)")
    print("3. 요일별 (week)")
    print("4. 이닝별 (inning)")
    print("5. 아웃카운트별 (outcount)")
    print("6. 볼카운트별 (ballcount)")
    print("7. 타격 위치별 (hitside)")
    print("8. 홈/원정별 (homeaway)")
    print("9. 구장별 (stadium)")
    print("10. 상대팀별 (opposite)")
    print("11. 주자 상황별 (base)")
    print("12. 승패별 (winlose)")
    print("13. 투수 유형별 (pitchside)")
    
    choice = input("선택 (0-13): ").strip()
    
    if choice == "0":
        return "", "", ""
    
    split_options = {
        "1": ("month", "시작 월(1-12)", "종료 월(1-12)"),
        "2": ("day", "시작 날짜(YYYY-MM-DD)", "종료 날짜(YYYY-MM-DD)"),
        "3": ("week", "요일(0:월 ~ 6:일)", None),
        "4": ("inning", "시작 이닝(1-9+)", "종료 이닝(1-9+)"),
        "5": ("outcount", "시작 아웃카운트(0-2)", "종료 아웃카운트(0-2)"),
        "6": ("ballcount", "볼 카운트(0-7)", "스트라이크 카운트(0-4)"),
        "7": ("hitside", "타격위치(hitR/hitL)", None),
        "8": ("homeaway", "구분(home/away)", None),
        "9": ("stadium", "구장번호(1-20)", "홈/원정(H/A)"),
        "10": ("opposite", "상대팀번호(1-16)", None),
        "11": ("base", "주자상황(onBase/emptyBase/onScoring/fullBase/base1/base2/base3/base12/base13/base23)", None),
        "12": ("winlose", "승패(win/draw/lose)", None),
        "13": ("pitchside", "투수유형(pitchL/pitchR/pitchU)", None)
    }
    
    if choice not in split_options:
        return None, None, None
    
    split01, prompt1, prompt2 = split_options[choice]
    
    print(f"\n{prompt1}:")
    split02_1 = input().strip()
    
    split02_2 = ""
    if prompt2:
        print(f"{prompt2}:")
        split02_2 = input().strip()
    
    return split01, split02_1, split02_2

def get_player_data(player_id, split01, split02_1, split02_2=""):
    """선수 데이터 수집 함수"""
    try:
        url = f"http://www.kbreport.com/player/detail/{player_id}?rows=20&order=&orderType=&teamId=2&defense_no=&split01={split01}&split02_1={split02_1}"
        if split02_2:
            url += f"&split02_2={split02_2}"

        response = requests.get(url)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        # 선수 이름 추출 (로깅용)
        player_name_tag = soup.find('h4', {'class': 'table-title'})
        player_name = player_name_tag.text.strip() if player_name_tag else "Unknown"

        # 각 테이블 박스 찾기
        table_boxes = soup.find_all('div', {'class': 'ltb-table-box'})
        
        # 각 테이블별 데이터프레임 저장
        table_dfs = []
        
        for box in table_boxes:
            box_id = box.get('id', '')
            record_type = "메인기록" if box_id == "p1" else "기본기록" if box_id == "p2" else "세부기록" if box_id == "p3" else "기타"
            
            # 먼저 responsive 클래스로 시도
            table = box.find('table', {'class': 'ltb-table responsive'})
            if table is None:
                # responsive 클래스가 없으면 기본 ltb-table로 시도
                table = box.find('table', {'class': 'ltb-table'})
            if table is None:
                print(f"테이블을 찾을 수 없습니다: {box_id}")
                continue

            # 테이블 헤더 찾기
            header_row = table.find('tr')
            if header_row is None:
                print(f"헤더 행을 찾을 수 없습니다: {box_id}")
                continue
                
            headers = [th.text.strip() for th in header_row.find_all('th')]
            if not headers:
                print(f"헤더를 찾을 수 없습니다: {box_id}")
                continue

            # 데이터 추출
            rows = table.find_all('tr')[1:]  # 첫 번째 행(헤더)를 제외하고 모든 행 가져오기
            if not rows:
                print(f"데이터 행을 찾을 수 없습니다: {box_id}")
                continue
                
            table_data = []
            
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data = [ele for ele in cols if ele]
                if data:  # 데이터가 있는 경우에만 추가
                    # 기본 정보 추가
                    row_data = {
                        '선수ID': player_id,
                        '기록유형': record_type
                    }
                    
                    # 각 컬럼의 데이터 추가 (기록유형 prefix 없이)
                    for header, value in zip(headers, data):
                        row_data[header] = value
                        
                    table_data.append(row_data)
            
            if table_data:
                # DataFrame 생성 및 리스트에 추가
                df = pd.DataFrame(table_data)
                table_dfs.append(df)
                
                # 데이터 로깅
                print(f"\n[{player_name}] {record_type} 데이터:")
                print(f"컬럼 수: {len(df.columns)}")
                print("컬럼:", list(df.columns))

        if not table_dfs:
            print(f"Warning: No valid data found for player {player_id}")
            return pd.DataFrame()

        # 모든 테이블 데이터 병합
        final_df = pd.concat(table_dfs, ignore_index=True)
        
        # DataFrame 정보 로깅
        print(f"\nDataFrame 전체 크기: {final_df.shape}")
        print("전체 컬럼:", list(final_df.columns))
        
        return final_df

    except Exception as e:
        print(f"Error processing player {player_id}: {e}")
        return pd.DataFrame()

def get_split_data(player_id):
    """각 분석 기준별로 개별 테이블을 생성하는 함수"""
    split_options = {
        "month": {
            "name": "월별",
            "ranges": [("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), 
                      ("6", "6"), ("7", "7"), ("8", "8"), ("9", "9"), ("10", "10"),
                      ("11", "11")]
        },
        "week": {
            "name": "요일별",
            "ranges": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4"), ("5", "5"), ("6", "6")]
        },
        "outcount": {
            "name": "아웃카운트별",
            "ranges": [("0", "0"), ("1", "1"), ("2", "2")]
        },
        "hitside": {
            "name": "타격위치별",
            "ranges": [("hitR", ""), ("hitL", "")]
        },
        "homeaway": {
            "name": "홈/원정별",
            "ranges": [("home", ""), ("away", "")]
        },
        "opposite": {
            "name": "상대팀별",
            "ranges": [("1", ""), ("2", ""), ("3", ""), ("4", ""), ("5", ""), ("6", ""), ("7", ""), ("8", ""), ("9", ""), ("10", "")]
        },
        "base": {
            "name": "주자상황��",
            "ranges": [("onBase", ""), ("emptyBase", ""), ("onScoring", ""), ("fullBase", ""),
                      ("base1", ""), ("base2", ""), ("base3", ""), ("base12", ""),
                      ("base13", ""), ("base23", "")]
        },
        "winlose": {
            "name": "승패별",
            "ranges": [("win", ""), ("draw", ""), ("lose", "")]
        },
        "pitchside": {
            "name": "투수유형별",
            "ranges": [("pitchL", ""), ("pitchR", ""), ("pitchU", "")]
        }
    }
    
    all_tables = {}
    
    # 전체 기록 먼저 수집
    print("\n=== 전체 기록 수집 중 ===")
    total_df = get_player_data(player_id, "", "", "")
    if not total_df.empty:
        all_tables["전체"] = total_df
    
    # 각 분석 기준별 데이터 수집
    for split01, split_info in split_options.items():
        print(f"\n=== {split_info['name']} 데이터 수집 중 ===")
        
        for split02_1, split02_2 in split_info['ranges']:
            print(f"- {split02_1}{'-'+split02_2 if split02_2 else ''} 수집 중")
            df = get_player_data(player_id, split01, split02_1, split02_2)
            
            if not df.empty:
                key = f"{split_info['name']}_{split02_1}{'-'+split02_2 if split02_2 else ''}"
                all_tables[key] = df
            
            time.sleep(random.uniform(1, 2))
    
    return all_tables

def save_split_tables(player_id, tables):
    """각 분석 기준별 테이블을 개별 파일로 저장"""
    player_name = "Unknown"
    if tables and "전체" in tables and not tables["전체"].empty:
        player_name = tables["전체"]["선수명"].iloc[0] if "선수명" in tables["전체"].columns else str(player_id)
    
    # 저장할 디렉토리 생성
    base_directory = f"player_{player_id}_{player_name}"
    os.makedirs(base_directory, exist_ok=True)
    
    # 각 테이블 저장
    for table_name, df in tables.items():
        if not df.empty:
            # 파일명에서 사용할 수 없는 문자 처리
            safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in table_name)
            filename = f"{base_directory}/{safe_name}_기록.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"{table_name} 데이터가 {filename}에 저장되었습니다.")

def main():
    # 여러 선수 ID 입력 받기
    print("선수 ID를 입력하세요 (여러 명인 경우 쉼표로 구분):")
    player_ids = input().strip().split(',')
    player_ids = [int(pid.strip()) for pid in player_ids]
    
    # 각 선수별로 데이터 수집 및 저장
    for player_id in player_ids:
        print(f"\n=== 선수 ID: {player_id} 데이터 수집 시작 ===")
        
        # 각 분석 기준별 테이블 수집
        tables = get_split_data(player_id)
        
        # 테이블 저장
        if tables:
            save_split_tables(player_id, tables)
            print(f"=== 선수 ID: {player_id} 데이터 저장 완료 ===\n")
        else:
            print(f"선수 ID: {player_id}의 수집된 데이터가 없습니다.")
        
        # 선수 간 요청 간격
        if player_id != player_ids[-1]:  # 마지막 선수가 아닌 경우에만 대기
            time.sleep(random.uniform(2, 3))

if __name__ == "__main__":
    main()