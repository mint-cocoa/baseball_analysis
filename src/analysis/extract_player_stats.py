import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 상수 정의
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned/player_stats"
OUTPUT_DIR = BASE_DIR / "output/player_stats"

# 삼성 선수 목록
TARGET_PLAYERS = {
    '738': '강민호',  '1754': '김지찬', '517': '구자욱',   '2070': '김영웅',
    '1436': '이성규', '1990': '김헌곤',  '704': '이재현',  '478': '박병호',
    '2073': '류지혁', '1499': '이병헌', '2178': '안주형', '2005': '김현준',
    '86': '맥키넌',   '2121': '윤정빈', '1508': '전병우', '2262': '김재상',
    '2267': '김재혁', '1285': '김성윤', '1321': '김동진', '871': '디아즈',
    '2250': '김호진', '2143': '강한울', '1687': '양도근'
}

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_player_data(player_id: str) -> pd.DataFrame:
    """선수 데이터 로��"""
    try:
        df = pd.read_csv(CLEANED_DIR / f"player_{player_id}.csv")
        return df
    except Exception as e:
        logging.error(f"선수 ID {player_id} 데이터 로드 실패: {str(e)}")
        return pd.DataFrame()

def calculate_key_metrics(df: pd.DataFrame) -> dict:
    """주요 지표 계산"""
    total_stats = df[df['category'] == 'total'].iloc[0]
    
    # 출장률 계산 (출장 경기 / 전체 경기)
    total_games = 144  # KBO 정규시즌 경기 수
    appearance_rate = (total_stats['games'] / total_games) * 100
    
    metrics = {
        '출장경기': total_stats['games'],
        '출장률': appearance_rate,
        '타율': total_stats['avg'],
        'OPS': total_stats['ops'],
        'WAR': total_stats['war'],
        '타점': total_stats['rbis'],
        '홈런': total_stats['home_runs'],
        '도루': total_stats.get('stolen_bases', 0),
        '볼넷': total_stats['walks'],
        '삼진': total_stats['strikeouts']
    }
    
    return metrics

def create_markdown_report(player_data: list) -> str:
    """마크다운 형식의 레포트 생성"""
    report = """# 삼성 라이온즈 선수 분석 레포트

## 1. 선수별 성과 분석

### 1.1 주요 지표 요약
| 선수명 | 출장경기 | 출장률(%) | 타율 | OPS | WAR  | 타점 | 홈런 | 도루 | 볼넷 | 삼진 |
|--------|--------|-----------|-----|-----|-----|--------|--------|-------|------|------|
"""
    
    # 선수별 데이터 추가
    for player in player_data:
        report += f"| {player['name']} | {int(player['metrics']['출장경기'])} | {player['metrics']['출장률']:.1f} | {player['metrics']['타율']:.3f} | {player['metrics']['OPS']:.3f} | {player['metrics']['WAR']:.1f} | {int(player['metrics']['타점'])} | {int(player['metrics']['홈런'])} | {int(player['metrics']['도루'])} | {int(player['metrics']['볼넷'])} | {int(player['metrics']['삼진'])} |\n"
    
    # 팀 전체 통계 추가
    report += "\n### 1.2 팀 전체 통계\n"
    
    # 주요 지표별 평균 계산
    metrics_avg = {
        metric: np.mean([p['metrics'][metric] for p in player_data])
        for metric in ['타율', 'OPS', 'WAR', '출장률']
    }
    
    report += f"""
- 팀 평균 출장률: {metrics_avg['출장률']:.1f}%
- 팀 평균 타율: {metrics_avg['타율']:.3f}
- 팀 평균 OPS: {metrics_avg['OPS']:.3f}
- 팀 평균 WAR: {metrics_avg['WAR']:.1f}

### 1.3 주요 발견사항
"""
    
    # 주요 발견사항 추가
    # 타율 상위 3명
    top_avg = sorted(player_data, key=lambda x: x['metrics']['타율'], reverse=True)[:3]
    report += "\n#### 타율 상위 3명\n"
    for p in top_avg:
        report += f"- {p['name']}: {p['metrics']['타율']:.3f}\n"
    
    # WAR 상위 3명
    top_war = sorted(player_data, key=lambda x: x['metrics']['WAR'], reverse=True)[:3]
    report += "\n#### WAR 상위 3명\n"
    for p in top_war:
        report += f"- {p['name']}: {p['metrics']['WAR']:.1f}\n"
    
    # 출장률 상위 3명
    top_appearance = sorted(player_data, key=lambda x: x['metrics']['출장률'], reverse=True)[:3]
    report += "\n#### 출장률 상위 3명\n"
    for p in top_appearance:
        report += f"- {p['name']}: {p['metrics']['출장률']:.1f}%\n"
    
    return report

def main():
    logger = setup_logging()
    
    try:
        # 출력 디렉토리 생성
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 선수별 데이터 수집
        player_data = []
        for player_id, player_name in TARGET_PLAYERS.items():
            logger.info(f"{player_name} 선수 데이터 분석 중...")
            
            df = load_player_data(player_id)
            if df.empty:
                continue
            
            metrics = calculate_key_metrics(df)
            player_data.append({
                'id': player_id,
                'name': player_name,
                'metrics': metrics
            })
        
        # 마크다운 레포트 생성
        logger.info("레포트 생성 중...")
        report_md = create_markdown_report(player_data)
        
        # 레포트 저장
        report_path = OUTPUT_DIR / "samsung_player_stats_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        logger.info(f"레포트가 생성되었습니다: {report_path}")
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 