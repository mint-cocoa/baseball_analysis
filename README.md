# KBO 선수 성적 분석 프로젝트

## 프로젝트 개요
이 프로젝트는 KBO 선수들의 상황별 성적을 분석하여 의미 있는 패턴을 발견하고 전략적 인사이트를 도출하는 것을 목표로 합니다.

## 주요 기능
1. **데이터 분석**
   - 투수 좌우 구분에 따른 성적 분석
   - 상황별(주자, 홈/원정 등) 성적 분석
   - 시계열 분석 (월별 성적 추이)
   - 선수 간 상관관계 분석

2. **클러스터링**
   - 상황별 선수 군집화
   - 유사 선수 그룹 분석
   - 성적 패턴 기반 선수 유형 분류

3. **시각화**
   - 3D 인터랙티브 시각화
   - 상황별 성적 대시보드
   - 시계열 트렌드 분석

## 폴더 구조
```
baseball_analysis/
├── data/                  # 데이터 파일
├── docs/                  # 문서화
├── notebooks/            # 분석 노트북
├── src/
│   ├── preprocessing/    # 데이터 전처리
│   ├── analysis/        # 데이터 분석
│   ├── visualization/   # 시각화
│   └── models/         # 분석 모델
└── tests/               # 테스트 코드
```

## 설치 방법
```bash
# 필요한 패키지 설치
pip install -r requirements.txt
```

## 실행 방법
```bash
# 좌우투수 분석
python -m baseball_analysis.src.analysis.clustering.handedness_analysis

# 상황별 클러스터링
python -m baseball_analysis.src.analysis.clustering.player_clustering
```

## 데이터 소스
- KBO 기록실 데이터
- 선수별 상황별 성적 데이터

## 주요 분석 지표
- WAR (승리 기여도)
- OPS (출루율+장타율)
- WOBA (가중 출루율)
- RC27 (27아웃당 득점 생산력)
- 기타 세부 지표들

## 라이선스
이 프로젝트는 MIT 라이선스 하에 공개되어 있습니다.
