import numpy as np
import os


def main():
    season = "2024"
    
    # 첫 번째 info 파일 찾기
    for file in os.listdir():
        if file.endswith("_info.npy"):
            player_id = file.split("_")[1]
            info = load_player_info(player_id, season)
            if info is not None:
                print(f"\n첫 번째 발견된 info 파일: player_{player_id}_{season}_info.npy")
                print_info_structure(info)
                break
    else:
        print("info 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main() 