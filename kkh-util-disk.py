#############################################################
## 📜 문서 타입 분류 대회
## kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
## 2024.07.30.화 10:00 ~ 2024.08.11.일 19:00
## 파일 설명:
## - 서버 사용량 확인
#############################################################
 
import psutil
import time
import os
import subprocess

def get_disk_usage():
    disk_usage = psutil.disk_usage('/')
    total_disk = disk_usage.total / (1024 ** 3)  # GB로 변환
    free_disk = disk_usage.free / (1024 ** 3)  # GB로 변환
    used_disk_percent = disk_usage.percent  # 사용량 퍼센트
    free_disk_percent = (disk_usage.free / disk_usage.total) * 100  # 남은 용량 퍼센트

    return total_disk, free_disk, used_disk_percent, free_disk_percent

def get_memory_usage():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 3)  # GB로 변환
    free_memory = memory_info.available / (1024 ** 3)  # GB로 변환
    used_memory_percent = memory_info.percent  # 사용량 퍼센트
    free_memory_percent = (memory_info.available / memory_info.total) * 100  # 남은 용량 퍼센트

    return total_memory, free_memory, used_memory_percent, free_memory_percent

def get_cpu_usage():
    # 각 코어의 CPU 사용률 측정
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
    cpu_percent = sum(cpu_per_core) / len(cpu_per_core)  # 평균 사용률 계산

    return cpu_percent

def get_gpu_usage():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        gpu_percent = float(result.strip().split('\n')[0])
    except Exception as e:
        gpu_percent = None
        print(f"GPU 정보를 가져오는 중 오류 발생: {e}")

    return gpu_percent

def print_usage():
    try:
        while True:
            total_disk, free_disk, used_disk_percent, free_disk_percent = get_disk_usage()
            total_memory, free_memory, used_memory_percent, free_memory_percent = get_memory_usage()
            cpu_percent = get_cpu_usage()
            gpu_percent = get_gpu_usage()

            os.system('clear')

            # 출력 형식을 맞추기 위한 문자열 포맷
            print(f"전체 디스크 용량: {total_disk:>8.2f} GB | 남은 디스크 용량: {free_disk:>8.2f} GB ({free_disk_percent:>6.2f}%)")
            print(f"-----------------------------------------------------------------------")
            print(f"전체 메모리 용량: {total_memory:>8.2f} GB | 남은 메모리 용량: {free_memory:>8.2f} GB ({free_memory_percent:>6.2f}%)")
            print(f"-----------------------------------------------------------------------")
            print(f"CPU 사용율: {cpu_percent:>16.2f}% | ", end='')
            if gpu_percent is not None:
                print(f"GPU 사용율: {gpu_percent:>16.2f}%", end='')
            else:
                print("GPU 사용율 정보를 가져올 수 없습니다.", end='')
            print()  # 줄바꿈

            time.sleep(1)
    except KeyboardInterrupt:
        print("프로그램을 종료합니다.")

if __name__ == "__main__":
    print_usage()
