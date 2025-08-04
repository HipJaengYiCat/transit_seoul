#!/bin/bash
#SBATCH --job-name=transit_seoul                        # 작업 이름
#SBATCH --output=./output/%j.out
#SBATCH --error=./output/%j.err
#SBATCH --partition=gpu7(수정필요)                       # 사용할 파티션 이름 (팀별 노드에 대응되는 파티션 번호로 변경 필요, gpu7 -> gpu3 or gpu5)
#SBATCH --nodelist=n107(수정필요)                        # 사용할 노드 이름 (팀별 노드 번호로 변경 필요, n107 -> 팀별 전용 노드)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1                               # 하나의 태스크가 사용할 CPU 코어 수
##SBATCH --mem=128G                                     # 메모리 할당량 (##이므로 해당 명령어 비활성화)
##SBATCH --time=48:00:00                                # 최대 실행 시간 (##이므로 해당 명령어 비활성화)

echo "start at: $(date)"                                # 접속한 날짜 표기
echo "node: $HOSTNAME"                                  # 접속한 노드 번호 표기
echo "jobid: $SLURM_JOB_ID"                             # jobid 표기

# Load modules (cuda 환경)
module load cuda/11.8.0 

# Load env (python 환경)
source ~/miniconda3/etc/profile.d/conda.sh     

# 가상환경 활성화 (설치한 가상환경 이름으로 변경 필요, ubai -> 가상환경 이름)
conda activate ubai(수정필요)                            # ubai라는 conda 환경에서, 슈퍼컴퓨팅 쓸 준비 완료

# python 스크립트 실행
python test.py
