#!/bin/bash
#SBATCH --job-name=t2t                                  # 작업 이름
#SBATCH --output=./output/t2t_%j.out
#SBATCH --error=./output/t2t_%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu5                                # 사용할 파티션 이름
#SBATCH --nodelist=n106                                 # 사용할 노드 이름
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1                               # 하나의 태스크가 사용할 CPU 코어 수
##SBATCH --mem=128G                                     # 메모리 할당량
##SBATCH --time=48:00:00                                # 최대 실행 시간 (형식: HH:MM:SS)

echo "start at: $(date)" # 접속한 날짜 표기
echo "node: $HOSTNAME" # 접속한 노드 번호 표기
echo "jobid: $SLURM_JOB_ID" # jobid 표기

# Load modules (cuda 환경)
module load cuda/11.8.0 

# Load env (python 환경)
source /home1/rldnjs16/ENTER/etc/profile.d/conda.sh     

# 가상환경 활성화
conda activate ubai    

# 필요 라이브러리 설치
pip install -r requirements.txt

# python 스크립트 실행
python t2t.py
