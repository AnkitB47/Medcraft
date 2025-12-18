from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'medcraft',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 1. Data Pipeline
with DAG('medcraft_data_pipeline', default_args=default_args, schedule_interval='@daily', catchup=False) as data_dag:
    download_data = BashOperator(
        task_id='download_data',
        bash_command='python scripts/download_data.py'
    )
    
    preprocess_data = BashOperator(
        task_id='preprocess_data',
        bash_command='python scripts/preprocess_data.py'
    )
    
    dvc_push = BashOperator(
        task_id='dvc_push',
        bash_command='dvc push'
    )
    
    download_data >> preprocess_data >> dvc_push

# 2. Training Pipeline
with DAG('medcraft_training_pipeline', default_args=default_args, schedule_interval='@weekly', catchup=False) as train_dag:
    train_nanovlm = BashOperator(
        task_id='train_nanovlm',
        bash_command='python services/nanovlm/finetune.py'
    )
    
    train_yolo = BashOperator(
        task_id='train_yolo',
        bash_command='yolo detect train model=yolov8n.pt data=data/coco128.yaml epochs=10 imgsz=640'
    )
    
    # Titans training would be here
    
    register_models = BashOperator(
        task_id='register_models',
        bash_command='python scripts/register_models.py'
    )
    
    [train_nanovlm, train_yolo] >> register_models

# 3. Evaluation Pipeline
with DAG('medcraft_evaluation_pipeline', default_args=default_args, schedule_interval='@weekly', catchup=False) as eval_dag:
    eval_nanovlm = BashOperator(
        task_id='eval_nanovlm',
        bash_command='python eval/eval_nanovlm.py'
    )
    
    eval_yolo = BashOperator(
        task_id='eval_yolo',
        bash_command='python eval/eval_yolo.py'
    )
    
    eval_titans = BashOperator(
        task_id='eval_titans',
        bash_command='python eval/eval_titans.py'
    )
    
    generate_report = BashOperator(
        task_id='generate_report',
        bash_command='python eval/generate_report.py'
    )
    
    [eval_nanovlm, eval_yolo, eval_titans] >> generate_report

# 4. Deployment Pipeline
with DAG('medcraft_deployment_pipeline', default_args=default_args, schedule_interval=None, catchup=False) as deploy_dag:
    build_images = BashOperator(
        task_id='build_images',
        bash_command='docker-compose build'
    )
    
    push_images = BashOperator(
        task_id='push_images',
        bash_command='docker-compose push'
    )
    
    deploy_helm = BashOperator(
        task_id='deploy_helm',
        bash_command='helm upgrade --install medcraft ops/helm/medcraft --namespace medcraft --create-namespace'
    )
    
    smoke_tests = BashOperator(
        task_id='smoke_tests',
        bash_command='python tests/smoke_tests.py'
    )
    
    build_images >> push_images >> deploy_helm >> smoke_tests
