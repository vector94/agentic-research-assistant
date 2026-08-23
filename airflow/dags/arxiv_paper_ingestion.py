from datetime import UTC, datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

from arxiv_ingestion.fetching import fetch_daily_papers
from arxiv_ingestion.indexing import index_papers_hybrid
from arxiv_ingestion.reporting import generate_daily_report

# Import task functions from modular structure
from arxiv_ingestion.setup import setup_environment

# Default DAG arguments
default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=30),
}

# Create the DAG
dag = DAG(
    "arxiv_paper_ingestion",
    default_args=default_args,
    description="Daily arXiv CS.AI paper pipeline: fetch, store, chunk, embed, and index",
    schedule="0 6 * * 1-5",  # Monday-Friday at 6 AM UTC
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    max_active_runs=1,
    catchup=False,
    tags=["arxiv", "papers", "ingestion", "hybrid-search", "embeddings", "chunks"],
)

# Task definitions
setup_task = PythonOperator(
    task_id="setup_environment",
    python_callable=setup_environment,
    dag=dag,
)

fetch_task = PythonOperator(
    task_id="fetch_daily_papers",
    python_callable=fetch_daily_papers,
    dag=dag,
)

# Hybrid search indexing task
index_hybrid_task = PythonOperator(
    task_id="index_papers_hybrid",
    python_callable=index_papers_hybrid,
    dag=dag,
)

report_task = PythonOperator(
    task_id="generate_daily_report",
    python_callable=generate_daily_report,
    dag=dag,
)

cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command=(
        "find /opt/airflow/data/arxiv_pdfs -name '*.pdf' -type f "
        "-mtime +${PDF_RETENTION_DAYS:-30} -delete 2>/dev/null || true"
    ),
    dag=dag,
)

# Task dependencies
# Pipeline: setup -> fetch -> hybrid index -> report -> cleanup
setup_task >> fetch_task >> index_hybrid_task >> report_task >> cleanup_task
