"""Compile the training pipeline and run it on Vertex AI.

Called by Cloud Build (ci-cd/training-pipeline.yaml, step 3) from the pipeline/ folder.
Reads PROJECT_ID, REGION and MODEL_BUCKET from environment variables.
"""
import os

from google.cloud import aiplatform
from kfp import compiler

from pipeline import training_pipeline

# 1. Compile pipeline.py into a YAML pipeline definition (published in step 4)
compiler.Compiler().compile(pipeline_func=training_pipeline, package_path="pipeline.yaml")

# 2. Run the compiled pipeline on Vertex AI Pipelines
aiplatform.init(project=os.environ["PROJECT_ID"], location=os.environ["REGION"])
job = aiplatform.PipelineJob(
    display_name="california-housing-training-job",
    template_path="pipeline.yaml",
    enable_caching=False,  # always retrain, never reuse a cached result
    pipeline_root=f"gs://{os.environ['MODEL_BUCKET']}/pipeline_root",
    parameter_values={
        "bucket_name": os.environ["MODEL_BUCKET"],
        "model_path": "models/model.pkl",
    },
)

# run() waits until the pipeline has finished and raises an error if it failed,
# so a failed training run also makes the Cloud Build fail
job.run()