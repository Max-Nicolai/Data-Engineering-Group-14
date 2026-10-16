from kfp import dsl, compiler

# 1. Define the containerized component
@dsl.container_component
def train_model_component(bucket_name: str, model_path: str):
    return dsl.ContainerSpec(
        # This image will be built and pushed by our Cloud Build CI/CD pipeline
        image='us-central1-docker.pkg.dev/<YOUR_PROJECT_ID>/labrepo/model-trainer:latest',
        command=['python', 'src/component.py'],
        args=[
            '--bucket_name', bucket_name,
            '--model_path', model_path,
        ]
    )

# 2. Define the pipeline
@dsl.pipeline(
    name='california-housing-training-pipeline',
    description='Trains a RandomForest model on California Housing data and saves it to GCS'
)
def training_pipeline(
    bucket_name: str = '<YOUR_MODELS_BUCKET_NAME>',
    model_path: str = 'model.pkl'
):
    # Execute the training step
    train_task = train_model_component(
        bucket_name=bucket_name,
        model_path=model_path
    )

if __name__ == '__main__':
    # 3. Compile the pipeline to YAML
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path='training_pipeline.yaml'
    )
    print("Pipeline successfully compiled to training_pipeline.yaml")
