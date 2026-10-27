# California Housing Price Prediction – MLOps on Google Cloud

Data Engineering 2026-2027, Assignment 1 (Group 14).

An ML application that predicts the median house value of a California block group
from 8 census features, with a complete MLOps system on Google Cloud:
a Vertex AI training pipeline, a prediction API and UI on Cloud Run, and
Cloud Build CI/CD pipelines for continuous training and continuous deployment.

- **Dataset:** California Housing (1990 U.S. census, 20,640 block groups), loaded with
  `sklearn.datasets.fetch_california_housing`
- **Model:** `RandomForestRegressor` (20 trees), evaluated on a 20% hold-out test set
  (R² ≈ 0.80, MAE ≈ $33,600)

## Architecture

```
                 push (pipeline/**)            weekly schedule (Mon 03:00)
                         │                               │
          continuous-training-trigger           training-scheduled  ◄── Cloud Scheduler
                         └──────────────┬────────────────┘
                                        ▼
            Cloud Build: ci-cd/training-pipeline.yaml
            1. build trainer image   2. push to Artifact Registry
            3. compile + run the pipeline on Vertex AI (waits for the result)
            4. publish compiled pipeline → gs://<bucket>/pipelines/training_pipeline-<commit>.yaml
                                        │
                                        ▼
            Vertex AI Pipelines: train-model-component
            load data → train → evaluate (R², MAE) → quality gate (R² ≥ 0.7) → upload
                                        │
                                        ▼
            gs://<bucket>/models/model.pkl  ──►  Pub/Sub topic housing_model_update
                                                         │
                     push (serving/**)                   ▼
                            │                   serving-on-model-update
               serving-deployment-trigger                │
                            └───────────┬────────────────┘
                                        ▼
            Cloud Build: ci-cd/serving-pipeline.yaml
            1. download model   2. pytest (API + model)   3–5. build/push/deploy API
            6–8. build/push/deploy UI (with the API URL)
                                        │
                                        ▼
            Cloud Run: housing-prediction-api  ◄──  housing-prediction-ui  ◄──  user
```

## Repository structure

| Path | Content |
|---|---|
| `pipeline/pipeline.py` | Kubeflow pipeline definition (one container component) |
| `pipeline/submit_pipeline.py` | Compiles the pipeline and runs it on Vertex AI (called by Cloud Build) |
| `pipeline/components/model_trainer/` | Training component: `src/component.py`, `Dockerfile`, `requirements.txt` |
| `serving/api/` | Flask prediction API (`POST /predict`), `Dockerfile`, tests in `tests/` |
| `serving/ui/` | Flask web UI that calls the API, `Dockerfile`, `templates/index.html` |
| `ci-cd/training-pipeline.yaml` | Cloud Build config: training operationalization (CI/CD + CT) |
| `ci-cd/serving-pipeline.yaml` | Cloud Build config: model and serving app deployment (CD) |
| `data/train.py` | Original experimentation script (local baseline model) |

## Training pipeline

`pipeline/components/model_trainer/src/component.py` runs inside Vertex AI:

1. loads the California Housing dataset and splits it 80/20 (`random_state=42`)
2. trains a `RandomForestRegressor(n_estimators=20)`
3. evaluates it on the test set and logs `Test R2` and `Test MAE`
4. **quality gate:** if R² < 0.7 the step fails and no model is uploaded,
   so the currently deployed model stays live
5. uploads the model to `gs://<bucket>/models/model.pkl`

Caching is disabled (`enable_caching=False`), so every run retrains the model.
`job.run()` waits for the pipeline, so a failed training run also fails the Cloud Build.

## Prediction API and UI

`POST /predict` on `housing-prediction-api` with the 8 features:

```json
{"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.984, "AveBedrms": 1.024,
 "Population": 322, "AveOccup": 2.556, "Latitude": 37.88, "Longitude": -122.23}
```

returns the predicted price in dollars, e.g. `{"predicted_price": 414860.2}`.
The UI (`housing-prediction-ui`) shows a form for these features and calls the API.

## CI/CD triggers (Cloud Build, region us-central1)

| Trigger | Event | Build config |
|---|---|---|
| `continuous-training-trigger` | push to `main`, only if `pipeline/**` or `ci-cd/training-pipeline.yaml` changed | `ci-cd/training-pipeline.yaml` |
| `training-scheduled` | manual, run weekly by Cloud Scheduler job `weekly-training` (`0 3 * * 1`) | `ci-cd/training-pipeline.yaml` |
| `serving-on-model-update` | Pub/Sub topic `housing_model_update` (new object in `models/`) | `ci-cd/serving-pipeline.yaml` |
| `serving-deployment-trigger` | push to `main`, only if `serving/**` or `ci-cd/serving-pipeline.yaml` changed | `ci-cd/serving-pipeline.yaml` |

Substitution variables: `_REGION=us-central1`, `_ARTIFACT_REPO=labrepo`,
`_MODEL_BUCKET=data-engineering-509009-models`.

## Google Cloud resources

| Resource | Name |
|---|---|
| Artifact Registry (Docker) | `labrepo`: `model-trainer`, `prediction-api`, `prediction-ui` |
| Cloud Storage (model repository) | `data-engineering-509009-models`: `models/`, `pipelines/`, `pipeline_root/` |
| Bucket notification | `models/` → Pub/Sub `housing_model_update` (`OBJECT_FINALIZE`) |
| Cloud Run | `housing-prediction-api` (port 5000), `housing-prediction-ui` (port 5001) |

## Running a build manually

From the repository root (for example in Cloud Shell):

```bash
gcloud builds submit . --region=us-central1 --config=ci-cd/training-pipeline.yaml \
  --substitutions=_REGION=us-central1,_ARTIFACT_REPO=labrepo,_MODEL_BUCKET=data-engineering-509009-models
```

Use `--config=ci-cd/serving-pipeline.yaml` for the deployment pipeline.

## Design decisions

| Decision | Reason |
|---|---|
| Unique Cloud Run service names (`housing-prediction-*`) | The lab pipelines deployed services named `prediction-api/ui` in the same project and overwrote the app |
| Vertex AI caching disabled | With caching, a run "succeeded" in seconds by reusing the previous result instead of retraining |
| Model stored under `models/` | A dedicated folder lets the bucket notification fire only for new models, not for pipeline files |
| Compiled pipeline published per commit (`$SHORT_SHA`) | Every pipeline version is stored and traceable to the exact source code |
| Pub/Sub trigger for deployment | Serving starts only after a new model exists, so it always deploys the newest model (with push triggers alone, training and serving ran in parallel and serving deployed the previous model) |
| Included-files filters on push triggers | A change only runs the pipeline it affects; documentation changes run nothing |
| Separate manual trigger for the schedule | Cloud Build schedules can only run manual triggers |
| Tests before deployment | A broken API or model never replaces the running version |
| Quality gate in training (R² ≥ 0.7) | A bad model is never uploaded, so it is never deployed |
| `job.run()` instead of `job.submit()` | The build waits for Vertex AI, so a failed or rejected training run shows as a failed build |
| `scikit-learn==1.9.1` pinned in trainer and API | The model is trained and loaded with the same library version |


## Known limitations and possible improvements

- **Models are not versioned:** each run overwrites `models/model.pkl`; a model registry or
  archived copies per commit would allow rollback.
- **Static data:** the dataset does not change, so scheduled retraining produces the same model;
  a real system would ingest new data from a data bucket.
- **Single pipeline component:** ingestion, training, evaluation and upload could be separate
  components, and the new model could be compared with the production model before upload.
- **Images tagged `latest` only:** tagging with the commit hash would allow redeploying older versions.
- **No monitoring:** prediction and data drift monitoring with alerts that trigger retraining
  (outside the scope of this assignment).

## References

- Dataset: Pace, R. K. and Barry, R. (1997). Sparse Spatial Autoregressions.
  *Statistics & Probability Letters*, 33(3), 291–297. Available through
  [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).
- Course labs (DE2026, Labs 2–5) for the Cloud Build, Vertex AI and Cloud Run patterns.
