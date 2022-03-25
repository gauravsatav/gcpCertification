# Best practices for implementing machine learning on Google Cloud

[Best practices for implementing machine learning on Google Cloud  | Cloud Architecture Center](https://cloud.google.com/architecture/ml-on-gcp-best-practices)

[TOC]



## Recommended Tools

https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-recommended-tools-and-products

### Before using custom models check if other tools are enough

1. BigQueryML

2. AutoML

   6,11,4,5,

   3,3,2,3

## 1. [Machine learning environment setup](https://cloud.google.com/architecture/ml-on-gcp-best-practices#machine-learning-environment-setup)

6 things to remember.

[Use Vertex AI Workbench user-managed notebooks for experimentation and development](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-notebooks-for-experimentation-and-development).
[Create a user-managed notebooks instance for each team member](https://cloud.google.com/architecture/ml-on-gcp-best-practices#create-a-notebooks-instance-for-each-team-member).
[Help secure PII in a user-managed notebooks instance](https://cloud.google.com/architecture/ml-on-gcp-best-practices#help-secure-pii-in-your-notebooks).
[Store prepared data and your model in the same project](https://cloud.google.com/architecture/ml-on-gcp-best-practices#store-prepared-data-and-your-model-in-the-same-project).
[Optimize performance and cost](https://cloud.google.com/architecture/ml-on-gcp-best-practices#optimize-performance-cost).
[Use Vertex SDK for Python](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-vertex-sdk-for-python).



## 2. [Machine learning development](https://cloud.google.com/architecture/ml-on-gcp-best-practices#machine-learning-development)

11 things


###  [Prepare training data](https://cloud.google.com/architecture/ml-on-gcp-best-practices#prepare-training-data).


###  [Store tabular data in BigQuery](https://cloud.google.com/architecture/ml-on-gcp-best-practices#store-tabular-data-in-bigquery).


###  [Store image, video, audio and unstructured data on Cloud Storage](https://cloud.google.com/architecture/ml-on-gcp-best-practices#store-image-video-audio-and-unstructured-data-on-cloud-storage).


###  [Use Vertex Data Labeling for unstructured data](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-vertex-data-labeling).


###  [Use Vertex AI Feature Store with structured data](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-vertex-feature-store-with-structured-data).


###  [Avoid storing data in block storage](https://cloud.google.com/architecture/ml-on-gcp-best-practices#avoid-storing-data-in-block-storage).


###  [Use Vertex AI TensorBoard to visualize experiments](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-vertex-tensorboard-to-visualize-experiments).


###  [Train a model within a user-managed notebooks instance for small datasets](https://cloud.google.com/architecture/ml-on-gcp-best-practices#train-a-model-within-notebooks-for-small-datasets).


###  [Maximize your model's predictive accuracy with hyperparameter tuning](https://cloud.google.com/architecture/ml-on-gcp-best-practices#maximize-your-model's-predictive-accuracy-with-hyperparameter-tuning).


###  [Use a user-managed notebooks instance to evaluate and understand your models](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-notebooks-to-evaluate-and-understand-your-models). 

* [What-if Tool (WIT)](https://www.youtube.com/watch?v=qTUUwfG1vSs) and [Language Interpretability Tool (LIT)](https://pair-code.github.io/lit/)


###  [Use feature attributions to gain insights into model predictions](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-feature-attributions-to-gain-insights-into-model-predictions).

* [Vertex Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview) 

## 3. [Data processing](https://cloud.google.com/architecture/ml-on-gcp-best-practices#data-processing)

4 things

[Use TensorFlow Extended when leveraging TensorFlow ecosystem ](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-tensorflow-extended-when-leveraging-tensorflow-ecosystem).
[Use BigQuery to process tabular data](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-bigquery-to-process-tabular-data).
[Use Dataflow to process unstructured data](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-dataflow-to-process-unstructured-data).
[Use managed datasets to link data to your models](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-managed-datasets-to-link-data-to-your-models).

## 4. [Operationalized training](https://cloud.google.com/architecture/ml-on-gcp-best-practices#operationalized-training)

5 things

[Run your code in a managed service](https://cloud.google.com/architecture/ml-on-gcp-best-practices#run-your-code-in-a-managed-service).
[Operationalize job execution with training pipelines](https://cloud.google.com/architecture/ml-on-gcp-best-practices#operationalize-job-execution-with-training-pipelines).
[Use training checkpoints to save the current state of your experiment](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-training-checkpoints-to-save-the-current-state-of-your-experiment).
[Prepare production artifacts for serving in Cloud Storage](https://cloud.google.com/architecture/ml-on-gcp-best-practices#prepare-production-artifacts-for-serving-in-cloud-storage).
[Regularly compute new feature values](https://cloud.google.com/architecture/ml-on-gcp-best-practices#regularly-compute-new-feature-values).

## 5. [Model deployment and serving](https://cloud.google.com/architecture/ml-on-gcp-best-practices#model-deployment-and-serving)

3 things

[Specify the number and types of machines you need](https://cloud.google.com/architecture/ml-on-gcp-best-practices#specify-the-number-and-types-of-machines-you-need).
[Plan inputs to the model](https://cloud.google.com/architecture/ml-on-gcp-best-practices#plan-inputs-to-the-model).
[Turn on automatic scaling](https://cloud.google.com/architecture/ml-on-gcp-best-practices#turn-on-automatic scaling).


## 6. [Machine learning workflow orchestration](https://cloud.google.com/architecture/ml-on-gcp-best-practices#machine-learning-workflow-orchestration)

3 things

[Use ML pipelines to orchestrate the ML workflow](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-ml-pipelines).
[Use Kubeflow Pipelines for flexible pipeline construction](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-kubeflow-pipelines-sdk-for-flexible-pipeline-construction).
[Use TensorFlow Extended SDK to leverage pre-built components for common steps](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-tensorflow-extended-sdk-to-leverage-pre-built-components-for-common-steps).

## 7. [Artifact organization](https://cloud.google.com/architecture/ml-on-gcp-best-practices#artifact-organization)

2 things

[Organize your ML model artifacts](https://cloud.google.com/architecture/ml-on-gcp-best-practices#organize-your-ml-model-artifacts).
[Use a Git repository for pipeline definitions and training code](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-a-git-repository-for-pipeline-definitions-and-training-code).

## 8. [Model monitoring](https://cloud.google.com/architecture/ml-on-gcp-best-practices#model-monitoring)

3 things

[Use skew detection](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-skew-detection).
[Fine tune alert thresholds](https://cloud.google.com/architecture/ml-on-gcp-best-practices#fine-tune-alert-thresholds).
[Use feature attributions to detect data drift or skew](https://cloud.google.com/architecture/ml-on-gcp-best-practices#use-feature-attributions-to-detect-data-drift-or-skew).