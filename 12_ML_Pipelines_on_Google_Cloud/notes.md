
|[Home](../README.md)|[Course Page]()|
|---------------------|--------------|

# ML Pipelines on Google Cloud

[TOC]

##  Introduction

* [Architecture for MLOps using TFX, Kubeflow Pipelines, and Cloud Build  | Cloud Architecture Center  | Google Cloud](https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build)
* [There's a difference between CI/CD and CT Training Pipelines](https://cloud.google.com/architecture/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build#cicd_pipeline_compared_to_ct_pipeline)
  * <img src="images/image-20220321182952589.png" alt="image-20220321182952589" style="zoom:50%;" />
  * 

##  Introduction to TFX Pipelines

<img src="https://cloud.google.com/architecture/images/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build-3-tfx-google-cloud.svg" alt="Steps of a TFX-based ML system on Google Cloud." style="zoom: 67%;" />

* <img src="images/image-20220321184049895.png" alt="image-20220321184049895" style="zoom:67%;" />
* 
* 

###  TFX concepts

* Example Notebook [DL E2E | Taxi dataset - TFX E2E.ipynb - Colaboratory (google.com)](https://colab.research.google.com/gist/rafiqhasan/2164304ede002f4a8bfe56e5434e1a34/dl-e2e-taxi-dataset-tfx-e2e.ipynb)

  

<img src="images/image-20220323123807664.png" alt="image-20220323123807664" style="zoom:50%;" />

TFX components: <img src="images/image-20220323123903746.png" alt="image-20220323123903746" style="zoom:50%;" />

* Driver : Boilerplate code, you don't need to change it often. It handles job execution and feeding data to executor
* Publisher :  Boilerplate code, you don't need to change it often. Takes result of executor and updates metadata store.

How to work with components?

* <img src="images/image-20220323124150121.png" alt="image-20220323124150121" style="zoom:50%;" />
  1. We need a `config` for our component. This is done in python
  2. We need input to our component and place to store the results. ( For most component, input will come from metadata store and will be written back into the metadata store.)

Need for an orchestrator:

* An orchestrator provides a management interface 
* TFX Provides facility to <img src="images/image-20220323124624263.png" alt="image-20220323124624263" style="zoom:50%;" /> like Apache airflow and kubeflow.
* Example of DAG's on different orchestrator                                                                               

What do we store in metadata store?

* This is a relational database like SQL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
* The data itself is stored outside the metadata store, we only save the pointer to the location of the data.
*  We refer to things we store in metadata as **<u>artifacts</u>**

<img src="images/image-20220323124927473.png" alt="image-20220323124927473" style="zoom:50%;" />

* Several TFX Components run on top of Apache Beam

  * <img src="images/image-20220323125427867.png" alt="image-20220323125427867" style="zoom:50%;" />
  * Apache Beam is a unified programming model that can run on nearly any execution engine. Beam allows you to use a distributed processing you already have. 
  * TFX interoperates with several other managed google cloud services like Cloud Dataflow, and Apache Beam
  * Several TFX components use Apache Beam to implement data parallel pipelines and it means you can distribute data processing workflows using cloud data flow
  * TFX also inter operates with Vertex AI for traning and prediction 

  

* TFX Components

  * We also have a reference architecture for tfx pipeline

  <img src="images/image-20220323152030646.png" alt="image-20220323152030646" style="zoom:50%;" />

  * Ingest and split the data using <img src="images/image-20220323130747554.png" alt="image-20220323130747554" style="zoom:50%;" />

    * Instead of passing the csv file from local machine, there is also a different component called the "BigQureyGen"

      <img src="images/image-20220323151829650.png" alt="image-20220323151829650" style="zoom: 33%;" />

  * Calculate statistics of dataset using<img src="images/image-20220323130848263.png" alt="image-20220323130848263" style="zoom:50%;" />

  * Examine statistics and create data schema<img src="images/image-20220323130922748.png" alt="image-20220323130922748" style="zoom:50%;" />

  * Look for anomalies and missing value<img src="images/image-20220323130945611.png" alt="image-20220323130945611" style="zoom:50%;" />

  * To Increase predictive quality of data, feature engineering and reduce dimensionality<img src="images/image-20220323131155667.png" alt="image-20220323131155667" style="zoom:50%;" />

    * For constant values (like mean, stddev) Transform will  output tf.constants()
    * For changing values, transform will output "ops"
    * Same transformations are applied during training and serving which eliminates training/serving skew.
    * Transform eliminates the traning serving skew by running the exact same code

  * To train model<img src="images/image-20220323131806502.png" alt="image-20220323131806502" style="zoom:50%;" /><img src="images/image-20220323132040078.png" alt="image-20220323132040078" style="zoom:50%;" />

    * Training takes in the transform graph and data from "TransformGen"   and schema from "SchemaGen" and trains

    * It will output two models

      * Normal saved model: to be deployed in production
      * Eval saved model: used to analyze the performance of model.

    * You can use Tensorboard to keep track of model while training.

    * From the reference architecture you can use Vertex AI component to carry out Training jobs and also for deployment

      * <img src="images/image-20220323152948227.png" alt="image-20220323152948227" style="zoom:50%;" />

        

  * Perform deep analysis of training results<img src="images/image-20220323132219982.png" alt="image-20220323132219982" style="zoom:50%;" />

    * It looks at individual slices of the data for evaluation not just entire dataset.

  * <img src="images/image-20220323132404850.png" alt="image-20220323132404850" style="zoom:50%;" />

    * Deploy model to serving infra<img src="images/image-20220323132421193.png" alt="image-20220323132421193" style="zoom:50%;" />

      

###  TFX standard data components

###  TFX standard model components

###  TFX pipeline nodes

###  TFX libraries

###  TFX Standard Components Walkthrough

##  Pipeline orchestration with TFX

###  TFX Orchestrators

###  Apache Beam

###  TFX on Cloud AI Platform

###  TFX on Cloud AI Platform Pipelines

##  

##  ML Metadata with TFX

###  TFX Pipeline Metadata

###  TFX ML Metadata data model

###  TFX Metadata

##  Containerized Training Applications

###  Continuous Training

##  Continuous Training with Cloud Composer

![image-20220321191105955](images/image-20220321191105955.png)

###  Core Concepts of Apache Airflow

###  Continuous Training Pipelines with Cloud Composer

##  ML Pipelines with MLflow

###  Introduction

###  Overview of ML development challenges

###  How MLflow tackles these challenges

###  MLflow tracking

###  MLflow projects

###  MLflow models

###  MLflow model registry

##  Summary

###  Course Summary