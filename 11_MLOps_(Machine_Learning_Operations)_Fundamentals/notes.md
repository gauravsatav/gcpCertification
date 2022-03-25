
|[Home](../README.md)|[Course Page]()|
|---------------------|--------------|

# MLOps (Machine Learning Operations) Fundamentals

[TOC]


##  Introduction to MLOps Fundamentals

###  Course Introduction

##  Why and When to Employ MLOps

* Pain Points:

* the real challenge isn't building an ml model it is building an integrated ml system in continuously operating it in production

  * keeping track of the many models they have trained is difficult they want to keep track of the different versions of the code the values they chose for the different hyper parameters and the metrics they are evaluating 

  * they have trouble keeping track of which ideas have been tried which ones worked and which ones did not 

  * they cannot pinpoint the best model which was possibly trained two weeks previously reproduce it and run it on full production data 

  * reproducibility is a major concern because data scientists want to be able to rerun the best model with a more through parameter sweep

  * putting a model in production is difficult unless it can be reproduced because many companies have that as a policy or requirement 

  * when the team is able to successfully train a model and make it ready for production usage in a streamlined fashion performance and agility are considerably

  *  traceability becomes paramount

    <img src="images/image-20220320123919825.png" alt="image-20220320123919825" style="zoom: 67%;" /><img src="images/image-20220320124008492.png" alt="image-20220320124008492" style="zoom:67%;" />

    <img src="images/image-20220320124812700.png" alt="image-20220320124812700" style="zoom:50%;" />

    

###  Machine Learning Lifecycle

###  MLOps Architecture and TensorFlow Extended Components

* <img src="images/image-20220320125652742.png" alt="image-20220320125652742" style="zoom:80%;" />
* <img src="images/image-20220320125728129.png" alt="image-20220320125728129" style="zoom:80%;" />
* 

##  Introduction

###  Introduction to Containers

###  Containers and Container Images

###  Lab introduction

###  Working with Cloud Build

###  Lab solution

###  Introduction to Kubernetes

[Kubernetes Components explained! Pods, Services, Secrets, ConfigMap | Kubernetes Tutorial 14 - YouTube](https://www.youtube.com/watch?v=Krpb44XR0bk)



[Kubernetes Components explained! Pods, Services, Secrets, ConfigMap | Kubernetes Tutorial 14 - YouTube](https://www.youtube.com/watch?v=Krpb44XR0bk)

* Key terms
  * Cluster: 
  * Master Node:
    * 
  * Worker Nodes:
    * Pods
    * Container
  * Volumes:
  * Networking:
    * Service: In case a pod stops working and new pod is created, it is assigned a new IP address. The "Service" component is an abstraction on top of this which keeps a track of the changing underlying IP address so that we don't have to keep a track of the IP address.
    * Ingress:
  * Declarative Componets:
    * Deployments: This is where we declare the state we require our desired application to be in. This is not used for stateful services like database, because pods are considered to be effemeral (i.e. can be destroyed easily) components. This is where you write the config.yaml file.
    * Stateful Sets: 
  * Environment Variable Storage
    * ConfigMap : Instead of hardcoding the ip address of database or any other service, we can declare it here. This ensures that we don't have to make changes in our code, build new image if the url of the database changes.
    * Secrets: This is same like config map but instead of links to different services, we save our username and password here.

###  Introduction to Google Kubernetes Engine

###  Compute Options Detail

###  Kubernetes Concepts

###  The Kubernetes Control Plane

###  Google Kubernetes Engine Concepts

###  Lab Introduction

###  Deploying Google Kubernetes Engine

###  Lab Solution

###  Deployments 1

###  Ways to Create Deployments

###  Services and Scaling

###  Updating Deployments

###  Rolling Updates

###  Canary Deployments

###  Managing Deployments

###  Lab Intro

###  Creating Google Kubernetes Engine Deployments

###  Jobs and CronJobs

###  Parallel Jobs

###  CronJobs

##  Introduction to AI Platform Pipelines

###  Overview

<img src="images/image-20220320130359344.png" alt="image-20220320130359344" style="zoom: 67%;" />

<img src="images/image-20220320131009173.png" alt="image-20220320131009173" style="zoom:67%;" />

<img src="images/image-20220320131106438.png" alt="image-20220320131106438" style="zoom:50%;" />

<img src="images/image-20220320131141227.png" alt="image-20220320131141227" style="zoom:50%;" />

###  Introduction to AI Platform Pipelines

* <img src="images/image-20220320131329021.png" alt="image-20220320131329021" style="zoom:50%;" />

* <img src="images/image-20220320131430098.png" alt="image-20220320131430098" style="zoom:67%;" />

  * pipeline components are self-contained sets of code that perform one step in a pipeline's workflow such as data preprocessing data transformation model training and so on 
  * <img src="images/image-20220320131613767.png" alt="image-20220320131613767" style="zoom:50%;" />
  * Tasks are instances of pipeline components
    * They have input parameters,outputs and container image.
  * to create the workflow graph ai platform pipelines analyzes the task dependencies. for eg.
    * the pre-processing task does not depend on any other tasks so it can be the first task in the workflow or it can run concurrently with other tasks
    * the training task relies on the data produced by the pre-processing task so training must occur after preprocessing 
    * the prediction task relies on the trained model produced by the training task so prediction must occur after the training task or the training step 
    * building the confusion metrics and performing roc receiver operating characteristic analysis both rely on the output of the prediction task so they must occur after prediction is complete 
    * building the confusion metrics and performing roc analysis can occur concurrently because they both depend on the output of the prediction task but they are independent of each other 
    * based on this analysis the ai platform pipeline systems runs the pre-processing training and prediction tasks sequentially and then runs the confusion matrix and roc tasks concurrently

* Why was AI Platform needed?

  * <img src="images/image-20220320132514336.png" alt="image-20220320132514336" style="zoom: 67%;" />
  * <img src="images/image-20220320132541333.png" alt="image-20220320132541333" style="zoom:50%;" /><img src="images/image-20220320132659396.png" alt="image-20220320132659396" style="zoom:67%;" />

* Components of Google AI Platform:

  * <img src="images/image-20220320132814024.png" alt="image-20220320132814024" style="zoom:50%;" /><img src="images/image-20220320132844824.png" alt="image-20220320132844824" style="zoom:67%;" />
  * <img src="images/image-20220320132925329.png" alt="image-20220320132925329" style="zoom:67%;" />

* How to setup Kubeflow Pipelines?

  <img src="images/image-20220320133028613.png" alt="image-20220320133028613" style="zoom:67%;" />

  * You specify your pipeline by using the kubeflow pipelines sdk or by customizing the tensorflow extended

  * ai pipelines uses the argo workflow engine to run the pipeline and has additional microservices to record metadata handle components i o and schedule pipeline runs.

  * pipeline steps are executed as individual isolated pods in a gke google kubernetes engine cluster which enables the kubernetes native experience for the pipeline components 

  * the components can leverage google cloud services such as data flow ai platform training and prediction and bigquery for handling scalable computation and data processing 

  * the pipelines can also contain steps that perform sizable gpu and tpu computation in the cluster directly leveraging gke auto scaling and node auto provisioning 

    <img src="images/image-20220320133635974.png" alt="image-20220320133635974" style="zoom:67%;" />

  * the kubeflow pipeline's sdk is a lower level sdk that's ml framework neutral and enables direct kubernetes resource control and simple sharing of containerized components as shown in the pipeline steps 

  * the tfx sdk is currently in preview mode and is designed for ml workloads. 

    * It provides a higher level abstraction with prescriptive but customizable components with predefined machine learning types that represent google best practices for durable and scalable machine learning pipelines 
    * it also comes with a collection of customizable tensorflow optimized templates developed and used internally at google consisting of component archetypes for production machine learning 
    * you can configure the pipeline templates to build train and deploy your model with your own data,
    * automatically perform schema inference data validation model evaluation and model analysis and automatically deploy your trained model to the ai platform prediction service 

* Features of AI Pipelines:

  * Feature 1- TFX Examples:
    * to make it easier for developers to get started with machine learning pipeline code the tfx sdk provides templates or scaffolds with step-by-step guidance on building a production machine learning pipeline for your own data 
    * with a tfx template you can incrementally add different components to the pipeline and iterate on them 
    * tfx templates can be accessed by the ai platform pipelines getting started page in the google cloud console
    * the tfx sdk currently provides a template for classification problem types and is optimized for tensorflow with more templates being designed for different use cases and problem types
  * Feature 2- Pipeline Versioning:
    * ai platform pipelines also supports pipeline versioning it lets you upload multiple versions of the same pipeline and group them in the ui so you can manage semantically related workflows together
  * Feature 3 - Artifact tracking.
  * Feature 4 - Lineage Tracking.

* When to use AI Pipelines?

  Below are top 3 features AI Pipelines enable out of the box:

  * Workflow Orchestration : this feature provides you to actually visualize your pipeline and various components used in it as a lego train

    <img src="images/image-20220320135628756.png" alt="image-20220320135628756" style="zoom:67%;" />

  * Rapid reliable and repeatable experimentation: this is a very important feature for any organization to run quick experiments and make decisions accordingly

    * keep track of metrics and outputs
    * you can clone pipelines

  * Ecosystem to share reuse and compose your work:

    * <img src="images/image-20220320140504659.png" alt="image-20220320140504659" style="zoom:67%;" />

    * <img src="images/image-20220320142336288.png" alt="image-20220320142336288" style="zoom:67%;" />

    * AI HUB:

      <img src="images/image-20220320142429651.png" alt="image-20220320142429651" style="zoom:50%;" />

      <img src="images/image-20220320142447940.png" alt="image-20220320142447940" style="zoom:50%;" />

      

###  Running AI Platform Pipelines

##  System and concepts overview

###  Create a reproducible dataset

* Use of hashing
  * To ensure that the same training examples end up in the training bucket every time, we use hashing
  * Join all the column into a single dummy string column and compute the has for that column
  * Then divide it into training, validation and test datasets by using modulo function.

###  Implement a tunable model

###  Build and push a training container

###  Train and tune the model

###  Serve and query the model

###  Using custom containers with AI Platform Training

###  Lab Solution

##  Kubeflow Pipelines on AI Platform

* <img src="images/image-20220323143126125.png" alt="image-20220323143126125" style="zoom:50%;" />
  * the smallest component in the above graph is the top most (tensorflow) which is engulfed in the bigger component (TFX) which in turns is controlled by Kubeflow pipelines and so on..
* Cloud setup for using kubeflow.
  * When you create kubeflow instance in Google cloud, you also get a notebook server instance along with it.
* In kubeflow we have two steps
  1. Define the ops (operations)
  2. Connect them via a graph

###  System and concept overview

* As long as a container follows a protocol as to how they accept inputs and how they produce output any containerized task can be converted into a workflow step.

## Steps for writing pipeline.

### Step 1  and Step 2

<img src="images/image-20220323141942024.png" alt="image-20220323141942024" style="zoom:80%;" />

### Step 3  Define Ops

<img src="images/image-20220323142041512.png" alt="image-20220323142041512" style="zoom:80%;" /><img src="images/image-20220323142158186.png" alt="image-20220323142158186" style="zoom:80%;" />

### Step 4 Define Pipeline

<img src="images/image-20220323142235244.png" alt="image-20220323142235244" style="zoom:80%;" />

### Step 5 Compile Pipeline

<img src="images/image-20220323142316175.png" alt="image-20220323142316175" style="zoom:80%;" />

### Step 6 Submit Pipeline Run

<img src="images/image-20220323142411143.png" alt="image-20220323142411143" style="zoom:80%;" />

### Step 7 Change the pipeline parameters and run pipeline again.

<img src="images/image-20220323142513656.png" alt="image-20220323142513656" style="zoom:80%;" />

* The parameters defined in the below step, become available to us in the Cloud portal when we define a pipleline run in kubeflow dashboard.<img src="images/image-20220323141348475.png" alt="image-20220323141348475" style="zoom:67%;" />

  <img src="images/image-20220323141512565.png" alt="image-20220323141512565" style="zoom:50%;" />

  

* Using the decorator as mentioned here, this will convert your custom code into a docker image... <img src="images/image-20220323135258402.png" alt="image-20220323135258402" style="zoom:50%;" />

* The pipeline ops can be TFX components and using those we can create a Kubeflow pipeline with TFX components.

* <img src="images/image-20220323142754765.png" alt="image-20220323142754765" style="zoom:80%;" />

* For running TFX Components on the kubeflow pipeline use the following steps

  Cloud setup for using kubeflow.

  * When you create kubeflow instance in Google cloud, you also get a notebook server instance along with it.

  * ![image-20220323143704261](images/image-20220323143704261.png)

    ![image-20220323143745252](images/image-20220323143745252.png)

    <img src="images/image-20220323143823638.png" alt="image-20220323143823638" style="zoom: 80%;" />

    <img src="images/image-20220323143856207.png" alt="image-20220323143856207" style="zoom:67%;" />

    <img src="images/image-20220323143927122.png" alt="image-20220323143927122" style="zoom:67%;" />

  * ![image-20220323144018305](images/image-20220323144018305.png)

    <img src="images/image-20220323144031532.png" alt="image-20220323144031532" style="zoom:67%;" />

    <img src="images/image-20220323144050257.png" alt="image-20220323144050257" style="zoom:67%;" />

    * This steps create copies a template folder. <img src="images/image-20220323144218165.png" alt="image-20220323144218165" style="zoom:67%;" />

      

  * <img src="images/image-20220323144450165.png" alt="image-20220323144450165" style="zoom:67%;" />

    * As seen above inside the pipeline folder we have a `pipeline.py` file. 

      * This file is where you define the TFX Pipeline

        * Step 1  Import libraries.<img src="images/image-20220323151028748.png" alt="image-20220323151028748" style="zoom:50%;" />

        * Step 2: Create TFX Pipeline

          <img src="images/image-20220323151137304.png" alt="image-20220323151137304" style="zoom:67%;" />

          <img src="images/image-20220323151235842.png" alt="image-20220323151235842" style="zoom:80%;" />

          

    * The `kubeflow_v2_dag_runner.py` file defines runners for each orchestration engine.

      * i.e. the `kubeflow_runner.py`  and
      * `local_runner.py`

  * ![image-20220323145258225](images/image-20220323145258225.png)

    * First copy data from local system to cloud

      ![image-20220323145325081](images/image-20220323145325081.png)

    * ![image-20220323145751263](images/image-20220323145751263.png)

    * After exection of above command additonal files are created in the ml-pipeline folder as below

      * Dockerfile
      * build.yaml
      * my_pipeline.tar.gz - This is the pipeline definition file which will be used by "**<u>Argo</u>**"

      <img src="images/image-20220323145924717.png" alt="image-20220323145924717" style="zoom:67%;" />

    * ![image-20220323145808807](images/image-20220323145808807.png)

###  Describing a Kubeflow Pipeline with KF DSL

###  Lightweight Python Components

###  Custom components

###  Continuous Training Pipeline with Kubeflow Pipeline and Cloud AI Platform

###  Lab Solution

##  Concept Overview

* The goal is to run a cloud builder whenever a trigger is detected. This requires two things

  * The actual individual cloud builder which will build the containers.

  * A config file to tell which cloud builders to run 

  * Trigger the actual build using the following command

    <img src="images/image-20220324113353960.png" alt="image-20220324113353960" style="zoom:67%;" />

    

###  Cloud Build Builders

<img src="images/image-20220324111517061.png" alt="image-20220324111517061" style="zoom:67%;" />

<img src="images/image-20220324111544990.png" alt="image-20220324111544990" style="zoom:67%;" />

* Standard Builders are already packaged configuration actions that are really common such as building a docker container and pushing that docker container to a registry all these pre-packaged configuration actions are in the google cloud platform cloud builders repository.

  * <img src="images/image-20220324111913328.png" alt="image-20220324111913328" style="zoom:67%;" />
  * Components of a standard Cloud Builders.<img src="images/image-20220324111957874.png" alt="image-20220324111957874" style="zoom:50%;" />
    * They boil down to two parts 
      * one a script that executes the configuration actions and 
      * two a docker container that wraps the script with all the dependencies that it needs to be executed
      * Example Docker File<img src="images/image-20220324112512054.png" alt="image-20220324112512054" style="zoom: 67%;" />
      * Example Script File <img src="images/image-20220324112610816.png" alt="image-20220324112610816" style="zoom:67%;" />

* Custom Builders:  custom cloud builders are special configuration actions that you have to package into docker containers yourself generally these are pushed to your own google cloud registry

  * <img src="images/image-20220324112703830.png" alt="image-20220324112703830" style="zoom:67%;" />

  

###  Cloud Build Configuration

* remember that the goal is to run a cloud builder whenever a trigger is detected well this raises the question how does cloud build know which cloud builder to run well the answer lies in a cloud build configuration file we tell cloud build which builders to run in a cloudbuild.yaml.

* cloudbuild.yaml describes the cloud builder to be run and what arguments should be passed to the entry point command defined in the corresponding docker file 

* <img src="images/image-20220324113150453.png" alt="image-20220324113150453" style="zoom:67%;" />

  * the name is the uri of the corresponding cloud builder container 

  * the args contains the arguments to be passed to the entry point.

  * Trigger Cloud Build using the command

    <img src="images/image-20220324113353960.png" alt="image-20220324113353960" style="zoom:67%;" />

  * A single cloud builder step as defined in the above config file

    <img src="images/image-20220324115838706.png" alt="image-20220324115838706" style="zoom:67%;" />

    * The `dir` variable is a persistent data storage which will allow the data to be shared by the next pipeline step.

      <img src="images/image-20220324120029093.png" alt="image-20220324120029093" style="zoom:50%;" />

  * In the above command to run the actual cloud build we have a $Substitution variable which allows us to substitue variable values before the cloud build is run

    * For example in the while runninng the cloud build command below we are substituting the 
      * $_PIPELNE_FOLDER = . (this is the current directory)
      * $_IMAGE_NAME with value = trainer_base

    <img src="images/image-20220324120233670.png" alt="image-20220324120233670" style="zoom:67%;" />

    * ​	Another way to passing dynamic values is to pass environment variables.<img src="images/image-20220324120652352.png" alt="image-20220324120652352" style="zoom:67%;" />

* Difference between custom and standard builder

  * Nothing except the uri will point to our own image in gcr<img src="images/image-20220324120519393.png" alt="image-20220324120519393" style="zoom:67%;" />

* After building the images we need to push them

  * Add the `Images` section to your config.yaml file.
  * <img src="images/image-20220324120850757.png" alt="image-20220324120850757" style="zoom:67%;" />
  * note you don't actually need to specify where to push these containers because the location is already given by the name of the container image that you built 
  * if you forget this field your container will simply be built but not pushed so not available in production so it's very important to remember to push

###  Cloud Build Triggers

* Trigger 1: Manual<img src="images/image-20220324121236565.png" alt="image-20220324121236565" style="zoom:67%;" />

  * <img src="images/image-20220324121322147.png" alt="image-20220324121322147" style="zoom:67%;" />

* Trigger 2: Automatically

  <img src="images/image-20220324121414567.png" alt="image-20220324121414567" style="zoom:67%;" />

  * Step 1: Link the Github Repo with the project.

    * Go to github and activate Cloud Build App
    * Allow access to the required repository.
    * On the GCP side, specify which repo you want to add to cloud build

  * Step 2: Setup trigger type

    <img src="images/image-20220324121729100.png" alt="image-20220324121729100" style="zoom:50%;" />

    * Specify location of the `config.yaml` file for cloud build.

      <img src="images/image-20220324121833150.png" alt="image-20220324121833150" style="zoom:50%;" />

  * Done!

    <img src="images/image-20220324121901475.png" alt="image-20220324121901475" style="zoom:67%;" />

  * Test

    * push new code to github using a specific tag

      ![image-20220324122109009](images/image-20220324122109009.png)

    * Monitor the cloud build steps<img src="images/image-20220324122037048.png" alt="image-20220324122037048" style="zoom: 80%;" />

      * Here the first training code image is built and pushed.
      * Next, the base image of all the lightweight components are built and pushed.
      * Then a Kubeflow pipeline is compiled into a .yaml file
      * Finally the same kubeflow pipeline is uploaded to the cluster.

##  Course Summary

###  Course Summary

