
|[Home](../README.md)|[Course Page]()|
|---------------------|--------------|

## 3 Launching into Machine Learning

[TOC]


###  Introduction to Course


####  Intro to Course

###  Improve Data Quality and Exploratory Data Analysis

####  Introduction

####  Improving Data Quality

####  Exploratory Data Analysis
*  Classical Analysis: Problem => Data => Model => Analysis =>  Conclusions

*  EDA: Problem => Data => Analysis => Model => Conclusions

*  Baysian: Problem => Data => Model => Prior Distribution => Analysis => Conclusions

<img src="images/image-20220308132958718.png" alt="image-20220308132958718" style="zoom: 80%;" />

####  Exploratory Data Analysis Using Python and BigQuery

###  Practical ML

####  Supervised Learning
There are two types of problem in supervised learning

* classification 

* regression


####  Regression and Classification
* **<u>MEAN SQUARED ERROR</u>** : a hyperplane which is just the generalization of a line to get a continuous value for the label in regression problems we want to minimize the error between our predicted continuous value and the label's continuous value usually using mean squared error 

* **<u>DECISION BOUNDARY</u>** : in classification problems instead of trying to predict a continuous variable we are trying to create a decision boundary that separates the different classes

* **<u>CROSS ENTROPY</u>** : in classification problems we want to minimize the error or misclassification between our predicted class and the labels class this is done usually using cross entropy

* A raw continuous feature can be discretized into a categorical feature and the reverse process a categorical feature can be embedded into a continuous space

* Both of these problem types regression and classification can be thought of as prediction problems in contrast to unsupervised problems which are like description problems

* <img src="images/image-20220309123610248.png" alt="image-20220309123610248" style="zoom:50%;" />

* In the below chart 

  * Yellow - Classification
  * Green - Linear Regression
  * In Classification the points towards the boundary carry more weights to decide where the decision boundary is while in regression the decision boundary is decided by reducing the mean error

  ![image-20220309130210701](images/image-20220309130210701.png)

  * plotted in yellow is the output of a one-dimensional linear classifier logistic regression notice that it is very close to the linear regression's green line but not exactly because **regression models usually use mean squared error as their loss function whereas classification models tend to use cross entropy**

  * so what is the difference between the two without going into too much of the details just yet there is a quadratic penalty for mean squared error so it is essentially trying to minimize the euclidean distance between the actual label and the predicted label on the other hand 

  * **with classifications cross entropy the penalty is almost linear when the predicted probability is close to the actual label but as it gets farther away it becomes exponential when it gets close to predicting the opposite class of the label** therefore if you look closely at the plot the most likely reason the classification decision boundary line has a slightly more negative slope is so that some of those noisy red points red being the noisy distribution fall on the other side of the decision boundary and lose their high error contribution since they are so close to the line their error contribution would be small

    [10:40](javascript:;)for linear regression because not only is the error quadratic but there is no preference to be on one side of the line or the other for regression as long as the distance is as small as possible so as you can see this data set is a great fit for both linear regression and

    [10:58](javascript:;)linear classification unlike when we looked at the tip data set where it was only acceptable for linear regression and would be better for a non-linear classification


####  Introduction to Linear Regression

* LR was used to predict sweet pea sizes?

* <img src="images/image-20220309133643891.png" alt="image-20220309133643891" style="zoom:50%;" />

* <img src="images/image-20220309133659294.png" alt="image-20220309133659294" style="zoom:50%;" />

* <img src="images/image-20220309133732819.png" alt="image-20220309133732819" style="zoom:50%;" />

* Assumption: we are first assuming that the gram matrix x transpose x is non-singular meaning that all the columns of our feature matrix x are linearly independent

* Finding inverse of matrix also has a time complexity of O(n^3) using the naive algorithm but still doesn't get much better using fancier algos.

* **<u>Cholesky or QR Decomposition</u>** : the multiplication to create the gram matrix we might instead solve the normal equations using something called a cholesky or qr decomposition

* Since the above methods are computationally expensive we make use of gradient descent optimization algorithm which is one less expensive computationally in both time and memory 

*  In Gradient Descent we want to traverse the loss hypersurface searching for the global minimum in other words we hope to find the lowest value regardless of where we start on the hypersurface this can be done by finding the gradient of the loss function and multiplying that with a hyperparameter learning rate and then subtracting that value from the current weights this process iterates until convergence choosing the optimal.

* <img src="images/image-20220309134821711.png" alt="image-20220309134821711" style="zoom: 80%;" />

  ### History of ML: Neural Network

* Single Layer Perceptron

* The number of neurons in the layer controls the dimension so if you have two inputs and you have three neurons you are mapping the input 2d space to a 3d space

* it is the number of neurons per layer that determine how many dimensions of vector space you are in if i begin with three input features i am in the r3 vector space even if i have a hundred layers but with only three neurons each i will still be in r3 vector space.

* the activation function changes the basis of the vector space but doesn't add or subtract dimensions, think of them as simply rotations and stretches and squeezes they may be non-linear but you remain in the same vector space as before

* the loss function is your objective you are trying to minimize it is a scalar that uses its gradient to update the parameter weights of the model this only changes how much you rotate and stretch and squeeze not the number of dimensions

  * <img src="images/image-20220309135411362.png" alt="image-20220309135411362" style="zoom:50%;" />
  * <img src="images/image-20220309135507542.png" alt="image-20220309135507542" style="zoom:50%;" />
  * <img src="images/image-20220309144038651.png" alt="image-20220309144038651" style="zoom:80%;" />
  * Different Activation Functions<img src="images/image-20220309144222374.png" alt="image-20220309144222374" style="zoom:80%;" />
  * Dropout layers began being used to help with generalization which works like ensemble methods
  * Convolutional layers were added that reduce the computational and memory load due to their non-complete connectedness as well as being able to focus on local aspects for instance images rather than comparing unrelated things in an image

####  Introduction to Logistic Regression

####  Decision Trees 

* **<u>CART (Classification and Regression Tree) Algorithm</u>** : the algorithm tries to choose a feature and threshold pair that will produce the purest subsets when split. A common metric to use is the gini impurity but there is also entropy once it has found a good split it searches for another feature threshold pair and does so recursively
* Recursively creating these hyperplanes in a tree is analogous to layers of linear classifier nodes in a neural network
* To generalize the model there are some methods to regularize it such as setting the minimum number of samples per leaf node or you can also build the full tree and then prune unnecessary nodes to really get the most out of trees
* <img src="images/image-20220309145525564.png" alt="image-20220309145525564" style="zoom:67%;" />

####  Random Forests

* **<u>Ensemble Learning</u>** : A group of predictors is an ensemble which when combined in this way leads to ensemble learning the algorithm that performs this learning is an ensemble method
* <img src="images/image-20220309151240538.png" alt="image-20220309151240538" style="zoom:67%;" />
* Decision Tree for 
  * Classification : if this is classification there could be a majority vote across all trees which would then be the final output class
  * Regression : if it is regression it could be an aggregate of the values such as the mean max median etc
* To Generalize the model we need to random sample records or features
  * **<u>Bagging</u>** (bootstrap aggregating) : random sampling examples/rows with replacement
  * **<u>Pasting</u>** : random sampling examples/rows without replacement 
* Method of validation your error:
  * **<u>K-fold</u>** : validation using random holdouts 
  * **<u>Random Subspaces</u>** : are made when we sample from the features and
  * **<u>Random Patches</u>** :  if we random sample examples too is called random patches
* Boosting : we aggregate a number of weak learners to create a strong learner typically this is done by training each learner sequentially which tries to correct any issues the learner before it has had.
  * AdaBoost (Adaptive Boosting)
  * Gradient boosting
* Use your validation set to use early stopping so that we don't start overfitting
* **<u>Stacking</u>** :  where we can have meta learners learn what to do with the pictures of the ensemble which can in turn also be stacked into meta meta learners and so on
* <img src="images/image-20220309154026096.png" alt="image-20220309154026096" style="zoom:67%;" />

#### Kernel Methods (SVM)

* support vector machines which are maximum margin classifiers 
* core to an svm is a non-linear activation and a sigmoid output for maximum margins
* <img src="images/image-20220309154720354.png" alt="image-20220309154720354" style="zoom:50%;" />
* svm classifiers aim to maximize the margin between the two support vectors using a hinge loss function compared to logistic regression's minimization of cross-entropy
* what happens if the data is not linearly separable into the two classes
  * **<u>Kernel Transformation</u>** : apply a kernel transformation which maps the data from our input vector space to a vector space that now has features that can be linearly separated <img src="images/image-20220309155248802.png" alt="image-20220309155248802" style="zoom:50%;" />
  *  kernel transformation is similar to how an activation function in neural networks maps the input to the function to transform space 
  * **<u>the number of neurons in the layer controls the dimension so if you have two inputs and you have three neurons you are mapping the input 2d space to a 3d space</u>**
  * Type of kernel:
    * basic linear kernel 
    * Polynomial kernel and 
    * Gaussian radial basis function kernel (Gaussian RBF Kernel) : Input space is mapped to infinite dimensions.
* When should an svm be used instead of logistic regression?
  * kernelized svms tend to provide sparser solutions and thus have better scalability
  * svms perform better when there is a high number of dimensions and when the predictors nearly certainly predict the response
* <img src="images/image-20220309160049860.png" alt="image-20220309160049860" style="zoom:67%;" />

###  Optimization

####  Introduction Loss Functions

* RMSE - For regression

* Cross Entropy - For classification

   <img src="images/image-20220309231051752.png" alt="image-20220309231051752" style="zoom:80%;" />

####  Gradient descent

* <img src="images/image-20220309232333638.png" alt="image-20220309232333638" style="zoom:50%;" />
* **<u>MINI BATCH GRADIENT DESCENT</u>** : to reduce the data points that is used to calculate the derivative of the loss function seems most feasible. (Step 1 data points). 
  * Typical Batch Size is between 10 to 1000
  * Sampling is done uniformly
  * Mini Batch Size = Batch Size (confusingly! but this what is referred in tf)
* **<u>BATCH GRADIENT DESCENT</u>** is different from Mini Batch Gradient Descent :batch gradient descent the batch there refers to batch processing so batch grading descent computes the gradient on the entire data
* **<u>STOCHASTIC GRADIENT DESCENT</u>**: One training sample (example) is passed through the neural network at a time and the parameters (weights) of each layer are updated with the computed gradient.

####  Troubleshooting Loss Curves

####  ML Model Pitfalls

Every backprop step consists of

* Step 1: Calculate derivative at that point of loss curve: The time taken to do it is proportional to 
  * Number of parameters (In practice this can be 10 to 100 Million)
  * Number of data points we're putting in our loss function (1K to 10's of B) (This can be a sample of those data points.)

<img src="images/image-20220309233302564.png" alt="image-20220309233302564" style="zoom:67%;" />

* Take a step: Updating the parameter, happens just once in a loop and hence doesn't take much time.

* Check loss: (check loss step needed NOT be done at every pass)

  * number of data points in the set that we're using for measuring the loss. 
  * the complexity of our model
  * check loss step needed to NOT done at every pass and the reason for this is that most changes in the loss function are incremental so what can we change to improve training time 

* To decrease the training time, from there are 4 options available to us as show in the image below of which we choose:

  <img src="images/image-20220310000158853.png" alt="image-20220310000158853" style="zoom:50%;" />

  * **<u>MINI BATCH GRADIENT DESCENT</u>** : to reduce the data points that is used to calculate the derivative of the loss function seems most feasible. (Step 1 data points). 

    * Typical Batch Size is between 10 to 1000
    * Sampling is done uniformly
    * Mini Batch Size = Batch Size (confusingly! but this what is referred in tf)

  * **<u>BATCH GRADIENT DESCENT</u>** is different from Mini Batch Gradient Descent :batch gradient descent the batch there refers to batch processing so batch grading descent computes the gradient on the entire data

  * **<u>STOCHASTIC GRADIENT DESCENT</u>**: One training sample (example) is passed through the neural network at a time and the parameters (weights) of each layer are updated with the computed gradient.

    

* Frequency at which Step 3 is done

  * Time Based - once every hour
  * Step Based - like once every 1000 steps

  

####  LectureLoss Curve Troubleshooting 

####  Performance metrics

* Performance Metrics help us to reject models that have settled into inappropriate minima.

* There will always be a gap between the metrics we care about and the metrics that work well with gradient descent.

* the issue boils down to differentiability gradient descent makes incremental changes to our weights this in turn requires that we can differentiate the weights with respect to the loss piecewise functions however have gaps in their ranges and while tensorflow can differentiate them the resulting loss surface will have discontinuities that will make it much more challenging to traverse 

  <img src="images/image-20220310005809917.png" alt="image-20220310005809917" style="zoom:80%;" />

* Performance Matrices are much more connected to business goals

####  Confusion Matrix

* Precision 
* Recall
* AUC

* Summary of Optimization![image-20220310004829820](images/image-20220310004829820.png)

###  Generalization and Sampling

* Data Splits if you have lots of data and can afford to hold out a test dataset

  * While Training
    * training
    * validation

  * While Reporting and deciding to use in production
    * test

* Less data or cannot afford to hold out data

  * Use cross validation methods

####  When to Stop Model Training

![image-20220310011114083](images/image-20220310011114083.png)



####  Lecture Creating Repeatable Samples in BigQuery

####  Maintaining Consistency in Training with Repeatable Splitting

####  Explore and create ML datasets

####  Module Quiz

###  Summary


####  Course Summary

####  Course Quiz

###  Course Resources


####  Course Resources

####  All Course Readings
