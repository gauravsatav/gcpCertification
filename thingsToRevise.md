3. ML Crash Course
2. My Notes
1. Blog post


7. Google Products Revision and architecture
5. Tensorflow Ecosystem
6. How google does ML
4. Sample Questions



-------------------Questions---------------------
* To mitigate the impact of the various data extraction overheads, the tf.data.Dataset.interleave transformation can be used to parallelize the data loading step,
  * Sequential interleave
  * Parallel interleave

* =New Question6= You work for a global footwear retailer and need to predict when an item will be out of stock based on historical inventory dat a. Customer behavior is highly dynamic since footwear demand is influenced by many different factors. You want to serve models that are trained on all available data, but track your performance on specific subsets of data before pushing to production. What is the most streamlined and reliable way to perform this validation? 
* A. Use the TFX Mode!Validator tools to specify performance metrics for production readiness 
* B. Use k-fold cross-validation as a validation strategy to ensure that your model is ready for production.
*  C. Use the last relevant week of data as a validation set to ensure that your model is performing accurately on current data.
*  D. Use the entire dataset and treat the area under the receiver operating characteristics curve (AUC ROC) as the main metric.
* ANS - (A)

-------------------------
* Does automl table require code or no code?

* Data fusion and big query relation with pyspark and sql for data transformation jobs

* loss function troubleshooting page (for oscillations)

* How to transform categorical column in big query for running bigquery ML jobs. is it dataprep?

* how many trainable parameters are there if layers are defined as they are in keras along with bias

* building sentiment analysis tools based on sentences/words/syntactical analysis to avoid gender age difference

* categorical entropy vs sparse categorical entropy. which one to use for multi-class classification of images and not multi-label

* Cloud DLP best practice for streaming data. and which other gcp products are used along with it. Do we require a placeholder first before seperating it as sensitive or not.

* AutoML Time series data with timecolumn

* Mean average precision in context of evaluating image classification model versions.

* Tensorflow Shapes of input while prediction, how does (-1,2) shaped tensor look.

* Insurance models in terms of dp/tracebility/explainability.

* In tensorflow (tensorflow I/O bottlenecks)
	* interleave
	* repeat parameters
	* what is buffer size in shuttle option

	* prefetch = training batch size
	* effect of decreasing batch size on I/O bound operatinos

* For high throughput online training prediction, does it include pub/sub  dataflow and having dataflow submit prediciton request to AI platform or do we use general cloud functions.

* Is there a link between different optimzation parameters and memory?
* Which parameters of Tensorflow Serving to improve performance? what does max_batch_size do in context of serving?
* What is tensorflow model-server-universal version

* A. TFProfiler
B. TF function
C. TF Trace
D. TF Debugger
E. TF Checkpoint
A. TensorFlow Hub
B. TensorFlow Probability
C. TensorFlow Enterprise
D. TensorFlow Statistics