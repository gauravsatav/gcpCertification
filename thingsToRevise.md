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