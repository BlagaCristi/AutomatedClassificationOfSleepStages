
## Project description
The team project was implemented during the course of Advanced Machine Learning at ETH Zurich. The final goal of the project was to perform automated sleep stage classification on mice EEG and EMG data. For this, each member of the team came up with one model that were afterwards aggregated using **Ensembling** in order to make the final prediction. The model that I have chosen consists of a deep-learning pipeline, as described in https://arxiv.org/abs/1809.08443

A key difference between this implementation and the original proposal is that we do not do artifact detection because our labels do not include artifacts. Therefore, the changes made to the paper implementation are the **changes in the loss function**. Instead of having a joint loss function (one for sleep, one for artifacts), we only have one (Negative log-likelihood).

The pipeline can be summarised as follows:
* *Preprocessing*:
	* every epoch was **low-pass filtered** with a *4th-order Butterworth filter (cutoff 25.6Hz)* using a forward-backwards scheme to ensure zero-phase filtering.
	*   Each epoch was **downsampled** from *128Hz to 64Hz*.  
	-   Each epoch was extended to the surrounding epochs in order to imitate the scoring procedure used to manually label data. Therefore, *5 timewise consecutive epochs are used to classify the middle epoch*.
* *Learning* done in three parts:
	* Preprocessing:
		-   Increase **depth channels** using *Conv1x1 to 64* channels.  
		-   Apply **BatchNormalization**.
	-   Feature extraction:
		-   *successive downsampling* of the multivariate timeseries into meaningful features.
		-   Implemented using **successive convolutional layers** (8 in total). Each layer is represented by a *Conv1x5* with no padding and dilation. For odd layers, the stride is 1 and, for even layers, the stride is 2.
		-   After each convolutional layer, we apply **ReLU** for *nonlinearity* and **BatchNormalization** to reduce *internal covariate shift*.
	-   Classifier:
		-   the output of the convolutional pipeline is **flattened** (with respect to each batch).
		-   the resulting vector is passed through a **fully connected layer** of size 80, followed by ReLU for nonlinearity.
		-   Finally, the result is passed through a **Softmax layer** for creating a probability distribution and a logarithm layer for implementation of the negative log-likelihood function.    

Regarding the learning process, the following has been done:
-   the model was trained using **batch learning**, with a batch size of *256*.
-   training done using **RmsProp** as an optimizer, having a smoothing factor of *0.99*.
-   cost function used is **negative log-likelihood**.
-   to prevent exploding gradients, we applied a **maximum-norm normalization** of gradients with *max_norm = 0.1*.
-   total of **20** training epochs, divided into 3 stages (improves generalization):
	-   first 5 epochs => **“warming up”** learning: the learning rate linearly increases from 0 to 0.00128 * batch_size.
	-   next 10 epochs => learning rate **constant**, set to 0.00128 * batch size.
	-   last 5 epochs => **“cooldown”** learning: learning rate linearly dicreases from 0.00128*batch_size to 0.

The final results of the project (after Ensembling) were satisfactory, placing us in the top **15%** amongst our peers.