# Analyses

1. without_expansion: The same elastic net as before, but only including concept labels that contrasts are directly tagged with (e.g., we don't "walk up tree" to find parent concepts).
2. cca_explore: Latent space modeling exploration.

# Notes

Below are my thoughts on feedback from the meeting and potential next steps. This will mostly focus on modelling. I imagine we will discuss the data and perhaps some overall direction on the other thread.

Preprocessing + additional analysis:
1. Joke had a great point on the effects of sample size on z-stats. As mentioned in the meeting, the easy fix is to standardize each brain image before fitting. The more complicated fix is to extract the sample size from the papers and do some correction. I think standardization is the better approach.

2. Latent space modeling: I like PLS/CCA for trying to examine if there is a low dimensional space that explains variation jointly over labels and images well enough (http://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition). We can also use the fit PLS model directly for encoding / decoding. Ideally, one could make the CCA a plug-in replacement to the current encoding model, and run all the same analyses we are running for the same encoding / decoding models - in order to investigate if the latent space modeling helps.

Forward modeling:
1. We should have a measure of encoding regression performance (in addition to classification). I like out of sample R^2 i.e. how much variance is explained by the model using only cognitive concepts.

2. We should consider standardizing the cognitive concepts before fitting the encoding model (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html, so one can apply the same standardization when predicting test images.).

Reverse Modeling:
We did not talk much about this in the meeting. however:
1. I like the forward model based decoder as a first step. It keeps everything to a single model which simplifies further comparisons and further investigation. In practice, this kind of approach seems to work pretty well (it work on the same principle as naive Bayes):
  a) we can use the correlation based decoding used by neurovault. 
  b) alternatively, I suspect the model fitting approach we discussed a while ago would be a bit more accurate i.e. where we fit the labels given the beta maps. I also realize that this is pretty much the same as the dictionary learning approach I suggested in the other email, because you get weights for every cognitive process based on the fit model. The main difference is that the dictionary modeling approach tries to enforce sparsity on the cognitive process loadings. Either way is reasonable.

2. Threshold selection is still a messy part of multilabel classification when one does not have a preferred metric to optimize e.g. top K error, Hamming error, F-measure. If there are no strong preferences, Hamming error is a good default i.e. pick a threshold for each label separately to optimize label-wise accuracy. If there is a preference, then we should pick threshold that optimize that preferred metric, then we can also see how the selected threshold performs for other metrics. If we want to avoid selecting a threshold, then 1/2 is a good default (if we used 0/1 encoding, analogously threshold at zero if we used +1/-1 encoding).

3. In addition, we should measure ranking loss for each label separately, which avoids thresholding. There is also the ranking loss across labels, which is likely more relevant for prediction (http://scikit-learn.org/stable/modules/model_evaluation.html#ranking-loss)

4. If the forward modelling fails, I suggest a simple per label classifier using logistic regression, or better using Random Forests (seems to work better in practice for many problems)
