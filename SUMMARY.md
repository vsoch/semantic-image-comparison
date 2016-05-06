# Summary of Semantic Analyses

### 1) Reverse Inference
October 4, 2015

My first attempt was assessing reverse inference in a classification framework. I had two methods - calculating reverse inference using different ranges of thresholds (call this `ranges`) and using a single binary threshold (`binary`). I implemented these methods in pybraincompare, and ran the analyses. I had the following questions

######  Does including thresholds in the calculation give us more information?

I was first interested in comparing my "ranges" vs. "binary" method. The binary method is a lot faster and less complicated, so I want to know if they produce different results, at least for this dataset of 93 images. The answer was no. The distributions were as I would have expected - mostly very low scores, with a few scattered high scores. As a reminder - these were reverse inference scores for each of two strategies (calculations done considering the P(cognitive process | range in X) and P(cognitive process | above single threshold). Both approaches appeared to be capturing the same thing. This doesn't say anything about individual images - each is a reverse inference score for an entire concept node given all images defined / and not defined there.

###### Is there more evidence (higher RI scores) overall for particular concepts?
What I want to see is the main RI score for each node. If it's SUPER high then adding another image is probably not going to change it much (and this reflects solidity in the concept as defined by the set). It was not surprising that concepts with more images have higher scores, and as a result of this finding I did a separate analysis to assess the influence of number of images on the reverse inference scores. My conclusion was that we need about 20-25 to see the score plateau. The algorithm was as follows:

      For a single concept, C:
          Define "out set" as images not tagged with the concept (this set stays constant)
          Find all images for "in" set:
              Initialize "changing in" set to be the empty set
                  for timepoint, T, from 1...(N in "in" set)
                  Randomly select an image from "in", add to "in changing"
                  Calculate reverse inference score, save 
                  (Continue until all images are in set)


###### Do cognitive enrichment (CE) scores give evidence for cognitive concepts?
The final RI score for an entire concept node was what I had first looked at. This next part I wanted to look on the level of the images. The CE, or "cognitive enrichment" score, would be the bayes factor to compare:

      P(cognitive process for image I | spatial map of "in" node group) 
      vs
      P(cognitive process for image I | spatial map of "out" node group) 

My thinking was that we could determine the relative value of an image to contribute evidence for a cognitive process if it's RI score for the "in" group is relatively higher than it's RI score for the "out" group, as represented by the ratio above, what I am calling the CE "cognitive enrichment" score. I did a very basic algorithm of counting evidence "for" or "against" each concept by doing the following:

      for each node group:
           for each image labeled as "in" the group:
                +1 points for "for" if RI(image for "in") > RI(image for "out")
                +1 points for "against" if RI(image for "in" <= RI(image for "out")

           for each image labeled as "out" the group:
                +1 points for "for" if RI(image for "out") > RI(image for "in")
                +1 points for "against" if RI(image for "out" <= RI(image for "in")

I could then look at the tables, and (hope) to see evidence for concepts, but I did not see this evidence. Why? When I first looked at the overall numbers, I was disappointed, because evidence "against" is >> "for." The reason I speculated was related to the comparison set. There was no reason that a particular image should be better matched to a big, random group of images than a specific node. In fact, the opposite is even more likely, and so if I was considering evidence "for" and "against" a concept, I needed to limit this evidence to the images that I had asserted belong to the node. Let's try again. A better accuracy metric would be based on the question of If the "in" group images provide evidence for the concept, on the level of the node. Evidence FOR the concept should be based on images tagged WITH the concept, and including images NOT tagged with the concept seems like a different thing. So to assess the evidence "for" and "against" a concept, we should limit our subset to CE scores just for images defined at the node.

This approach told a totally different story. I saw a binary thing going on: The evidence was either for or against, and there was no split between the two. Unfortunately, I also saw an association between size of the "in" group and whether there is evidence "for" or "against" - for the larger groups, (eg, "Perception") the evidence is against. But again, we had the interesting cases where there was a pretty good sized set ("Reasoning and Decision Making" and "word recognition") and the evidence was "for!" I concluded that we probably couldn't make strong conclusions about the concepts that have less than 10 images tagged, but perhaps there is some.


### 2) Updated Reverse Inference
October 8, 2015

We then had a meeting, Sanmi + Russ + Chris + Vanessa. This was the meeting when Russ pointed out that we want the base probability (the prior) to be 0.5 (and not biased based on the N of tagged images with the concept in the database). Russ also suggested using more of a distance metric to calculate the p(activation|concept), and I updated the methods in pybraincompare, [adding this as a new method](https://github.com/vsoch/pybraincompare/blob/add/ontological-comparison/pybraincompare/ontology/inference.py#L342) `distance`. A summary of the changes is as follows:

- I used the abs(pearson) to get the p(activation | concept) and p(activation | ~ concept). I chose abs(pearson) because it gives us a score between 0 and 1 that can be inferred as a probability, a query image that is identical to the mean image for the set will get a score of 1, and then deviate from that as it gets more different. This decision also makes the statement that we are indifferent with regard to an image vs. its inverse, because it would be measuring the same thing but the contrast map calculated the other way around.

- In the calculation as well as the bayes factor (to determine if we believe evidence for the concept is much different after adding the image) I use 0.5 as the prior. I thought about this, and I think that for small image sets it would be unwise to say that we know anything about this. Thus, I made it a variable in the function, the default (equal_priors=True) is suggested for small data sets. It also makes sense in a classification framework, because we want to generate a "concept evidence" score for an image without the database we are using adding a HUGE bias.

Chris noted that

>> Traditionally square has been used to do the same thing. this has a more intuitive interpretation (r^2 == percentage of shared variance between images)

And Sanmi questioned needing a Bayes factor, and I was happy to remove it. This is also when Sanmi suggested some kind of multilabel confusion matrix. I liked both these ideas, and implemented the multilabel matrix, and updated the method.

[node accuracies](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/reverse_inference/result/accuracies_df_nodes.csv)
[image accuracies](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/reverse_inference/result/accuracies_mutlilabel_nodes.csv)

### 3) Janky ROC Curves
October 8, 2015

This is when I flipped two variables, and produced some ROC "curves" that looked like angry squiggles. I fixed the errors, and we had a nice set of ROC curves! Russ comments:

>> now that looks like an ROC!

Russ then suggested to look at AUC instead:

>> might be worth trying for all possible values and then plotting AUC as a function of cutoff...

And this led to the figure that was first present in the paper, that showed the AUC across the different concepts. 

[node aucs](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/reverse_inference/result/node_aucs_132.csv)
[ROCs](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/reverse_inference/result/roc_gr4.pdf)

### 4) Pairwise AUC Computation
Sanmi had an idea to do a pairwise AUC computation, as follows:

- Goal: Measure fraction of times when concept i would be confused for concept j with some fixed threshold
- Only evaluate on data subset containing only concept i or only concept j.

###### Definitions:
- dataset = Includes ground truth concepts and probability predictions
- concepts = list of concepts
- auc_matrix = matrix of size #concepts x # concepts: Main output

###### Pseudocode
      for i in concepts:
        for j in concepts and j != i
          data_subset = numpy.setxor1d( [all data in dataset where "i" in data.label], [all data in dataset where "j" in data.label])
          # see http://docs.scipy.org/doc/numpy/reference/generated/numpy.setxor1d.html#numpy.setxor1d

          # Compute label vector with "positive_labels" = i, "negative_labels" = j 
          labels = [(1 if "i" in data.label else -1) for data in data_subset ] # i.e. set as -1 if "j" in data.label
   
          # compute a score for each example. I suggest the likelihood ratio. Values >>1 indicate that i is much more likely than j
          # note that likelihood ratio can be much larger than 1. Most auc software can handle this.
          # Edit: modified to score only when p_i>p_j since these are the cases where "i" would be correctly predicted
          score_vector = [ (p_i/p_j if p_i > p_j, else 0.0) for data in data_subset]
    
          # COMPUTE AUC
          auc_matrix[i, j] = AUC(truth = labels, prediction = score_vector)

We generated the matrix, and didn't really know how to interpret it (we had 49 emails from 10/13-10/22) and the final conclusion by Sanmi was:

>> More importantly, I completely agree, there is not enough data to dig into pairwise predictions (@ least not in this way). I'll think a bit if some other strategy comes up.

And I agreed. At this point I wrote an email summarizing the plan to move forward (title is "Next steps for reverse inference classification work") but I don't see any responses to it. My guess is that we met and talked about things? After doing the new confusions, I also summarized our current status [in this update](https://sites.google.com/site/vanessasochat/updates-1/reverseinferenceforclassificationconfusionmatrices). On 10/22 I wrote up the work from that point for feedback (title of email is "Reverse Inference Update").


### 5) Tom Mitchell Leave Two Out
January 17, 2016

Russ and I met (I don't think Sanmi could make it) and we threw away the entire reverse inference framework, and Russ suggested trying the "tom mitchell leave-two-out trick to test classification." We emailed Sanmi for advice about how to generate a null, and I think we must have met soon after that, because on January 19th we had correspondence about the timing of different algorithms (e.g., ridge regression vs. elastic net). The reason for the discussion of time/computation is because we needed to generate these null distributions, and it was clear that we needed to use a larger voxel size, and test the speed of the algorithms. It still took almost 2 weeks to generate all the nulls, and everyone was mad at me for using up the nodes on Sherlock (email title is "Classification Timing Test") :)

After this, it looks like a few weeks / month was taken to give feedback on the paper draft, and let the null distribution generate.


### 6) NeuroSynth Decoding Model
Russ asked me to run the encoding model analysis on the neurosynth data (using mentions of cognitive atlas terms rather than hand-coded terms). We used a leave 1% of abstracts hold-out approach.  I generated a complete mapping (no parameter selection, no ontology weighting, and no holdout) for all 11K+ pmids and 614 features (Cognitive Atlas terms that I searched for in the abstracts). This meant that I generated a regression parameter map for each feature. The [maps can be seen here](http://neurovault.org/collections/1257/). They looked crappy to me. 

We also looked at 

[script](https://github.com/vsoch/semantic-image-comparison/blob/master/preprocess/5.encoding_with_neurosynth.py)

### 7) Filtered NeuroSynth Decoding Model
Next we put this in a classification framework, and decided on a "hold out 1% of abstracts" strategy. We also filtered down to cognitive concepts that were present in at least 10 abstracts (N=399).

>> how about an extended version of the leave-two-out strategy, in which you train the model while leaving out say 1% of abstracts (which would be ~100 per holdout set), generate the predicted images for those left out abstracts, then randomly pair them and test whether the similarity is higher between the true and predicted for each pair? this would only require 100 training rounds but you would still get to assess accuracy on all of the abstracts. if you wanted to go even further you could test all possible pairs within each holdout set (which would be 100*99/2 pairs) - I assume that the testing is fast so this still should not take terribly long, and would give us much richer and more interesting confusion data.

The mean accuracy was ballpark around ~0.68 (worse than when we used NeuroVault images).

[script](https://github.com/vsoch/semantic-image-comparison/blob/master/preprocess/5.classification_neurosynth.py)

### 8) NeuroSynth Regparameter Map Decoding
We wanted to see if the regparam maps produced by the above approach could be decoded with NeuroSynth to produce labels that we would expect. Russ looked over [the top ten table](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/decode/data/concept_regparam_decoding_named_topten.tsv) and it was decided that the results weren't very good.

[script](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/decode/decode_classification_regparam_maps.py)

### 9) Recalculation of Null
February 11, 2016

I had to redo the null distributions because I didn't save enough data to make a confusion matrix. I don't remember if this turned out to be needed or not, but it took another 2 weeks.


### 10) RSA to compare spatial to semantic
March 5, 2016
I met with Russ and he drew on the board how to do RSA, and I calculated it for the (Wang) score analysis.

### 11) Naive Bayes Decoding, Round 1
March 4-6, 2016

This was the first go at using Naive Bayes to do decoding. 
[script](https://github.com/vsoch/semantic-image-comparison/blob/master/preprocess/4.naive_bayes_prediction.py)

I presented concept and image accuracy matrices, with accuracy calculated as follows:

      for concept in concepts:
          Yp = predictions.loc[:,concept]  # This is a vector of predictions for all images across one concept, like [0,0,0,1...0]
          Ya = Ymat.loc[Yp.index,concept].tolist()  # This is a vector of actual labels, for all images (Yp.index) across the same concept 
          Yp = Yp.tolist() # This doesn't matter
          acc = numpy.sum([1 for x in range(len(Yp)) if Yp[x]==Ya[x]]) / float(len(Yp))  # This creates a vector of 1s for each case that the predicted == actual, then I sum the 1's and divide by the total number of predictions (correct / total) 
          concept_acc.loc[concept,"accuracy"] = acc # save in the data frame

And this is when Russ pointed me to calculate the false alarm / hit /a-prime etc, and this was the massive email chain (N=74!) that resulted in two things: 1) wanting to see Naive Bayes in different contexts, compared to the base, and 2) deciding to remove it from my thesis.

### 12-14) Naive Bayes Decoding, Round 2
March 12, 2016

This included 3 different analyses to try, all different renditions of Naive Bayes. The [summary document between Sanmi and I is here](https://docs.google.com/document/d/1IxOudKEcR-dy5Kqcnt8TCEZGhux-b6VtJr81dsb1SCw/edit?ts=56e4b953#heading=h.bcrq1mk6hjgf). Broadly it includes:

- [standard naive bayes decoder](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/prediction/prediction_concept_accs_base.tsv): This was first building a base model, just naive bayes, predicting binary image concept labels.
- [naive bayes decoder using forward model](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/prediction/prediction_concept_accs_forward.tsv): This was integrating information from the forward model, and comparing to the base.
- [naive bayes decoder using forward model with tuning](https://github.com/vsoch/semantic-image-comparison/blob/master/analysis/prediction/prediction_concept_accs_tuned.tsv): This was tuning the model to find an optimal threshold, c, between 0.001 and 10 (with breaks of 10), and comparing this tuned to the base.

[code](https://github.com/vsoch/semantic-image-comparison/blob/master/preprocess/4.naive_bayes_decoding.py)

I had made a mistake with regard to the threshold values of c for the tuning, and this was fixed. Sanmi noted that we might not want to optimize for accuracy (see email "Naive Bayes Update") and we decided to meet when Russ got back to town.

### 15) Naive Bayes Decoder (Model 4)
After the meeting, we came up with Model 4, which is summarized as follows:

      # Ypred = Ytest x W
      # 1 x 132 = 1 x 28k by 28k x 132
      # Y{test,:} = numpy.linalg.lstsq(W.T, X_{test,:}.T)[0].T

[script](https://github.com/vsoch/semantic-image-comparison/blob/master/preprocess/4.naive_bayes_decoding.py#L394)

We also talked about an approach to:

>> @russpold seems that one simple approach would be to threshold the predicted concept vector to have the same density as the actual concept vector (e.g., if there are 5 concepts in the actual vector, then set the top 5 concepts in the predicted vector to 1 and the rest to zero).  then you could just ask what proportion overlap there is.

And this is the strategy used to calculate accuracy. However, this was done without any kind of cross validation, and Russ decided he wanted to look at the performance to assess if the CV was worth it. Russ first asked for the average accuracy across all images.  It was 51.8%. We didn't do cross validation.

### 16) Pubmed Cogat
March 18, 2016

I don't remember what this was for, but I searched the entirety of pubmed for the Cognitive Atlas terms, and generated counts, etc.

[script](https://github.com/vsoch/semantic-image-comparison/blob/master/preprocess/6.pubmed_cogat.py)


### 17) Model Visualization
April 5, 2016
I made two visualizations for the decoding models:

- [Cognitive Concepts Associated with a Region](http://vsoch.github.io/cogat_voxel/)
- [Cognitive Concepts Associated with a Region](http://vsoch.github.io/cogatvoxel/)

Descriptions are provided with the links above.
