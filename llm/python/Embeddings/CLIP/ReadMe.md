<!-- 1-SampleStreetNetwork.ipynb
- Sample (randomly?) locations on the road network, roughly Xm (?) apart
- Download images from street view using Google Maps API
    - For each point, sample 4 images, facing in the 4 cardinal directions
- Save a pickle file containing...

2-CalculateEmbeddings.ipynb 
- Create text embeddings for the categories we want to assign images to
- Create image embedding for each image
- Calculate similarity scores between each image and each embedding

3-AssessImageClassifications.ipynb  
- (Subjective) assessment of how good the matching of images to textual descriptions is  
- This includes assigning images to their highest-soring category  
- Plots of:  
    - The top-scoring images for particular categories  
    - The images with the least conclusive scoring  
    - The images classified within each of the category  

4-SummariseEmbeddings.ipynb  
- Calculate the % of images in each LSOA in each of the categories  
- Find mean/median/max embedding in each LSOA  
- Find mean/median/max embedding in each LSOA, using just images in each of the categories  
- Create a dataframe, one row per LSOA, containing all of this information  

5-InvestigateEmbeddings_CorrelationWithIMD.ipynb  
- Look at the correlation of each the embedding dimensions on its own with the IMD.  
- Compare this to the distribution of correlations for an embedding made up on random numbers.    
- Perform a PCA on the embedding and plot the relationship between the two and the IMD.  

6-Test-models.ipynb  
- Test performance of a random forest model with a 80-20% train-test split.
    - Compare performance with the median vs. max vs. mean embedding.    
    - Compare performance with just the median embedding, just the percentages in each category or both.
    - Compare performance using just images within each of the categories.  
- Fit the model properly:  
    - Fit several different models with cross validation and hyper-parameter tuning, in order to work out the best model and parameters.  

7-Test-models-FeatureImportance.ipynb  
- Fit a random forest model and look at both permutation importance and Gini importance.   -->

# Street-Level Image Embeddings & Deprivation Analysis

A pipeline for sampling street-view imagery, extracting CLIP embeddings, classifying scenes, and linking visual patterns to socio-economic indicators.

## Overview

This repository provides an end-to-end workflow for:

- Sampling points across a road network  
- Downloading Google Street View imagery  
- Generating CLIP image embeddings  
- Classifying images into semantic categories using text prompts  
- Aggregating embeddings and classifications at the neighbourhood (LSOA) level  
- Exploring relationships between visual indicators and the Index of Multiple Deprivation (IMD)  

The analysis is implemented across a set of Jupyter notebooks, described below.

### 1. 1-SampleStreetNetwork.ipynb  

This notebook:
- Samples points along the road network at approximately X metres spacing
- Downloads images from Google Street View via the Maps API
- Captures four images per location, facing the four cardinal directions
- Saves a pickled dataset containing:
    - Point coordinates
    - Image metadata
    - Filepaths to all downloaded images

### 2. 2-CalculateEmbeddings.ipynb  

This notebook performs all embedding calculations.

#### Text Embeddings  

- Creates CLIP text embeddings for the user-defined semantic categories  
- Produces one embedding per category prompt  

#### Image Embeddings  

- Loads each Street View image and extracts its 512-D CLIP image embedding  
- Computes similarity scores between the image embedding and all text embeddings  
- Converts similarity scores into softmax-normalised category probabilities  
- Stores per-image:  
     - 512-D embedding  
     - Category probability vector  

3. 3-AssessImageClassifications.ipynb

This notebook provides quality assessment of CLIP’s text-prompt classifications.

It includes:

Manual/subjective inspection of matches

Assigning each image to the category with the highest probability

Visualisations such as:

Highest-scoring images for each category

Images with the least decisive probability distributions

Category-specific image grids

4. 4-SummariseEmbeddings.ipynb

Aggregates image-level outputs to the LSOA level.

This notebook:

Computes the percentage of images in each category per LSOA

Generates mean, median, and max CLIP embeddings per LSOA

Repeats these calculations using only images belonging to each category

Produces a final LSOA-level dataframe containing:

Embedding summaries

Category proportions

Image counts

Additional metadata for downstream modelling

5. 5-InvestigateEmbeddings_CorrelationWithIMD.ipynb

Explores links between visual features and socio-economic deprivation (IMD).

Includes:

Correlation between each embedding dimension and IMD

Comparison to expected correlations from random noise

PCA decomposition of embeddings

Visualisations of PC–IMD relationships

6. 6-Test-models.ipynb

Tests a range of predictive models linking image-derived features to IMD.

This notebook:

Trains a random forest using an 80/20 train–test split

Compares performance using:

Mean vs. median vs. max embeddings

Category percentages

Category-specific embeddings

Implements full model fitting with:

Cross-validation

Hyperparameter tuning

Model comparison across feature sets

7. 7-Test-models-FeatureImportance.ipynb

Assesses which embedding dimensions or features contribute most to IMD predictions.

Includes:

Random forest permutation importance

Gini importance

Visualisations of feature rankings