# Street-Level Image Embeddings & Deprivation Analysis

Code for testing whether deprivation can be predicted using Google street view imagery. 

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
- Samples points along the road network at approximately 200 metres spacing
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
- For each semantic category, several prompts are defined
- One embedding is then derived per category, by averaging over the set of prompts


#### Image Embeddings  

- Loads each Street View image and extracts its 512 character CLIP image embedding  
- Computes similarity scores between the image embedding and all text embeddings  
- Converts similarity scores into softmax-normalised category "probabilities" (NB: these are not true probabilities)
- Stores per-image:  
     - 512 character embedding  
     - Category "probability" vector  

### 3. 3-AssessImageClassifications.ipynb

This notebook provides quality assessment of CLIP’s text-prompt classifications.

It includes:

- Manual/subjective inspection of matches  
- Assigning each image to the category with the highest score  
- Visualisations such as:  
    - Highest-scoring images for each category
    - Images with the least decisive probability distributions
    - Category-specific image grids

### 4. 4-SummariseEmbeddings.ipynb

Aggregates image-level outputs to the LSOA level.

This notebook:

- Computes the percentage of images in each category per LSOA  
- Generates mean, median, and max CLIP embeddings per LSOA  
- Repeats these calculations using only images belonging to each category  
- Produces a final LSOA-level dataframe containing:  
    - The overall mean/max/median LSOA embedding
    - Category proportions
    - The mean/max/median embedding for each of the categories

### 5. 5-InvestigateEmbeddings_CorrelationWithIMD.ipynb

Explores links between embeddings and IMD.

Includes:
- Correlation between each embedding dimension and IMD
- Comparison to expected correlations from random noise
- PCA decomposition of embeddings
- Visualisations of PC–IMD relationships

### 6. 6-Test-models.ipynb

Tests a range of predictive models linking image embeddings to IMD.

This notebook:
- Trains a random forest using an 80/20 train–test split
- Compares performance using:
    - Mean vs. median vs. max embeddings
    - Category percentages
    - Category-specific embeddings

- Implements full model fitting with:
    - Cross-validation
    - Hyperparameter tuning
    - Model comparison across feature sets

### 7. 7-Test-models-FeatureImportance.ipynb

Assesses which embedding dimensions or features contribute most to IMD predictions.  

Includes:  

- Random forest permutation importance  
- Gini importance  
- Visualisations of feature rankings  
