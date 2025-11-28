
1-SampleStreetNetwork.ipynb
- Sample (randomly?) locations on the road network, roughly Xm (?) apart
- Download images from street view using Google Maps API
    - For each point, sample 4 images, facing in the 4 cardinal directions
- Save a pickle file containing 
2-CalculateEmbeddings.ipynb
- Create text embeddings for the categories we want to assign images to
- Create image embedding for each image
- Calculate similarity scores between each image and each embedding
3-AssessImageClassifications.ipynb
4-SummariseEmbeddings.ipynb
5-InvestigateEmbeddings_CorrelationWithIMD.ipynb
6-Test-models.ipynb
7-Test-models-FeatureImportance.ipynb

# Estimating Gentrification using Street View Images and Embeddings

This script (initially produced by ChatGPT) does the following (_this was my query_):
 - Read a spatial boundary file (that I will hard code)
 - Obtain the road network (from OSM?) for that area
 - Generate sample points on the road network roughly X meters apart
 - At each sample point, download the most recent street images for that location (either a single 360 degree view of a few smaller images). Use whichever API service is the most appropriate for obtaining the images. Importantly please record the date that the image was taken.
 - For each image, calculate an embedding using an appropriate foundation model (one that has been pre-trained to distinguish street environments specifically). Please use Hugging Face libraries.
 - If necessary, calculate the mean embedding for each point (is this the best way to calculate a single embedding for a point represented by multiple images?)
 - Now, for each sampled point there will be a dataframe with information about the point and its embedding. Read another polygon spatial data file, that I will provide, which contains area-level estimates of gentrification.
 - Use point-in-polygon to get the gentrification for each point.
 - Use cross-validation to train a couple of ML models (probaly random forest, linear regression and a neural network) to estimate gentrification from the embedding vectors
 - Choose the best model and parameter configuration and test this model on some held-out data.