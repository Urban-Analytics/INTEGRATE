1-SampleStreetNetwork.ipynb
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
- Fit a random forest model and look at both permutation importance and Gini importance.  