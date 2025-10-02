# Presentation at GISScience 2025 in Christchurch

The Thirteenth International Conference on Geographic Information Science ([GIScience](https://giscience2025.org/)) was held in Christchurch, New Zealand, on 26-29 August 2025.

<ins>  Lex submitted the following paper: </ins> 

_Comber, A., Harris, R., & Brunsdon, C. (2025). Accommodating Space-Time Scaling Issues in GAM-Based Varying Coefficient Models.  In 13th International Conference on Geographic Information Science (GIScience 2025). Leibniz International Proceedings in Informatics (LIPIcs), Volume 346, pp. 15:1-15:9, Schloss Dagstuhl – Leibniz-Zentrum für Informatik, https://doi.org/10.4230/LIPIcs.GIScience.2025.15_

**Abstract**: The paper describes modifications to spatial and temporal varying coefficient (STVC) modelling, using Generalized Additive Models (GAMs). Previous work developed tools using Gaussian Process
(GP) thin plate splines parameterised with location and time variables, and has presented a spacetime toolkit in the stgam R package, providing wrapper functions to the mgcv R package. However,
whilst thin plate smooths with GP bases are acceptable for working with spatial problems they are not for working with space and time combined. A more robust approach is to use a tensor product
smooth with GP basis. However, these in turn require correlation function length scale or range parameters (ρ) to be defined. These are distances (in space or time) at which the correlation function
falls below some value, and can be used to indicate the scale of spatial and temporal dependencies between response and predictor variables (similar to geographically weighted bandwidths). The
paper describes the problem in detail, illustrates an approach for optimising ρ and methods for determining model specification.

The full conference paper is [available here]([https://github.com/Urban-Analytics/INTEGRATE/presentations/GISSCIENCE2025/LIPIcs.GIScience.2025.15.pdf)
