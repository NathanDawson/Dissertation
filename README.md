# Performance of Feature Selection Methods in The Prediction of Knee Osteoarthritis Structural Progression 

### Introduction
Knee Osteoarthritis (OA) is a widespread degenerative joint disease with no known cure, affecting millions worldwide, 
significantly impacting quality of life through reduced mobility. Characterised by variable progression rates, it 
poses challenges in clinical trial patient selection, where the effectiveness of treatments is difficult to ascertain 
whether observed improvements are due to the investigational treatment or simply natural variations in disease 
progression.

### Problem Statement
The selection of patients for knee OA clinical trials is complex, necessitating a method to predict which patients are 
more likely to show progression. Current machine learning algorithms, while predictive, are limited by their complexity 
and the extensive feature sets they utilise, making interpretation challenging for clinical practitioners.

### Project Objective
This project aims to leverage machine learning with a focus on advanced feature selection techniques to develop a 
streamlined and interpretable model for predicting knee OA progression. By enhancing model interpretability and 
potentially improving predictive performance, this project seeks to ensure a more effective and efficient patient 
selection process for clinical trials.

### Rational
The need for healthcare practitioners to trust and understand the decision-making processes of predictive tools is 
paramount. Employing advanced feature selection techniques aims to develop a model that uses a minimal yet significant 
set of features, enhancing interpretability and integration into clinical judgment. This project emphasises 
interpretability alongside accuracy to improve the utility of predictive modeling in managing knee OA, thereby 
enriching patient care and accelerating the development of effective treatments.

### Methodology
- <b>Exploratory Data Analysis:</b> Understand data characteristics and distributions.
- <b>Data Pre-processing:</b> Prepare data for modeling through imputation and the removal of redundant features.
- <b>Feature Selection Techniques:</b> Implement five feature selection methods to identify significant predictors.
- <b>Model Development:</b> Utilise machine learning algorithms, focusing on Random Forest models, to predict knee OA 
structural progression.
- <b>Comparative Analysis:</b> Comparatively assess the 5 distinct models, analysing their performance as well as their 
interpretability through the number of features they each utilise.

### Results and Discussion
The results of this project revealed that all models struggled with high-dimensional data when predicting knee 
osteoarthritis progression, with the highest median F1 score being 0.34. Two of the most frequently selected features 
related to the MOAKS system, highlighting the importance of structural changes in the knee, while two other key features 
were associated with type-II collagen degradation. Despite identifying potentially predictive features, none of the 
models achieved an average F1 score above 0.5, indicating the need for further research to improve predictive 
performance.

### Directory Structure
The project is organised into four main directories: 'data', 'notebooks', 'scripts', and 'results'.<br>
<br><b>data</b><br>
This directory stores the original processed data, the merged processed data ('final' sub-directory), and the results.

<b>notebooks</b><br>
This directory contains Jupyter notebooks for exploratory data analysis of each dataset

<b>scripts</b><br>
This directory contains the main scripts for the project:
- 'eda.py': Loads and combines individual datasets.
- 'pipeline_setup.py': Creates the pipeline with all necessary steps.
- 'feature_selection.py': Takes user input to select a feature selection method for the pipeline.
- 'transformers.py': Contains custom scikit-learn transformers for data pre-processing and feature selection wrappers.
- 'model1.py': Runs the machine learning models.
- 'results_handler.py': Stores the results from the trained models and other model metadata.
- 'comparative_analysis.py': Analyses the model results comparatively, with the graphs saved in the <b>'results'</b> directory.

### Running the Models
To run the models using any of the five feature selection methods, execute the 'model1.py' script using the command line 
interface:<br>
`python scripts/model1.py` <br>
Simply follow the prompts provided in the interface to proceed with the desired feature selection method and model 
configuration.

### Contributions
This project is developed as a dissertation and represents a sole piece of work by myself. As such, external 
contributions are not permitted to maintain the integrity and individuality of the research. I appreciate your 
understanding and interest in this project.

### License
This project is licensed under the GNU GENERAL PUBLIC LICENSE - see the LICENSE file for details.

### Acknowledgments
I would like to thank my supervisor, Dr. Paweł Widera, for his invaluable guidance, support, and knowledge during my 
dissertation. His insights and feedback have played a key role in this research. I am also grateful to Newcastle 
University for offering an extensive educational experience and resource, which have been essential to my academic 
achievements. Finally, my heartfelt thanks go to my partner, Millicent, for her patience and support during my 
dissertation work, including the countless cups of coffee she provided. 