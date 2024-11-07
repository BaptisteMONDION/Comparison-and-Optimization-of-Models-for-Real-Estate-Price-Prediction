# Comparison-and-Optimization-of-Models-for-Real-Estate-Price-Prediction

This project aims to develop a pipeline for predicting real estate prices using various machine learning models, comparing their performance, and optimizing the best among them. The goal is to explore the performance of different regression algorithms through a rigorous methodology, select the most suitable models, and then focus on enhancing specific models for accuracy, stability, and noise resistance.

Objectives

1.	Model Comparison: Compare multiple regression models (Random Forest, SVM, LightGBM, XGBoost) using MSE to evaluate baseline performance on real estate pricing data.
2.	Optimization: Optimize the XGBoost and LightGBM models for minimal MSE, tailored to the specific needs of price prediction.
3.	Robustness Analysis: Assess the MSE variance and robustness of optimized models to noise for stable prediction.

Libraries and Resources Used

•	Python 3.11: Primary development language.
•	Pandas: Data manipulation and dataset management.
•	Numpy: Numerical calculations and array manipulation.
•	Scikit-learn: For basic regression models (Random Forest, SVM) and performance metrics calculation (MSE).
•	LightGBM: Optimized gradient boosting model for large datasets.
•	XGBoost: High-performance boosting model with advanced optimization capabilities.
•	Matplotlib: Visualization of model comparison results.
•	California Housing Dataset: Dataset used for house price prediction.

Project Structure

1. Data Preparation
•	Data Loading and Exploration: Load and explore the California Housing Dataset.
•	Training and Testing Split: Split data for model evaluation.

2. Model Comparison
•	Tested Models: Linear Regression, Ridge Regression, Decision Tree, Random Forest, Support Vector Regression.
•	Performance Evaluation: Calculate MSE for each model for direct precision comparison.
•	Training Time: Measure the training time of each model to evaluate their efficiency.

3. Selected Model Optimization
•	Hyperparameter Tuning of LightGBM and XGBoost: Improve model performance through hyperparameter adjustment for optimal MSE.
•	Comparison with Random Forest: Analyze the performance enhancement of LightGBM and XGBoost over the base Random Forest model.

4. Variance Analysis and Noise Resistance
•	MSE Variance: Analyze the stability of optimized models by examining MSE variation across multiple runs.
•	Noise Resistance Test: Add noise to test data to assess the robustness of optimized models (LightGBM and XGBoost) to data perturbations.

Key Files

•	comparaison_modeles.py: Program for model comparison with MSE evaluation and training time measurement.
•	optimisation_xgboost_lightgbm.py: Script for optimizing LightGBM and XGBoost models.
•	analyse_variance_robustesse.py: Code for MSE variance analysis and noise resistance testing.

This project provides a comprehensive analysis of prediction models, covering comparison, optimization, and robustness evaluation to identify the most accurate and stable model for real estate price forecasting.


