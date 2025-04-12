# Multi-Model-Classification
ML Model Comparison Dashboard on Reviews Dataset
Project Overview:

             https://drive.google.com/file/d/12J-onkgJEvqALVJAKklJzRVkZu88mGfA/view?usp=sharing
             
This project demonstrates a comparative analysis of six popular machine learning models using a synthetic text reviews dataset. The goal is to classify reviews (e.g., positive vs. negative) and evaluate each model's performance using various metrics such as accuracy, confusion matrix, and ROC AUC. All results are visualized interactively in dashboard-style plots for intuitive understanding.

#Key Features & Technologies


1.Trains and evaluates 6 models:

      Logistic Regression
      
      Naive Bayes
      
      Support Vector Machine (SVM)
      
      Random Forest
      
      XGBoost
      
      Neural Network (LSTM)

#Tracks training & validation accuracy/loss.

#Displays performance metrics:

    a.Accuracy
    
    b.Confusion Matrix
    
    c.ROC AUC Curve
    
    d.Classification Report

#Visual comparison in:

    Interactive plots using Plotly
    
    Heatmaps using Seaborn
    
    PowerBI
  
 #Technologies Used
  
          Python (3.x)
                  
          TensorFlow / Keras
                  
          Scikit-learn
                  
          XGBoost
                  
          Pandas, NumPy
                  
          Seaborn, Matplotlib
              
          Plotly Express

#Setup Instructions
1. Clone the Repository

        git clone https://github.com/your-username/your-project-name.git
        cd your-project-name
2. Install Required Libraries

        pip install -r requirements.txt
        (Include packages like tensorflow, scikit-learn, xgboost, matplotlib, plotly, etc.)

3. Prepare the Dataset
   
        Use a labeled review dataset (e.g., synthetic or real product reviews).
        
        Format: X_train, y_train, X_test, y_test.

        Preprocessing (tokenization, padding, etc.) is included in the code.

4. Run the Project
Run the main script to:

          python main.py
   
a.Train all models

b.Evaluate performance

c.Generate and display all plots


  #Output Overview
  
         Accuracy & Loss graphs over epochs (for NN)
                    
         Confusion matrices of all models
                    
         ROC AUC curves for all models
                    
         Final test accuracy comparison

#Classification reports in terminal

    Results exported to .csv
