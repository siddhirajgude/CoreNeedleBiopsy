                BCNB Clinical Data
                       │
                       ↓
             1. DATA UNDERSTANDING
                       │
                       ↓
              2. DATA QUALITY CHECK
                       │
                       ↓
             3. TARGET IDENTIFICATION
                       │
              ┌────────┴────────┐
              ↓                 ↓
       Diagnosis Target    Stage Target
              │                 │
              └────────┬────────┘
                       ↓
             4. DATA PREPROCESSING
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Numerical Data      Categorical Data
             ↓                   ↓
       Missing Values        Encoding
       Outliers              Cleaning
       Scaling
             └─────────┬─────────┘
                       ↓
              5. FEATURE ENGINEERING
                       ↓
              6. FEATURE SELECTION
                       ↓
              7. PATIENT-LEVEL SPLIT
                       ↓
              8. CLASS BALANCE CHECK
                       ↓
              9. MODEL TRAINING
                       ↓
       ┌────────┬────────┬────────┬────────┐
       ↓        ↓        ↓        ↓
     LogReg     SVM      RF      XGBoost
       └────────┴────────┴────────┴────────┘
                       ↓
             10. MODEL COMPARISON
                       ↓
              11. MODEL EVALUATION
                       ↓
             12. BEST MODEL
                       ↓
            13. PREDICTION SYSTEM
                       ↓
          14. EXPLAINABILITY (SHAP)
                       ↓
              FINAL RESULTS