# Enzyme Commission Classification using Protein Sequences: A Machine Learning Approach

## 1. Introduction

This project addresses the problem of automated enzyme classification based on protein sequences. The Enzyme Commission (EC) classification system is a hierarchical taxonomy that categorizes enzymes based on the chemical reactions they catalyze. Accurately predicting EC classes from protein sequences is critical for functional annotation, drug discovery, and biotechnology applications.

We employ a machine learning pipeline that combines multiple feature engineering strategies with ensemble methods to achieve robust multi-class classification. The primary objective is to predict enzyme functional classes from raw amino acid sequences using various representation methods and classifier combinations.

## 2. Dataset

The dataset utilized in this study is the SwissProt-EC dataset, which contains protein sequences with their corresponding EC class labels. The dataset is split into three disjoint subsets:

- **Training Set**: Protein sequences used to fit model parameters
- **Development Set**: Used for hyperparameter tuning and model selection
- **Test Set**: Reserved for final model evaluation

Preprocessing includes removal of sequences with missing target labels or empty sequences. Each protein sequence is represented as a string of amino acids from the standard 20-letter amino acid alphabet: ACDEFGHIKLMNPQRSTVWY. The EC classification is multi-class, with samples assigned to primary EC categories (main_ec labels).

## 3. Methodology

Our classification pipeline follows a systematic approach:

1. **Data Loading & Preprocessing**: CSV files are loaded and cleaned by removing records with missing sequences or EC labels
2. **Feature Extraction**: Raw amino acid sequences are transformed into numerical feature vectors
3. **Model Training**: Base learners are trained on extracted features
4. **Ensemble Aggregation**: Multiple models are combined using ensemble strategies
5. **Evaluation**: Performance metrics assess model quality on held-out test data

All base learners apply dimensionality reduction via PCA (95% variance retention) before training to reduce computational overhead and mitigate the curse of dimensionality.

## 4. Feature Engineering

Multiple sequence-based feature representations were implemented to capture different aspects of protein composition and structure:

### 4.1 Amino Acid Composition (AAC)
Counts individual amino acids within each sequence and normalizes by sequence length, producing a 20-dimensional feature vector representing global amino acid frequencies. A SelectKBest filter (k=10) identifies the most discriminative features using f-classif scoring.

### 4.2 Dipeptide Frequency (2-mer)
Extracts bigram character patterns from sequences, capturing local compositional information. With k=200 selected features, this representation reflects sequential ordering dependencies beyond simple composition. Normalized by the effective number of dipeptides (sequence length - 1).

### 4.3 Tripeptide Frequency (3-mer)
Extends dipeptide analysis to trigrams, providing higher-order contextual information. The top k=200 features are selected, capturing sequential patterns at a finer granularity. Normalization uses sequence length - 2.

### 4.4 SVD Embedding
Trigrams are vectorized and projected onto n_components=128 dimensions using Truncated Singular Value Decomposition. This unsupervised dimensionality reduction captures the underlying structure of k-mer distributions without label information.

Feature selection uses f_classif scoring to rank features by their discriminative power for the classification task. All features are converted to dense arrays and cast as float32 for compatibility with downstream classifiers.

## 5. Models

Three base learners were implemented, each with built-in class balancing and PCA preprocessing:

### 5.1 Random Forest
Implemented via scikit-learn's RandomForestClassifier with:
- n_estimators=100 decision trees
- max_depth=25 to control complexity
- min_samples_split=5 for leaf purity
- class_weight="balanced" to handle class imbalance
- 4 parallel jobs for efficiency

Provides robust predictions through bootstrap aggregation and feature randomness.

### 5.2 LightGBM
Gradient boosting classifier with:
- n_estimators=100
- learning_rate=0.05 for gradual optimization
- class_weight="balanced" for imbalanced data
- 4 parallel jobs
- Verbose suppression for cleaner output

Offers fast training and memory efficiency compared to traditional gradient boosting.

### 5.3 Support Vector Machine (SVM)
LinearSVC with probabilistic calibration:
- C=1.0 regularization strength
- Dual formulation with "auto" selection
- max_iter=2000 for convergence
- CalibratedClassifierCV wrapper enables probability estimates required by soft voting and stacking

All models apply PCA with 95% variance retention before fitting, ensuring consistent dimensionality across features of varying sizes.

## 6. Ensemble Methods

Three ensemble strategies combine base learner predictions:

### 6.1 Hard Voting
VotingClassifier with voting="hard": Each base model casts a vote, and the class receiving the most votes is selected. Simple and interpretable, resistant to outlier predictions.

### 6.2 Soft Voting
VotingClassifier with voting="soft": Predictions are averaged in probability space, weighted equally across base learners. Leverages confidence information from calibrated models for more nuanced decisions.

### 6.3 Stacking
StackingClassifier with:
- Three base estimators (RF, LightGBM, SVM)
- LogisticRegression as meta-classifier
- 3-fold cross-validation for robustness
- stack_method="predict_proba" uses predicted probabilities as meta-features
- passthrough=False prevents concatenation of original features

Stacking learns optimal combination weights through supervised meta-model training, potentially capturing complementary strengths of base learners.

## 7. Results

The pipeline evaluates performance using multiple metrics:

- **Accuracy**: Overall proportion of correct predictions
- **Macro-averaged Precision**: Mean precision across classes, unweighted
- **Macro-averaged Recall**: Mean recall across classes, unweighted
- **Macro-averaged F1-Score**: Harmonic mean of macro precision and recall
- **Matthews Correlation Coefficient (MCC)**: Balanced multi-class metric (-1 to +1 scale)

Results are computed on the held-out test set after training on combined train+dev splits where applicable. Each feature representation is evaluated independently across all six models (3 base + 3 ensemble), permitting comparison of feature quality and model synergy.

The current codebase is structured to log performance metrics via utility functions and systematically report results for each feature-model combination, facilitating identification of optimal configurations.

## 8. Limitations

Several limitations should be noted:

1. **Fixed Architecture**: Model hyperparameters are statically defined; systematic tuning via grid/random search is not performed
2. **PCA Variance Threshold**: The 95% threshold may discard discriminative information for some feature types
3. **Equal Ensemble Weights**: Hard and soft voting assign equal weight to all base learners regardless of individual performance
4. **Limited Feature Diversity**: Only sequence composition-based features are considered; structure-based or external embeddings (e.g., pre-trained transformers) are excluded
5. **Scalability**: All computation occurs in-memory; processing very large datasets may encounter memory constraints
6. **Missing Cross-validation**: Final evaluation relies on a single train-dev-test split rather than k-fold assessment

## 9. Conclusion

This project demonstrates a comprehensive machine learning pipeline for enzyme classification from protein sequences. By combining multiple feature engineering approaches with ensemble methods, we create a flexible framework capable of leveraging complementary information sources.

Key contributions include:
- Modular implementation enabling easy feature/model swapping
- Multiple ensemble strategies for robustness
- Class-balanced training to handle imbalanced EC distributions
- Systematic evaluation across diverse metrics

Future work should incorporate hyperparameter optimization, cross-validation assessment, and exploration of deep learning approaches (e.g., sequence transformers). Integration with structural features and transfer learning from large pre-trained models could further improve performance.

## 10. References

1. ENZYME database: The enzyme information system. https://enzyme.expasy.org/
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
3. Ke, G., et al. (2017). LightGBM: A Fast, Distributed, Gradient Boosting Framework. In Advances in Neural Information Processing Systems (pp. 3146-3154).
4. Vapnik, V. (1995). The Nature of Statistical Learning Theory. Springer-Verlag.
5. Wolpert, D. H. (1992). Stacked Generalization. Neural Networks, 5(2), 241-259.
6. The UniProt Consortium. (2021). UniProt: the universal protein knowledgebase in 2021. Nucleic Acids Research, 49(D1), D480-D489.
