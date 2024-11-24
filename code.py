# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Initial Exploration
def load_and_explore_data(filepath):
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe().T)
    print("\nNull Values Check:")
    print(df.isnull().sum())
    return df

# 2. Statistical Analysis
def statistical_analysis(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    stats_results = {}
    
    for col in numerical_cols:
        stats_results[col] = {
            'skewness': stats.skew(df[col]),
            'kurtosis': stats.kurtosis(df[col]),
            'normality_test': stats.normaltest(df[col]),
            'descriptive_stats': df[col].describe()
        }
    
    print("\nStatistical Analysis Results:")
    for col, results in stats_results.items():
        print(f"\n{col}:")
        print(f"Skewness: {results['skewness']:.3f}")
        print(f"Kurtosis: {results['kurtosis']:.3f}")
        print(f"Normality test p-value: {results['normality_test'].pvalue:.3f}")
    
    return stats_results

# 3. Data Visualization
def create_visualizations(df):
    # Histograms
    df.hist(figsize=(15, 15))
    plt.tight_layout()
    plt.savefig('histograms.png')
    plt.close()
    
    # Box Plots
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    df.boxplot(column=numerical_cols)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('boxplots.png')
    plt.close()
    
    # Pie Charts
    categorical_cols = ['MTRANS', 'CAEC']
    for col in categorical_cols:
        plt.figure(figsize=(10, 8))
        df[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(f'Distribution of {col}')
        plt.savefig(f'{col}_pie.png')
        plt.close()
    
    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.savefig('correlation_matrix.png')
    plt.close()

# 4. Feature Importance Analysis
def feature_importance_analysis(X, y):
    # ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    anova_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_Score': selector.scores_,
        'P_value': selector.pvalues_
    })
    
    # Random Forest Feature Importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })
    
    print("\nFeature Importance Analysis:")
    print("\nTop 5 Features (ANOVA F-test):")
    print(anova_scores.sort_values('F_Score', ascending=False).head())
    print("\nTop 5 Features (Random Forest):")
    print(rf_importance.sort_values('Importance', ascending=False).head())
    
    return {'anova_scores': anova_scores, 'rf_importance': rf_importance}

# 5. Data Preprocessing
def preprocess_data(df):
    processed_df = df.copy()
    
    # Label Encoding
    le = LabelEncoder()
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        processed_df[col] = le.fit_transform(processed_df[col])
    
    # Split features and target
    X = processed_df.drop('NObeyesdad', axis=1)
    y = processed_df['NObeyesdad']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# 6. Normalization and Standardization
def normalize_and_standardize(X_train, X_test):
    # MinMax Normalization
    minmax = MinMaxScaler()
    X_train_minmax = minmax.fit_transform(X_train)
    X_test_minmax = minmax.transform(X_test)
    
    # Standardization
    standard = StandardScaler()
    X_train_standard = standard.fit_transform(X_train)
    X_test_standard = standard.transform(X_test)
    
    return (X_train_minmax, X_test_minmax), (X_train_standard, X_test_standard)

# 7. Model Implementation Functions
def implement_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate detailed report
    report = classification_report(y_test, y_pred)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    
    return accuracy, cm, report

# 8. Main execution function
def main():
    # Load data
    print("Loading and exploring data...")
    df = load_and_explore_data('obesity_dataset.csv')
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    stats_results = statistical_analysis(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    feature_importance = feature_importance_analysis(X_train, y_train)
    
    # Apply scaling
    print("\nApplying scaling...")
    minmax_data, standard_data = normalize_and_standardize(X_train, X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC()
    }
    
    # Train and evaluate models with both scaling methods
    scaling_methods = {
        'MinMax': minmax_data,
        'Standard': standard_data
    }
    
    for scaling_name, (X_train_scaled, X_test_scaled) in scaling_methods.items():
        print(f"\nResults with {scaling_name} scaling:")
        for model_name, model in models.items():
            accuracy, cm, report = implement_model(
                model, X_train_scaled, X_test_scaled, 
                y_train, y_test, f"{model_name}_{scaling_name}"
            )
            print(f"\n{model_name}:")
            print(f"Accuracy: {accuracy*100:.2f}%")
            print("Classification Report:")
            print(report)

if __name__ == "__main__":
    main()