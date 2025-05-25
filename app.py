import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Import your custom XGBoost model
from XGBoost import XGBoostClassifier

# Page configuration
st.set_page_config(page_title="Adult Income Prediction ML App", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the adult dataset"""
    try:
        df = pd.read_csv("adult.csv")
    except FileNotFoundError:
        st.error("Please ensure adult.csv is in the same directory as this script")
        return None, None, None, None
    
    # Store original categorical columns before preprocessing
    original_df = df.copy()
    
    # Preprocess following your existing approach
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace('?', df[col].mode()[0])
    
    # Drop education column as in your implementations
    df = df.drop(["education"], axis=1)
    original_df = original_df.drop(["education"], axis=1)
    
    # One-hot encode for model training
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.drop("income_<=50K", axis=1)
    df_encoded = df_encoded.fillna(df_encoded.mean())
    
    # Get feature names before converting to numpy
    feature_names = df_encoded.columns[:-1].tolist()
    
    # Prepare X and y for models
    X = df_encoded.iloc[:, :-1].values.astype('float64')
    y = df_encoded.iloc[:, -1].values.astype('float64')
    
    # Prepare original categorical features for web interface
    categorical_features = {}
    for col in original_df.select_dtypes(include='object').columns:
        if col != 'income':
            categorical_features[col] = sorted(original_df[col].unique())
    
    # Add numerical features
    numerical_features = {}
    for col in original_df.select_dtypes(include=['int64', 'float64']).columns:
        if col != 'income':
            numerical_features[col] = {
                'min': float(original_df[col].min()),
                'max': float(original_df[col].max()),
                'mean': float(original_df[col].mean())
            }
    
    feature_info = {
        'categorical': categorical_features,
        'numerical': numerical_features,
        'original_df': original_df
    }
    
    return X, y, feature_names, feature_info

def convert_user_input_to_encoded(user_inputs, feature_info, feature_names):
    """Convert user inputs back to one-hot encoded format"""
    # Create a sample row from user inputs
    sample_row = {}
    
    # Add categorical features
    for col, value in user_inputs.items():
        if col in feature_info['categorical']:
            sample_row[col] = value
        else:
            sample_row[col] = value
    
    # Create DataFrame from sample
    sample_df = pd.DataFrame([sample_row])
    
    # One-hot encode the sample (same process as training data)
    sample_encoded = pd.get_dummies(sample_df)
    
    # Ensure all columns from training are present
    encoded_input = np.zeros(len(feature_names))
    
    for i, feature in enumerate(feature_names):
        if feature in sample_encoded.columns:
            encoded_input[i] = sample_encoded[feature].iloc[0]
    
    return encoded_input.reshape(1, -1)

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model using sklearn"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), list(range(X_train.shape[1])))
        ],
        remainder='passthrough'
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=0
        ))
    ])
    
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    xgb = XGBoostClassifier()
    xgb.fit(
        X_train, y_train, X_val=X_val, y_val=y_val, 
        subsample_cols=1.0, min_child_weight=1, depth=6,
        min_leaf=5, learning_rate=0.05, boosting_rounds=50, 
        lambda_=1, gamma=0, early_stopping_rounds=10, random_state=0
    )
    return xgb

def train_random_forest(X_train, y_train):
    """Train Random Forest model using sklearn"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('passthrough', 'passthrough', list(range(X_train.shape[1])))
        ]
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=0,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), list(range(X_train.shape[1])))
        ],
        remainder='passthrough'
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=0
        ))
    ])
    
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """Train Support Vector Machine model"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), list(range(X_train.shape[1])))
        ],
        remainder='passthrough'
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=0
        ))
    ])
    
    model.fit(X_train, y_train)
    return model

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), list(range(X_train.shape[1])))
        ],
        remainder='passthrough'
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    """Train K-Nearest Neighbors model"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), list(range(X_train.shape[1])))
        ],
        remainder='passthrough'
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ))
    ])
    
    model.fit(X_train, y_train)
    return model

def get_model_metrics(X, y, test_size=0.3, random_state=42):
    """Train all models and return comprehensive metrics"""
    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=random_state
    )
    
    models = {}
    metrics = {}
    
    model_configs = {
        'Logistic Regression': lambda: train_logistic_regression(X_train, y_train),
        'Random Forest': lambda: train_random_forest(X_train, y_train),
        'XGBoost': lambda: train_xgboost(X_train, y_train, X_val, y_val),
        'Gradient Boosting': lambda: train_gradient_boosting(X_train, y_train),
        'SVM': lambda: train_svm(X_train, y_train),
        'Naive Bayes': lambda: train_naive_bayes(X_train, y_train),
        'K-Nearest Neighbors': lambda: train_knn(X_train, y_train)
    }
    
    for model_name, train_func in model_configs.items():
        with st.spinner(f"Training {model_name}..."):
            model = train_func()
            
            if model_name == "XGBoost":
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            metrics[model_name] = {
                'Accuracy': accuracy,
                'F1 Score': f1,
                'Precision': precision,
                'Recall': recall
            }
            models[model_name] = model
    
    return metrics, models

def main():
    st.title("🎯 Adult Income Prediction ML Application")
    st.markdown("Predict income levels using machine learning models trained on the Adult Census dataset")
    
    # Load data
    data_result = load_and_preprocess_data()
    if data_result[0] is None:
        return
    
    X, y, feature_names, feature_info = data_result
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    mode = st.sidebar.selectbox("Select Mode", ["🔮 Prediction Mode", "📊 Model Comparison Mode"])
    
    if mode == "🔮 Prediction Mode":
        prediction_mode(X, y, feature_names, feature_info)
    else:
        model_comparison_mode(X, y)

def prediction_mode(X, y, feature_names, feature_info):
    st.header("🔮 Prediction Mode")
    st.markdown("Adjust feature values to get income predictions from trained models")
    
    # Model selection
    st.subheader("Select Model")
    available_models = ["Logistic Regression", "XGBoost", "Random Forest", "Gradient Boosting", "SVM", "Naive Bayes", "K-Nearest Neighbors"]
    selected_model = st.selectbox("Choose a model for prediction:", available_models)
    
    # Train selected model for prediction
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    if selected_model not in st.session_state.trained_models:
        with st.spinner(f"Training {selected_model}..."):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
            
            if selected_model == "Logistic Regression":
                model = train_logistic_regression(X_train_split, y_train_split)
            elif selected_model == "XGBoost":
                model = train_xgboost(X_train_split, y_train_split, X_val, y_val)
            elif selected_model == "Random Forest":
                model = train_random_forest(X_train_split, y_train_split)
            elif selected_model == "Gradient Boosting":
                model = train_gradient_boosting(X_train_split, y_train_split)
            elif selected_model == "SVM":
                model = train_svm(X_train_split, y_train_split)
            elif selected_model == "Naive Bayes":
                model = train_naive_bayes(X_train_split, y_train_split)
            elif selected_model == "K-Nearest Neighbors":
                model = train_knn(X_train_split, y_train_split)
            
            st.session_state.trained_models[selected_model] = model
    
    model = st.session_state.trained_models[selected_model]
    
    # Feature input section
    st.subheader("🎛️ Adjust Feature Values")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    user_inputs = {}
    
    # Categorical features
    cat_features = list(feature_info['categorical'].keys())
    num_features = list(feature_info['numerical'].keys())
    
    all_features = cat_features + num_features
    
    for i, feature in enumerate(all_features):
        col_idx = i % 3
        current_col = [col1, col2, col3][col_idx]
        
        with current_col:
            if feature in feature_info['categorical']:
                # Categorical feature
                options = feature_info['categorical'][feature]
                value = st.selectbox(f"{feature.replace('.', ' ').title()}:", options, key=f"input_{feature}")
                user_inputs[feature] = value
            else:
                # Numerical feature
                min_val = feature_info['numerical'][feature]['min']
                max_val = feature_info['numerical'][feature]['max']
                mean_val = feature_info['numerical'][feature]['mean']
                value = st.slider(f"{feature.replace('.', ' ').title()}:", min_val, max_val, mean_val, key=f"input_{feature}")
                user_inputs[feature] = value
    
    # Make prediction
    if st.button("🎯 Make Prediction", type="primary"):
        try:
            # Convert user inputs to encoded format
            input_array = convert_user_input_to_encoded(user_inputs, feature_info, feature_names)
            
            if selected_model == "XGBoost":
                prediction = model.predict(input_array)[0]
                probability = model.predict_proba(input_array)[0]
                probabilities = [1-probability, probability]
            else:
                prediction = model.predict(input_array)[0]
                probabilities = model.predict_proba(input_array)[0]
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            result = "High Income (>$50K)" if prediction == 1 else "Low Income (≤$50K)"
            confidence = max(probabilities) * 100
            
            # Show prediction with confidence
            if prediction == 1:
                st.success(f"🎉 **Predicted Income: {result}**")
            else:
                st.info(f"📋 **Predicted Income: {result}**")
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Probability visualization
            prob_df = pd.DataFrame({
                'Income Level': ['≤$50K', '>$50K'],
                'Probability': probabilities
            })
            
            fig = px.bar(prob_df, x='Income Level', y='Probability', 
                        title="Prediction Probabilities",
                        color='Probability',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            input_df = pd.DataFrame([user_inputs]).T
            input_df.columns = ['Value']
            st.dataframe(input_df)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def model_comparison_mode(X, y):
    st.header("📊 Model Comparison Mode")
    st.markdown("Compare performance metrics of different models with adjustable train/validation splits")
    
    # Split configuration
    st.subheader("🎛️ Dataset Split Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 50, 30) / 100
        random_state = st.number_input("Random State", 0, 1000, 42)
    
    with col2:
        train_size = 1 - test_size
        st.metric("Train Size", f"{train_size:.1%}")
        st.metric("Test Size", f"{test_size:.1%}")
    
    # Train models and compare
    if st.button("🚀 Train Models & Compare Performance", type="primary"):
        with st.spinner("Training all models..."):
            metrics, models = get_model_metrics(X, y, test_size, random_state)
        
        st.subheader("🏆 Model Performance Results")
        
        # Create comprehensive results dataframe
        results_data = []
        for model_name, model_metrics in metrics.items():
            row = {'Model': model_name}
            row.update(model_metrics)
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Detailed Metrics", "🎯 Best Performers"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Sortable table
                sort_by = st.selectbox("Sort by:", ['F1 Score', 'Accuracy', 'Precision', 'Recall'])
                sorted_df = results_df.sort_values(sort_by, ascending=False)
                st.dataframe(sorted_df.round(4), use_container_width=True)
            
            with col2:
                # Bar chart for selected metric
                fig = px.bar(sorted_df, x='Model', y=sort_by,
                            title=f"{sort_by} Comparison",
                            color=sort_by,
                            color_continuous_scale='RdYlGn',
                            text=sort_by)
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(showlegend=False, height=500)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Radar chart for all metrics
            metrics_for_radar = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            
            fig = go.Figure()
            
            for _, row in results_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[metric] for metric in metrics_for_radar],
                    theta=metrics_for_radar,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Comparison (All Metrics)",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap
            heatmap_data = results_df.set_index('Model')[metrics_for_radar]
            fig_heatmap = px.imshow(heatmap_data.T, 
                                   title="Performance Heatmap",
                                   labels=dict(x="Model", y="Metric", color="Score"),
                                   color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab3:
            # Best performers for each metric
            st.subheader("🥇 Best Performers by Metric")
            
            for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
                best_model = results_df.loc[results_df[metric].idxmax()]
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{metric}:** {best_model['Model']}")
                with col2:
                    st.metric("Score", f"{best_model[metric]:.4f}")
                with col3:
                    if best_model[metric] > 0.8:
                        st.success("Excellent")
                    elif best_model[metric] > 0.7:
                        st.info("Good")
                    else:
                        st.warning("Fair")
        
        # Performance trend analysis
        st.subheader("📈 Performance Trend Analysis")
        if st.button("🔍 Analyze Performance Across Different Splits"):
            with st.spinner("Analyzing performance trends..."):
                splits = np.arange(0.2, 0.6, 0.05)
                trend_data = []
                
                progress_bar = st.progress(0)
                for i, split in enumerate(splits):
                    trend_metrics, _ = get_model_metrics(X, y, split, random_state)
                    for model_name, model_metrics in trend_metrics.items():
                        for metric_name, metric_value in model_metrics.items():
                            trend_data.append({
                                'Test Size': f"{split:.2f}",
                                'Model': model_name,
                                'Metric': metric_name,
                                'Score': metric_value
                            })
                    progress_bar.progress((i + 1) / len(splits))
                
                trend_df = pd.DataFrame(trend_data)
                
                # Line plots for each metric
                for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
                    metric_data = trend_df[trend_df['Metric'] == metric]
                    fig = px.line(metric_data, x='Test Size', y='Score', 
                                color='Model', markers=True,
                                title=f"{metric} vs Test Set Size",
                                line_shape='spline')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()