import pandas as pd
import numpy as np
import time

from collections import Counter
import matplotlib.pyplot as plt

# ======================================================================
# 0. Metrics Helper
# ======================================================================

def calculate_metrics(y_true, y_pred, labels):
    """Helper function to calculate accuracy and macro F1 score."""
    accuracy = np.mean(y_true == y_pred)
    f1_scores = []
    for cls in labels:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    macro_f1 = np.mean(f1_scores)
    return accuracy, macro_f1

# ======================================================================
# 1. Load Data
# This cell defines the function to load and scale our data.
# The scaling method (Standard Scaling) is taken from Knn_select.ipynb.
# ======================================================================

def load_and_scale_data(train_path='MNIST_train.csv', val_path='MNIST_validation.csv'):

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    label_col = 'label'
    feature_cols = [col for col in train_df.columns if col != label_col and col != 'even']

    X_train_raw = np.asarray(train_df[feature_cols]).astype(np.float64)
    y_train = np.asarray(train_df[label_col]).astype(np.int64)
    X_val_raw = np.asarray(val_df[feature_cols]).astype(np.float64)
    y_val = np.asarray(val_df[label_col]).astype(np.int64)
    labels = np.unique(y_train)
    
    # --- Apply Standard Scaling (from Knn_select.ipynb) ---
    print("Applying Standard Scaling (zero mean, unit variance)...")
    train_mean = np.mean(X_train_raw, axis=0)
    train_std = np.std(X_train_raw, axis=0)
    train_std_safe = train_std + 1e-10 # Avoid divide-by-zero
    
    X_train_scaled = (X_train_raw - train_mean) / train_std_safe
    X_val_scaled = (X_val_raw - train_mean) / train_std_safe
    print("Standard Scaling complete.")
    # --- End Scaling ---
    
    return X_train_scaled, y_train, X_val_scaled, y_val, labels

# ======================================================================
# 2. PCA Definition
# This cell defines the PCA_SVD class from your notebooks.
# ======================================================================

class PCA_SVD:
    """
    PCA implementation from scratch using SVD on the covariance matrix.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Data is already scaled, so mean is ~0, but we re-center for robustness.
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        U, S, Vh = np.linalg.svd(cov_matrix)
        self.components = U[:, :self.n_components].T
        return self

    def transform(self, X):
        X_centered = X - self.mean
        X_pca = np.dot(X_centered, self.components.T)
        return X_pca

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# ======================================================================
# 5. Define OvR (One-vs-Rest)
# This cell defines the helper functions for the OvR Logistic Regression classifier
# (from end_ovr.ipynb).
# ======================================================================

def sigmoid(z):
    """Numerically stable sigmoid function."""
    z = np.clip(z, -30, 30) # Clip to avoid overflow
    return 1.0 / (1.0 + np.exp(-z))

def fit_logistic_batch(X, y, epochs=100, lr=1.0, reg_lambda=0.01):
    """
    Trains a binary logistic regression model using batch gradient descent.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    for _ in range(epochs):
        z = X.dot(weights) + bias
        y_proba = sigmoid(z)
        dz = y_proba - y
        dw = (1/n_samples) * X.T.dot(dz) + (reg_lambda * weights)
        db = (1/n_samples) * np.sum(dz)
        weights -= lr * dw
        bias -= lr * db
        
    return {'w': weights, 'b': bias}

# ======================================================================
# 6. Show OvR Implementation
# This function will run the OvR training and evaluation.
# ======================================================================

def run_ovr_pipeline(X_train_pca, y_train, X_val_pca, y_val, labels):
    """
    Runs the OvR pipeline using the pre-computed PCA data.
    """
    print("\n" + "="*80)
    print("--- Running OvR Pipeline ---")
    start_time = time.time()
    
    # --- Hyperparameters from end_ovr.ipynb ---
    EPOCHS = 1000
    LEARNING_RATE = 0.3
    REG_LAMBDA = 0.03
    
    ovr_models = {}
    print(f"Training 10 OvR models (Epochs={EPOCHS}, LR={LEARNING_RATE})...")
    
    for cls in labels:
        y_train_binary = (y_train == cls).astype(int)
        model = fit_logistic_batch(X_train_pca, y_train_binary, 
                                     epochs=EPOCHS, lr=LEARNING_RATE, reg_lambda=REG_LAMBDA)
        ovr_models[cls] = model
    print("OvR training complete.")

    # Prediction
    val_probas_matrix = np.zeros((X_val_pca.shape[0], len(labels)))
    for i, cls in enumerate(labels):
        model = ovr_models[cls]
        z = X_val_pca.dot(model['w']) + model['b']
        val_probas_matrix[:, i] = sigmoid(z)
        
    y_val_pred = np.argmax(val_probas_matrix, axis=1)
    
    # Evaluate
    accuracy, macro_f1 = calculate_metrics(y_val, y_val_pred, labels)
    runtime = time.time() - start_time
    print(f"OvR Runtime: {runtime:.2f}s")
    
    return {'model': 'OvR LogReg', 'acc': accuracy, 'f1': macro_f1, 'runtime': runtime}

# ======================================================================
# 7. Define Softmax Regression
# This cell defines the SoftmaxRegression_scratch class (from end_sofmax.ipynb).
# ======================================================================

class SoftmaxRegression_scratch:
    """
    A from-scratch implementation of Softmax (Multinomial Logistic) Regression
    """
    def __init__(self, learning_rate=0.1, n_epochs=100, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.n_classes = -1
        self.n_features = -1

    def _one_hot_encode(self, y):
        y_one_hot = np.zeros((y.shape[0], self.n_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    def _softmax(self, z):
        z_stable = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_loss(self, X, y_one_hot):
        n_samples = X.shape[0]
        logits = X.dot(self.weights) + self.bias
        probas = self._softmax(logits)
        probas = np.clip(probas, 1e-15, 1 - 1e-15)
        data_loss = - (1 / n_samples) * np.sum(y_one_hot * np.log(probas))
        reg_loss = (self.reg_lambda / 2) * np.sum(self.weights * self.weights)
        return data_loss + reg_loss

    def fit(self, X, y):
        n_samples, self.n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        self.weights = np.random.randn(self.n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        y_one_hot = self._one_hot_encode(y)
        
        print(f"Starting Softmax Regression training for {self.n_epochs} epochs...")
        
        for i in range(self.n_epochs):
            logits = X.dot(self.weights) + self.bias
            probas = self._softmax(logits)
            grad_logits = (probas - y_one_hot) / n_samples
            dW = X.T.dot(grad_logits)
            dW += self.reg_lambda * self.weights 
            db = np.sum(grad_logits, axis=0, keepdims=True)
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            if (i + 1) % 100 == 0:
                loss = self._compute_loss(X, y_one_hot)
                print(f"  Epoch {i+1}/{self.n_epochs}, Loss: {loss:.4f}")
        print("Softmax training finished.")

    def predict_probas(self, X):
        logits = X.dot(self.weights) + self.bias
        return self._softmax(logits)

    def predict(self, X):
        probas = self.predict_probas(X)
        return np.argmax(probas, axis=1)

# ======================================================================
# 8. Show Softmax Implementation
# This function will run the Softmax training and evaluation.
# ======================================================================

def run_softmax_pipeline(X_train_pca, y_train, X_val_pca, y_val, labels):
    """
    Runs the Softmax Regression pipeline using the pre-computed PCA data.
    """
    print("\n" + "="*80)
    print("--- Running Softmax Regression Pipeline ---")
    start_time = time.time()

    # --- Hyperparameters from end_sofmax.ipynb ---
    EPOCHS = 500
    LEARNING_RATE = 0.5
    REGULARIZATION = 0.01
    
    softmax_model = SoftmaxRegression_scratch(
        learning_rate=LEARNING_RATE,
        n_epochs=EPOCHS,
        reg_lambda=REGULARIZATION
    )
    
    softmax_model.fit(X_train_pca, y_train)
    
    # Evaluate
    y_val_pred = softmax_model.predict(X_val_pca)
    accuracy, macro_f1 = calculate_metrics(y_val, y_val_pred, labels)
    runtime = time.time() - start_time

    print(f"Softmax Runtime: {runtime:.2f}s")
    return {'model': 'Softmax Reg', 'acc': accuracy, 'f1': macro_f1, 'runtime': runtime}

# ======================================================================
# 9. Define KNN (K-Nearest Neighbors)
# This cell defines the KNN_bruteforce_scratch class (from Knn_select.ipynb).
# ======================================================================

class KNN_bruteforce_scratch:
    """
    A from-scratch implementation of a brute-force K-Nearest Neighbors classifier.
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict_single(self, x_test):
        distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X_test):
        print(f"Predicting {len(X_test)} samples with KNN (k={self.k})... This may take a while.")
        y_pred = [self._predict_single(x) for x in X_test]
        return np.array(y_pred)

# ======================================================================
# 10. Show KNN Implementation
# This function will run the KNN evaluation.
# ======================================================================

def run_knn_pipeline(X_train_pca, y_train, X_val_pca, y_val, labels):
    """
    Runs the KNN pipeline using the pre-computed PCA data.
    """
    print("\n" + "="*80)
    print("--- Running KNN Pipeline ---")
    start_time = time.time()
    
    # --- Hyperparameters from Knn_select.ipynb ('accurate' preset) ---
    K_NEIGHBORS = 6
    
    knn_model = KNN_bruteforce_scratch(k=K_NEIGHBORS)
    knn_model.fit(X_train_pca, y_train)
    
    # Evaluate
    y_val_pred = knn_model.predict(X_val_pca)
    accuracy, macro_f1 = calculate_metrics(y_val, y_val_pred, labels)
    runtime = time.time() - start_time

    print(f"KNN Runtime: {runtime:.2f}s")
    return {'model': 'KNN', 'acc': accuracy, 'f1': macro_f1, 'runtime': runtime}

# ======================================================================
# 3. Apply PCA and 11. Show Comparison (Main Execution Block)
# ======================================================================

if __name__ == "__main__":
    
    # --- 3. Apply PCA and Store Data ---
    print("--- 1. Loading and Scaling Data ---")
    X_train_scaled, y_train, X_val_scaled, y_val, labels = load_and_scale_data()
    
    if X_train_scaled is not None:
        # --- Hyperparameter for PCA (used by all models) ---
        N_COMPONENTS = 54
        
        print(f"\n--- 2. Applying PCA (n_components={N_COMPONENTS}) ---")
        pca = PCA_SVD(n_components=N_COMPONENTS)
        
        # Fit on the *scaled* training data
        X_train_pca = pca.fit_transform(X_train_scaled)
        
        # Transform the *scaled* validation data
        X_val_pca = pca.transform(X_val_scaled)
        
        print(f"PCA complete. New feature shape: {X_train_pca.shape}")
        print("Data is ready for all classifiers.")

        # --- 4. Scale and Transform (This is just a marker for your step) ---
        print("\n--- 4. (Scaling & PCA transformation complete) ---")

        # --- 11. Show Comparison of All Models ---
        all_results = []
        
        # --- Run all three pipelines ---
        try:
            all_results.append(run_knn_pipeline(X_train_pca, y_train, X_val_pca, y_val, labels))
        except Exception as e:
            print(f"KNN pipeline failed: {e}")
            
        try:
            all_results.append(run_ovr_pipeline(X_train_pca, y_train, X_val_pca, y_val, labels))
        except Exception as e:
            print(f"OvR pipeline failed: {e}")

        try:
            all_results.append(run_softmax_pipeline(X_train_pca, y_train, X_val_pca, y_val, labels))
        except Exception as e:
            print(f"Softmax pipeline failed: {e}")

        # --- Create and display DataFrame ---
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.set_index('model')
            
            # Format for printing
            results_df_display = results_df.copy()
            results_df_display['acc'] = results_df_display['acc'].apply(lambda x: f"{x*100:.2f}%")
            results_df_display['f1'] = results_df_display['f1'].apply(lambda x: f"{x*100:.2f}%")
            results_df_display['runtime'] = results_df_display['runtime'].apply(lambda x: f"{x:.2f}s")

            print("\n" + "="*80)
            print("FINAL MODEL COMPARISON (on Validation Set)")
            print("="*80)
            print(results_df_display.to_string())
            
            # --- Create and display Plots ---
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
            fig.suptitle('Model Performance Comparison', fontsize=16)

            # F1 Score Plot
            results_df['f1'].plot(kind='bar', ax=axes[0], 
                                title='Macro F1-Score (Higher is Better)', 
                                grid=True, color=['blue', 'orange', 'green'])
            axes[0].set_ylabel('F1-Score')
            axes[0].set_xlabel('Model')
            axes[0].set_xticklabels(results_df.index, rotation=0)
            axes[0].set_ylim(bottom=max(0, results_df['f1'].min() * 0.95), top=max(1.0, results_df['f1'].max() * 1.05))

            # Runtime Plot
            results_df['runtime'].plot(kind='bar', ax=axes[1], 
                                     title='Total Runtime (Lower is Better)', 
                                     grid=True, color=['blue', 'orange', 'green'])
            axes[1].set_ylabel('Runtime (seconds)')
            axes[1].set_xlabel('Model')
            axes[1].set_xticklabels(results_df.index, rotation=0)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            print("\nDisplaying plots... (This may open a new window)")
            plt.show()
            
        else:
            print("No models ran successfully to compare.")

    else:
        print("\nError: Data was not loaded or PCA was not run.")
        print("Please ensure 'MNIST_train.csv' and 'MNIST_validation.csv' are present and run this script again.")