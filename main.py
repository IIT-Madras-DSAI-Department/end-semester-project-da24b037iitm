import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
TRAIN_PATH = 'MNIST_train.csv'
VAL_PATH   = 'MNIST_validation.csv'

# Load CSVs and auto-detect label column
start_time=time.time()
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

label_candidates = ['label', 'digit', 'target']
label_col = None
for c in label_candidates:
    if c in train_df.columns:
        label_col = c
        break
if label_col is None:
    label_col = train_df.columns[0]

print('Detected label column:', label_col)
print('Train shape:', train_df.shape)
print('Validation shape:', val_df.shape)


# Prepare raw arrays
X_train_raw = train_df.drop(columns=[label_col]).values.astype(float)
y_train = train_df[label_col].values.astype(int)

X_val_raw = val_df.drop(columns=[label_col]).values.astype(float)
y_val = val_df[label_col].values.astype(int)

print('X_train_raw shape:', X_train_raw.shape)


# Choose mode: 'fast', 'accurate', 'another', or 'custom'
MODE = 'accurate'   # change this to 'fast' or 'accurate' or 'custom'


# Preset definitions
presets = {
    'fast': {
        'pca_n_components': 40,    # integer components
        'k_neighbors': 5,
        'weighting': 'distance',   # 'uniform' or 'distance' or 'gaussian'
        'whiten': False,
    },
    'accurate': {
        'pca_n_components': 55,
        'k_neighbors': 6,
        'weighting': 'gaussian',
        'whiten': False,
    },
    'another': {
        # the 'another' strong set suggested earlier
        'pca_n_components': 70,
        'k_neighbors': 7,
        'weighting': 'gaussian',
        'whiten': False,
    }
}

# If custom, you can edit these values
custom_params = {
    'pca_n_components': 100,    # int or float (0-1) for fraction of variance
    'k_neighbors': 5,
    'weighting': 'uniform',     # 'uniform', 'distance', or 'gaussian'
    'whiten': False,
}

if MODE in presets:
    params = presets[MODE].copy()
elif MODE == 'custom':
    params = custom_params.copy()
else:
    raise ValueError('MODE must be one of fast/accurate/another/custom')

print('Selected MODE =', MODE)
print('Params:')
for k,v in params.items():
    print(' ', k, ':', v)



# Standard scaling (fit on train)
def fit_standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return mean, std_safe

def transform_standard_scaler(X, mean, std):
    return (X - mean) / std 

scaler_mean, scaler_std = fit_standard_scaler(X_train_raw)
X_train_scaled = transform_standard_scaler(X_train_raw, scaler_mean, scaler_std)
X_val_scaled   = transform_standard_scaler(X_val_raw, scaler_mean, scaler_std)

print('Scaling done. shapes:', X_train_scaled.shape, X_val_scaled.shape)

# PCA via SVD function
def compute_pca_svd(X, n_components=0.95):
    # X should be scaled (approx zero mean). Returns (components, explained_ratio, k)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var_explained = (S**2) / (X.shape[0] - 1)
    total = var_explained.sum()
    explained_ratio = var_explained / total
    if isinstance(n_components, float) and 0 < n_components < 1:
        cum = np.cumsum(explained_ratio)
        k = int(np.searchsorted(cum, n_components) + 1)
    else:
        k = int(n_components)
        k = min(k, Vt.shape[0])
    components = Vt[:k, :]
    return components, explained_ratio, k

# compute components: if params['pca_n_components'] is float fraction or int
pca_spec = params['pca_n_components']
components, explained_ratio, k_selected = compute_pca_svd(X_train_scaled, n_components=pca_spec)
X_train_pca = X_train_scaled.dot(components.T)
X_val_pca   = X_val_scaled.dot(components.T)

print('PCA done. selected k =', k_selected)
print('Shapes after PCA:', X_train_pca.shape, X_val_pca.shape)

# optional: print cumulative explained variance for first components
cum = np.cumsum(explained_ratio)
print('Cumulative explained variance (first 10):', cum[:10])



def whiten_transform(X_pca, explained_ratio):
    # create an eigenvalue-like vector proportional to explained_ratio
    # scale by 1/sqrt(evr)
    eps = 1e-12
    scales = 1.0 / np.sqrt(explained_ratio[:X_pca.shape[1]] + eps)
    return X_pca * scales

if params.get('whiten', False):
    X_train_pca = whiten_transform(X_train_pca, explained_ratio)
    X_val_pca   = whiten_transform(X_val_pca, explained_ratio)
    print('Whitening applied.')
else:
    print('Whitening NOT applied.')




def gaussian_weights(dist2, gamma):
    return np.exp(-gamma * dist2)

def vote_labels(neigh_labels, neigh_weights):
    # returns label with max accumulated weight; ties -> smallest label
    uniq = {}
    for lab, w in zip(neigh_labels, neigh_weights):
        uniq[lab] = uniq.get(lab, 0.0) + w
    maxw = max(uniq.values())
    candidates = [lab for lab,w in uniq.items() if w == maxw]
    return int(min(candidates))

def knn_predict_batch(X_query, X_train_feat, y_train_labels, k=5, weight_mode='uniform', kernel_gamma=None, eps=1e-8):
    Xq = np.asarray(X_query)
    Xt = np.asarray(X_train_feat)
    # squared norms
    Xq_norm2 = np.sum(Xq**2, axis=1).reshape(-1,1)
    Xt_norm2 = np.sum(Xt**2, axis=1).reshape(1,-1)
    cross = Xq.dot(Xt.T)
    d2 = Xq_norm2 + Xt_norm2 - 2*cross
    d2 = np.maximum(d2, 0.0)
    m = d2.shape[0]
    kk = min(k, d2.shape[1])
    idx = np.argpartition(d2, kth=kk-1, axis=1)[:, :kk]
    preds = np.empty(m, dtype=y_train_labels.dtype)
    for i in range(m):
        inds = idx[i]
        inds_sorted = inds[np.argsort(d2[i, inds])]
        neigh_labels = y_train_labels[inds_sorted]
        neigh_d2 = d2[i, inds_sorted]
        if weight_mode == 'uniform':
            neigh_weights = np.ones_like(neigh_d2)
        elif weight_mode == 'distance' or weight_mode == 'inv_distance':
            neigh_weights = 1.0 / (np.sqrt(neigh_d2) + eps)
        elif weight_mode == 'gaussian':
            neigh_weights = gaussian_weights(neigh_d2, kernel_gamma)
        else:
            raise ValueError('unknown weight_mode')
        preds[i] = vote_labels(neigh_labels, neigh_weights)
    return preds



# compute heuristic gamma using small sample for speed
def compute_gamma_heuristic(Xtr_pca):
    nsamp = min(1000, Xtr_pca.shape[0])
    rng = np.random.RandomState(0)
    sidx = rng.choice(Xtr_pca.shape[0], size=nsamp, replace=False)
    sample = Xtr_pca[sidx]
    # use distances from first to others as quick heuristic
    diff = sample - sample[0:1]
    d2 = np.sum(diff**2, axis=1)
    med = max(1e-6, np.median(d2))
    return 0.5 / med

gamma_heur = None
if params['weighting'] == 'gaussian':
    gamma_heur = compute_gamma_heuristic(X_train_pca)
    print('Gaussian gamma heuristic:', gamma_heur)
else:
    print('No gaussian weighting selected.')



pca_k = k_selected
k_neighbors = int(params['k_neighbors'])
weighting = params['weighting']
whiten_flag = params.get('whiten', False)

print('Running evaluation with: pca_k=', pca_k, 'k=', k_neighbors, 'weighting=', weighting, 'whiten=', whiten_flag)

Xtr_feat = X_train_pca
Xv_feat  = X_val_pca
if whiten_flag:
    Xtr_feat = whiten_transform(Xtr_feat, explained_ratio)
    Xv_feat  = whiten_transform(Xv_feat, explained_ratio)

kernel_gamma = gamma_heur if weighting == 'gaussian' else None
weight_mode_arg = 'uniform' if weighting == 'uniform' else ('gaussian' if weighting == 'gaussian' else 'distance')

y_pred = knn_predict_batch(Xv_feat, Xtr_feat, y_train, k=k_neighbors, weight_mode=weight_mode_arg, kernel_gamma=kernel_gamma)
accuracy = float((y_pred == y_val).mean())
print(f'Validation accuracy (k={k_neighbors}): {accuracy:.4f}')

# confusion matrix
labels = sorted(np.unique(np.concatenate([y_train, y_val])))
label_to_idx = {lab:i for i, lab in enumerate(labels)}
cm = np.zeros((len(labels), len(labels)), dtype=int)
for t,p in zip(y_val, y_pred):
    cm[label_to_idx[t], label_to_idx[p]] += 1

print('\nConfusion matrix (rows=true, cols=pred):')
print(cm)
end_time=time.time()
print('time taken=',end_time-start_time)
# classification report
def classification_report_custom(y_true, y_pred, labels):
    rows = []
    for lab in labels:
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        sup = int((y_true == lab).sum())
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        rows.append((lab, prec, rec, f1, sup))
    return rows

report_rows = classification_report_custom(y_val, y_pred, labels)
print('\nClassification report:')
print('label\tprecision\trecall\tf1\tsupport')
for r in report_rows:
    print(f"{int(r[0])}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}\t{r[4]}")



# Plot confusion matrix
plt.figure(figsize=(7,5))
plt.imshow(cm, interpolation='nearest', aspect='auto')
plt.title(f'Confusion matrix (k={k_neighbors}, acc={accuracy:.4f})')
plt.colorbar()
ticks = np.arange(len(labels))
plt.xticks(ticks, labels)
plt.yticks(ticks, labels)
thresh = cm.max()/2 if cm.max()>0 else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(int(cm[i,j])), ha='center', va='center', color='white' if cm[i,j]>thresh else 'black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
