# train_xgboost.py -- XGBoost fault classifier (6 classes, 22 inputs) with SHAP explanations
# Trained only on the library of 90 generic PV systems; tested on Systems 1-3 (never used in training).
import glob, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import shap
import pv_simulator_v3 as P            # simulator: datasets, 22-input feature builder

LABELS = P.CLASSES                       # normal, partial shading, dirt, open circuit, short circuit, degradation
K, K_SHOTS = len(LABELS), 5
T = {1: pd.read_csv('data_v6/S1_6class_1002.csv'),      # test systems (never used for training)
     2: pd.read_csv('data_v6/S2_6class_402.csv'),
     3: pd.read_csv('data_v6/S3_6class_402.csv')}
LIB = pd.read_csv(sorted(glob.glob('data_v6/library_6class_*.csv'))[0])   # 90 generic systems, 5940 readings

# 1. Hyper-parameters chosen by cross-validation on the library only; whole systems kept together (GroupKFold)
SPACE = {'n_estimators': [200, 400, 600], 'max_depth': [3, 4, 6], 'learning_rate': [0.05, 0.1, 0.2],
         'subsample': [0.7, 0.85, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'min_child_weight': [1, 3, 5]}
X0, y0 = P.inputs(LIB), LIB.Fault
search = RandomizedSearchCV(XGBClassifier(random_state=0, n_jobs=-1), SPACE, n_iter=15, cv=GroupKFold(5), random_state=0)
search.fit(X0, y0, groups=LIB.System); PARAMS = search.best_params_
print('Settings:', PARAMS, f'CV accuracy on unseen library systems: {search.best_score_:.3f}')
source = XGBClassifier(random_state=0, n_jobs=-1, **PARAMS).fit(X0, y0)

# 2. Commissioning: add 5 labelled readings per class of the new system (weight 5), test on its other readings
def commission(d, draws=5):
    acc, cm = [], np.zeros((K, K), int)
    for draw in range(draws):
        shots = d.groupby('Fault', group_keys=False).sample(K_SHOTS, random_state=500 + draw)
        rest = d.drop(shots.index)
        m = XGBClassifier(random_state=draw, n_jobs=-1, **PARAMS).fit(
            pd.concat([X0, P.inputs(shots)]), pd.concat([y0, shots.Fault]),
            sample_weight=np.r_[np.ones(len(X0)), np.full(len(shots), 5.0)])
        pred = m.predict(P.inputs(rest))
        acc.append(accuracy_score(rest.Fault, pred)); cm += confusion_matrix(rest.Fault, pred, labels=range(K))
    return np.mean(acc), cm

# 3. Evaluation on the three test systems
for s, d in T.items():
    zero = accuracy_score(d.Fault, source.predict(P.inputs(d)))          # zero-shot: system never seen
    after5, _ = commission(d)                                            # after 5 readings per class
    within = []                                                          # upper bound: trained on that system
    for run in range(3):
        tr, te = train_test_split(d.index, test_size=0.2, stratify=d.Fault, random_state=run)
        m = XGBClassifier(random_state=run, n_jobs=-1, **PARAMS).fit(P.inputs(d.loc[tr]), d.Fault[tr])
        within.append(accuracy_score(d.Fault[te], m.predict(P.inputs(d.loc[te]))))
    print(f'System {s}: zero-shot {zero:.3f}  after 5 readings {after5:.3f}  within-system {np.mean(within):.3f}')

# 4. Weather-only check: can the class be guessed from sunlight and temperature alone? (should be ~1/6)
w_only = cross_val_score(XGBClassifier(random_state=0, n_jobs=-1, **PARAMS), LIB[['G/1000', 'AT/50']], y0,
                         cv=GroupKFold(5), groups=LIB.System).mean()
print(f'Weather-only accuracy: {w_only:.3f} (guessing = {1 / K:.3f})')

# 5. Explainable AI: exact TreeSHAP values for every input and class
expl = shap.TreeExplainer(source); sv = expl(X0)
imp = pd.DataFrame(np.abs(sv.values).mean(0), index=X0.columns, columns=LABELS)
for c in LABELS:
    print(f'{c:16s} main inputs: ' + ', '.join(imp[c].sort_values(ascending=False).index[:3]))

# 6. Save the model used by the Raspberry Pi program, the website and MATLAB/Simulink
source.save_model('models/xgb_source.json'); json.dump(PARAMS, open('models/xgb_params.json', 'w'), indent=1)
