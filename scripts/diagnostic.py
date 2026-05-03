import sys, os, json
sys.path.insert(0, 'src')

print('='*55)
print('  CRICKET BIOMECHANICS V2 -- FULL DIAGNOSTIC CHECK')
print('='*55)

# 1. Check model files
print('\n[1] MODEL FILES:')
import joblib
models = {
    'RF Model':   'models/cricket_model.pkl',
    'SHAP Model': 'models/cricket_shap_model.pkl',
    'Meta JSON':  'models/feature_meta.json',
}
for name, path in models.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'  [OK] {name}: {size/1024/1024:.2f} MB')
    else:
        print(f'  [MISSING] {name}')

# 2. RF model
print('\n[2] RF MODEL:')
try:
    rf = joblib.load('models/cricket_model.pkl')
    print(f'  [OK] Type: {type(rf).__name__}')
    print(f'  [OK] Trees: {rf.n_estimators}')
    print(f'  [OK] Features used: {rf.n_features_in_}')
    print(f'  [OK] Classes: {list(rf.classes_)}')
except Exception as e:
    print(f'  [ERR] {e}')

# 3. SHAP model
print('\n[3] SHAP/GB MODEL:')
try:
    gb = joblib.load('models/cricket_shap_model.pkl')
    print(f'  [OK] Type: {type(gb).__name__}')
    print(f'  [OK] Trees: {gb.n_estimators}')
    print(f'  [OK] Features used: {gb.n_features_in_}')
except Exception as e:
    print(f'  [ERR] {e}')

# 4. Feature meta
print('\n[4] FEATURE METADATA:')
meta = None
try:
    with open('models/feature_meta.json') as f:
        meta = json.load(f)
    print(f'  [OK] Feature count: {len(meta["feature_columns"])}/22')
    print(f'  [OK] RF Accuracy: {meta["rf_accuracy"]*100:.2f}%')
    print(f'  [OK] RF F1 Score: {meta["rf_f1"]*100:.2f}%')
    print(f'  [OK] CV Score:    {meta["cv_mean"]*100:.2f}%')
except Exception as e:
    print(f'  [ERR] {e}')

# 5. Feature extractor
print('\n[5] FEATURE EXTRACTOR (22 features):')
try:
    from feature_extractor import FeatureExtractor
    fe = FeatureExtractor()
    names = fe.get_feature_names()
    print(f'  [OK] Total features: {len(names)}')
    for i, n in enumerate(names, 1):
        print(f'       {i:2d}. {n}')
except Exception as e:
    print(f'  [ERR] {e}')

# 6. Scorer + profiles
print('\n[6] SCORER + PRO PROFILES:')
try:
    from scorer import BiomechanicsScorer, PRO_PROFILES
    scorer = BiomechanicsScorer()
    print(f'  [OK] Target ranges: {len(scorer._TARGETS)} features')
    print(f'  [OK] Profiles: {list(PRO_PROFILES.keys())}')
except Exception as e:
    print(f'  [ERR] {e}')

# 7. SHAP explainer
print('\n[7] SHAP EXPLAINER:')
try:
    from shap_explainer import SHAPExplainer, SHAP_AVAILABLE
    print(f'  [OK] SHAP library available: {SHAP_AVAILABLE}')
    if meta:
        gb2 = joblib.load('models/cricket_shap_model.pkl')
        exp = SHAPExplainer(gb2, meta['feature_columns'])
        print(f'  [OK] TreeExplainer initialized: {exp._explainer is not None}')
    else:
        print('  [SKIP] No meta to init explainer')
except Exception as e:
    print(f'  [ERR] {e}')

# 8. Phase + Shot
print('\n[8] PHASE DETECTOR + SHOT CLASSIFIER:')
try:
    from phase_detector import PhaseDetector, PHASE_NAMES
    from shot_classifier import ShotClassifier, SHOT_NAMES
    print(f'  [OK] Phases: {list(PHASE_NAMES.values())}')
    print(f'  [OK] Shots:  {list(SHOT_NAMES.values())}')
except Exception as e:
    print(f'  [ERR] {e}')

# 9. Player DB
print('\n[9] PLAYER DATABASE:')
try:
    from player_db import PlayerDatabase
    db = PlayerDatabase()
    players = db.list_players()
    label = players if players else 'None yet'
    print(f'  [OK] DB initialized')
    print(f'  [OK] Players recorded: {label}')
except Exception as e:
    print(f'  [ERR] {e}')

print('\n' + '='*55)
print('  DIAGNOSTIC COMPLETE')
print('='*55)
