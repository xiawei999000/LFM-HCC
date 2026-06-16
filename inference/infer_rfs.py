#
RFS prediction model inference.
Usage:
  from infer_rfs import RFSInferencer
  infer = RFSInferencer()
  risk_scores = infer.predict(dataframe)

Input DataFrame requirements:
  -   - Deep feature models: must contain all feature columns (feat_0, feat_1, ...)
  -   - Clinical model: must contain Sex, Age, MVI, PathologicalGrade, HBsAg, AFP, ALT, AST, GGT
  -   - Multiple feature groups can be included; auto-matched by model name
#
import pandas as pd, numpy as np, os, json, joblib, warnings
warnings.filterwarnings('ignore')

from sksurv.linear_model import CoxnetSurvivalAnalysis


class RFSInferencer:
    #Load trained RFS models and run inference.#

    def __init__(self, model_dir=None):
        #
        model_dir: path to RFS_model directory (default: ./RFS_model)
        #
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RFS_model')
        self.model_dir = model_dir
        self.models = {}
        self._load_all()

    def _load_all(self):
        #Load all models#
        for name in sorted(os.listdir(self.model_dir)):
            path = os.path.join(self.model_dir, name)
            if not os.path.isdir(path):
                continue
            cfg_file = os.path.join(path, 'config.json')
            imp_file = os.path.join(path, 'imputer.joblib')
            scl_file = os.path.join(path, 'scaler.joblib')
            mdl_file = os.path.join(path, 'cox_model.joblib')
            if not all(os.path.exists(f) for f in [cfg_file, imp_file, mdl_file, scl_file]):
                print(f'[WARN] {name}: Missing model files, skipping')
                continue
            with open(cfg_file, 'r') as f:
                cfg = json.load(f)
            self.models[name] = {
                'config': cfg,
                'imputer': joblib.load(imp_file),
                'scaler': joblib.load(scl_file),
                'model': joblib.load(mdl_file),
            }
            nz = cfg.get('n_nonzero', '?')
            print(f'[LOADED] {name} ({cfg["model_type"]}), '
                  f'{cfg["n_selected"]} features, {nz} non-zero')

    @property
    def model_names(self):
        return list(self.models.keys())

    def predict(self, data, model_name=None):
        #
        Compute risk scores (log hazard ratio) from input DataFrame.
        Returns dict: {model_name: np.array of risk scores}
        If model_name is specified, returns only that model array.
        #
        if model_name is not None:
            return self._predict_one(data, model_name)

        results = {}
        for name in self.model_names:
            try:
                results[name] = self._predict_one(data, name)
            except Exception as e:
                print(f'[ERROR] {name}: {e}')
                results[name] = np.full(len(data), np.nan)
        return results

    def _predict_one(self, data, model_name):
        #Run inference for a single model.#
        if model_name not in self.models:
            raise KeyError(f'Unknown model: {model_name}. Available: {self.model_names}')

        m = self.models[model_name]
        cfg = m['config']
        features = cfg['selected_features']

        # Check if required features exist
        missing = [f for f in features if f not in data.columns]
        if missing:
            raise KeyError(f'{model_name}: Missing {len(missing)} features: {missing[:5]}...')

        X = data[features].values.astype(float)
        X_imp = m['imputer'].transform(X)
        X_scaled = m['scaler'].transform(X_imp)
        risk = m['model'].predict(X_scaled)
        return risk

    def predict_survival(self, data, times, model_name=None):
        #
        Predict survival probabilities at specified time points.
        times: list of time points, e.g. [12, 24, 36]
        Returns: {model_name: DataFrame(rows=samples, cols=times)}
        #
        risk = self.predict(data, model_name)
        if model_name is not None:
            risk = {model_name: risk}

        surv_results = {}
        for name, r in risk.items():
            cfg = self.models[name]['config']
            features = cfg['selected_features']
            X = data[features].values.astype(float)
            X_imp = self.models[name]['imputer'].transform(X)
            X_scaled = self.models[name]['scaler'].transform(X_imp)
            surv_matrix = self.models[name]['model'].predict_survival_function(X_scaled)
            model_times = self.models[name]['model'].event_times_
            surv_list = []
            for i in range(surv_matrix.shape[0]):
                surv_list.append(np.interp(times, model_times, surv_matrix[i, :],
                                           left=1.0, right=0.0))
            surv_at_times = np.column_stack(surv_list).T
            surv_results[name] = pd.DataFrame(surv_at_times, columns=[f'{t}m' for t in times])
        return surv_results

    def info(self):
        #Print configuration info for all loaded models.#
        rows = []
        for name, m in self.models.items():
            cfg = m['config']
            rows.append({
                'Model': name,
                'Type': cfg['model_type'],
                'Features': cfg['n_selected'],
                'Non-zero': cfg.get('n_nonzero', '-'),
                'Train C': cfg.get('train_c_index', '-'),
                'Test C': cfg.get('test_c_index', '-'),
            })
        return pd.DataFrame(rows)


# ============================================================
# CLI entry: python infer_rfs.py <input.xlsx> [output.xlsx]
# ============================================================
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage: python infer_rfs.py <input.xlsx> [output.xlsx]')
        print('    input.xlsx: must contain required feature columns')
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.xlsx', '_risk.xlsx')

    data = pd.read_excel(input_path)
    print(f'Loaded {len(data)} samples from {input_path}')

    infer = RFSInferencer()
    print(f'\nAvailable models: {infer.model_names}')

    risk_dict = infer.predict(data)
    df_risk = data.copy()
    for name, risk in risk_dict.items():
        df_risk[f'{name}_risk'] = risk
        print(f'  {name}: risk range [{risk.min():.4f}, {risk.max():.4f}]')

    df_risk.to_excel(output_path, index=False)
    print(f'\nSaved risk scores to: {output_path}')
