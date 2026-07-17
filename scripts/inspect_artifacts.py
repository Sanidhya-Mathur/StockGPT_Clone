"""Utility to inspect saved artifacts: metadata, model summary, scaler params.

Usage:
    python scripts/inspect_artifacts.py --metadata artifacts/metadata.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, default='artifacts/metadata.json')
    args = parser.parse_args()

    md_path = Path(args.metadata)
    if not md_path.exists():
        print(f"Metadata file not found: {md_path}")
        return

    md = json.loads(md_path.read_text(encoding='utf-8'))
    print("Metadata:")
    print(json.dumps(md, indent=2))

    # inspect model if present
    model_path = md.get('model_path')
    if model_path:
        mp = Path(model_path)
        if not mp.exists():
            print(f"Model file not found: {mp}")
        else:
            try:
                from tensorflow.keras.models import load_model
                print(f"Loading model from {mp} (this may take a moment)...")
                model = load_model(str(mp))
                print("\nModel architecture:")
                model.summary()
            except ImportError as e:
                print(f"Could not load model: TensorFlow not available. ({e})")
            except Exception as e:
                print(f"Error loading model: {e}")

    # inspect scaler
    scaler_path = md.get('scaler_path')
    if scaler_path:
        sp = Path(scaler_path)
        if not sp.exists():
            print(f"Scaler file not found: {sp}")
        else:
            try:
                import joblib
                scaler = joblib.load(str(sp))
                print("\nScaler details:")
                for attr in ('min_', 'scale_', 'data_min_', 'data_max_', 'feature_names_in_'):
                    if hasattr(scaler, attr):
                        val = getattr(scaler, attr)
                        if isinstance(val, (list, tuple)):
                            print(f"  {attr}: {val[:3]}..." if len(val) > 3 else f"  {attr}: {val}")
                        else:
                            print(f"  {attr}: {val}")
            except ImportError as e:
                print(f"Could not load scaler: joblib not available. ({e})")
            except Exception as e:
                print(f"Error loading scaler: {e}")


if __name__ == '__main__':
    main()
