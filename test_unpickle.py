import pickle, os, sys

for f in ("model.pkl","label_encoder.pkl"):
    print("---", f, "---")
    if not os.path.exists(f):
        print("MISSING")
        continue
    try:
        with open(f, "rb") as fh:
            obj = pickle.load(fh)
        print("OK - loaded type:", type(obj))
    except Exception as e:
        print("LOAD ERROR:", repr(e))
