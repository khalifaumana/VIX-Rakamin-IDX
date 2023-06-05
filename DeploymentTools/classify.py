import joblib
import pandas as pd


def scaleFeatures(loan_amnt, annual_inc):
    #[loan_amnt, annual_inc, dti, open_acc, inq_last_6mths]
    sc= joblib.load('/Users/fau/Project/Other/VIX/ID:X PArtner/VIX-Rakamin-IDX/DeploymentTools/scaler.bin')
    scaled = sc.transform([[loan_amnt, annual_inc]])
    return scaled[0]

def predict(data):
    clf = joblib.load("/Users/fau/Project/Other/VIX/ID:X PArtner/VIX-Rakamin-IDX/DeploymentTools/xgb_classified.sav")
    cat, proba = clf.predict(data), clf.predict_proba(data)
    return  cat, proba