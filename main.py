import pandas as pd
from fastapi import FastAPI
from model import iyzico
from scipy.stats import ks_2samp
import joblib
import pandas as pd

data = {
    'sales_ewm_alpha_09_lag_91': [2557.298, 1181.488, 2248.244, 2963.412],
    'day_of_week_6': [0, 0, 0, 0],
    'sales_roll_mean_182': [1726.533, 1149.270, 2585.613, 4218.276],
    'sales_roll_mean_360': [1712.203, 1205.222, 1264.684, 3839.742],
    'sales_ewm_alpha_095_lag_91': [2531.099, 1185.103, 2217.440, 2963.979],
    'sales_roll_mean_178': [1723.638, 1147.563, 2604.501, 4108.651],
    'sales_roll_mean_179': [1726.740, 1147.354, 2595.448, 4146.495],
    'sales_roll_mean_181': [1728.182, 1149.070, 2585.853, 4196.824],
    'sales_roll_mean_92': [1751.024, 1098.849, 2968.682, 2722.009],
    'sales_roll_mean_91': [1752.490, 1095.303, 2980.283, 2724.173],
    'sales_lag_364': [1982.756, 1112.679, 76.811, 1021.763],
    'sales_lag_91': [2493.808, 1187.727, 2183.526, 2966.549],
    'Total_Transaction': [8.157, 7.185, 7.998, 7.892]
}

train_df = pd.DataFrame(data)

predictions = [7.44434660373865, 7.023226223535835, 7.390302649003101, 8.143633824074207]


predict_df = pd.DataFrame({'Total_Transaction': predictions})

iyzico_load = joblib.load("iyizco_model1.pkl")

app = FastAPI()


def detect_drift(data1, data2):
    ks_result = ks_2samp(data1, data2)
    if ks_result.pvalue < 0.05:
        return "Drift exist"
    else:
        return "Drift doesn't exist"


def make_iyzico_predict(model, request):
    sales_roll_mean_91 = request["sales_roll_mean_91"]
    sales_roll_mean_92 = request["sales_roll_mean_92"]
    sales_roll_mean_360 = request["sales_roll_mean_360"]
    sales_roll_mean_182 = request["sales_roll_mean_182"]
    day_of_week_6 = request["day_of_week_6"]
    sales_lag_91 = request["sales_lag_91"]
    sales_lag_364 = request["sales_lag_364"]
    sales_ewm_alpha_095_lag_91 = request["sales_ewm_alpha_095_lag_91"]
    sales_ewm_alpha_09_lag_91 = request["sales_ewm_alpha_09_lag_91"]
    sales_roll_mean_178 = request["sales_roll_mean_178"]
    sales_roll_mean_179 = request["sales_roll_mean_179"]
    sales_roll_mean_181 = request["sales_roll_mean_181"]

    count = [[sales_roll_mean_91, sales_roll_mean_92, sales_roll_mean_360, sales_roll_mean_182,
              day_of_week_6, sales_lag_91, sales_lag_364, sales_ewm_alpha_095_lag_91, sales_ewm_alpha_09_lag_91,
              sales_roll_mean_178, sales_roll_mean_179, sales_roll_mean_181]]
    prediction = model.predict(count)

    return prediction[0]


@app.post("/prediction/iyzico")
async def predict_iyzico(request: iyzico):
    prediction = make_iyzico_predict(iyzico_load, request.dict())
    return {"predict": prediction}


@app.get("/detect-drift")
async def drift_detection():
    data1 = train_df["Total_Transaction"]  # Ýlk veri kümesini belirleyin
    data2 = predict_df["Total_Transaction"]  # Ýkinci veri kümesini belirleyin
    result = detect_drift(data1, data2)
    return {"drift_status": result}


@app.get("/")
async def root():
    return {"data": "mlops final project"}


