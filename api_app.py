from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json

app = FastAPI()

@app.get("/predictions")
def get_predictions():
    """
    Returns the single set of prediction values (first JSON).
    """

    with open("./results.json", "r") as file:
        predictions_data = json.load(file)

    return JSONResponse(content=predictions_data)


@app.get("/model-results")
def get_model_results():
    """
    Returns the list of model results (the second JSON).
    """
    with open("./modeling/predictions_metrics.json", "r") as file:
        models_data = json.load(file)
    return JSONResponse(content=models_data)

