import uvicorn
from io import StringIO, BytesIO
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model import CatBoostModel


class ModelTrainParams(BaseModel):
    weights_name: str
    target_column_name: str
    iterations: int
    depth: int

class ModelPredictParams(BaseModel):
    weights_name: str
    target_column_name: str


BACKEND_PORT = 12345

app = FastAPI()
model = CatBoostModel()

@app.post("/train/")
async def train(params: ModelTrainParams = Depends(ModelTrainParams), file: UploadFile = File(...)):

    bfile = await file.read()
    df = pd.read_csv(StringIO(bfile.decode('utf-8')))

    cbmodel = CatBoostModel()

    X_train, y_train = cbmodel.process_data(df, target_column_name=params.target_column_name)
    cbmodel.train(X_train,
                  y_train,
                  iterations=params.iterations,
                  depth=params.depth)

    cbmodel.save(params.weights_name)

    return params.weights_name

@app.post("/predict/")
async def predict(params: ModelPredictParams = Depends(ModelPredictParams), file: UploadFile = File(...)):
    bfile = await file.read()
    df = pd.read_csv(StringIO(bfile.decode("utf-8")))

    predictions = pd.DataFrame([], index=df.index)

    cbmodel = CatBoostModel()

    cbmodel.load(params.weights_name)
    predictions[params.target_column_name] = cbmodel.predict(df)
    bfile_response = BytesIO(predictions.to_csv().encode("utf-8"))

    return StreamingResponse(content=bfile_response, media_type="text/csv")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=BACKEND_PORT)