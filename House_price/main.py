import io
from fastapi import FastAPI,HTTPException,UploadFile,File
import pandas as pd
import joblib
from pydantic import BaseModel,Field
from fastapi.responses import StreamingResponse


app=FastAPI()

model=joblib.load("house_model.joblib")
features=joblib.load("house_features.joblib")

#input schema

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in the neighborhood")
    HouseAge: float = Field(gt=0, description="Median house age in the neighborhood")
    AveRooms: float = Field(gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms per household")
    Population: float = Field(gt=0, description="Block population")
    AveOccup: float = Field(gt=0, description="Average number of household members")
    Latitude: float = Field(description="Latitude of the block")
    Longitude: float = Field(description="Longitude of the block")


@app.get("/")
def home():
    return{
        "message":"California house prediction api",
        "status":"running",
        "endpoint":"send POST request to / predict"
    }    

@app.get("/health")
def health():
    return{
        "status":"running",
        "model":"RandomForestRegression",
        "features":features,
        "avg_error":"$32,773"
    }


@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        prediction = model.predict(input_data)[0]
        price_usd=prediction * 100000
        

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short":f"${prediction:,.2f} hundred thousands",
            "finance_range":f"${price_usd -32773:,.0f} to ${price_usd + 32773:,.0f} "
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed:{str(e)}"
        )

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    # 1. Check file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file only"
        )

    # 2. Read uploaded file
    content = await file.read()

    # 3. Convert CSV into DataFrame
    df = pd.read_csv(io.BytesIO(content))

    # 4. Required columns
    required_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ]

    # 5. Check missing columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file: {missing_columns}"
        )

    # 6. Check empty file
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has no data rows"
        )

    try:
        # 7. Select only model features
        input_data = df[required_columns]

        # 8. Predict
        predictions = model.predict(input_data)

        # 9. Convert to dollars
        predictions_usd = predictions * 100000

        # 10. Add prediction columns
        df["predicted_price_usd"] = predictions_usd

        df["predicted_price"] = df["predicted_price_usd"].apply(
            lambda value: f"${value:,.0f}"
        )

        # 11. Convert DataFrame to CSV
        output = df.to_csv(index=False)

        # 12. Return CSV file
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=prediction.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please upload a CSV File only"
        )
    content = await file.read()
    df=pd.DataFrame(io.BytesIO(content))
    required_columns={
                    "MedInc",
                    "HouseAge",
                    "AveRooms",
                    "AveBedrms",
                    "Population",
                    "AveOccup",
                    "Latitude",
                    "Longitude"
    }
    missing_columns={
        col for col in required_columns
        if col not in df.columns
    }
    
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing from your file{missing_columns}"
        )
    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail="The upload file has no data rows"
        )

    try:
        predictions_usd = model.predict(df[list(required_columns)]) * 100000
        df["predicted_price_usd"] = predictions_usd
        df["predicted_price"] = df["predicted_price_usd"].apply(
            lambda value: f"${value:,.0f}"
        )
        output = df.to_csv(index=False)
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=prediction.csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )   
        
        
                

 