from fastapi import FastAPI
app=FastAPI()
@app.get("/")
def home():
    return{"Message":"my first api is working"}