from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"message": "ResQTwin Backend is running!"}