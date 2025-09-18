from fastapi import FastAPI
from .routers import router

app = FastAPI(title="Adaptive Quiz API")
app.include_router(router)

# Optional root
@app.get("/")
def root():
    return {"name": "Adaptive Quiz API", "status": "running"}
