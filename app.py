from fastapi import FastAPI
from controllers import router

app = FastAPI(title="Face Recognition API")

app.include_router(router)