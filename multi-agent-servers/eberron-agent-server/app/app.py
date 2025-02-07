import uvicorn
from fastapi import FastAPI

from supervisor import supervisor

app = FastAPI()

@app.get("/")
def read_root():
    """Main application endpoint."""
    return {"message": "Hello, world!"}

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring service status."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)