from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from supervisor import supervisor


class Message(BaseModel):
    content: str


app = FastAPI()


@app.get("/")
def read_root():
    """Main application endpoint."""
    return {"message": "Hello, world!"}


@app.post("/respond")
def respond(message: Message):
    """Main application endpoint."""
    generator = supervisor.respond(message.content)
    return StreamingResponse(generator, media_type="text/event-stream")


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring service status."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
