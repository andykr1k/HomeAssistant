from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

class API:
    """
    Wrapper class around FastAPI to initialize and manage routes, middleware, and startup/shutdown events.
    """

    def __init__(self):
        # Initialize FastAPI app
        self.app = FastAPI(title="HomeAssistant API", version="1.0.0")

        # Configure CORS (adjust as needed)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # change to your frontend URL in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes()

        # Startup and shutdown hooks
        @self.app.on_event("startup")
        async def startup_event():
            print("[API] Server started.")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            print("[API] Server stopped.")

    def _register_routes(self):
        """
        Define your API routes here.
        You can later import route handlers from other modules (TTS, STT, VLM, etc.).
        """
        @self.app.get("/")
        async def root():
            return {"status": "ok", "message": "HomeAssistant API is running"}

        @self.app.get("/ping")
        async def ping():
            return {"response": "pong"}

    def get_app(self) -> FastAPI:
        """Expose the FastAPI app for running with uvicorn or other ASGI servers."""
        return self.app


# Optional: allow direct execution
if __name__ == "__main__":
    import uvicorn
    api = API()
    uvicorn.run(api.get_app(), host="0.0.0.0", port=8000)
