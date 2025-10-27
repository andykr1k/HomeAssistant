from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

class API:
    """
    Wrapper class around FastAPI to initialize and manage routes, middleware, and startup/shutdown events.
    """

    def __init__(self):
        self.app = FastAPI(title="HomeAssistant API", version="1.0.0")

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._register_routes()

        @self.app.on_event("startup")
        async def startup_event():
            print("[API] Server started.")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            print("[API] Server stopped.")

    def _register_routes(self):
        @self.app.get("/")
        async def root():
            return {"status": "ok", "message": "HomeAssistant API is running"}

        @self.app.get("/ping")
        async def ping():
            return {"response": "pong"}

    def get_app(self) -> FastAPI:
        return self.app

    def run(self):
        import uvicorn
        api = API()
        uvicorn.run(api.get_app(), host="0.0.0.0", port=8000)
