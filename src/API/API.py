import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class CommandRequest(BaseModel):
    text: str


class API:
    """Wrapper class around FastAPI to initialize routes and integrations."""

    def __init__(self, state=None, command_handler=None, debug: bool = False):
        self.app = FastAPI(title="HomeAssistant API", version="1.0.0")
        self._state = state
        self._command_handler = command_handler
        self._debug = debug
        self._logger = logging.getLogger(__name__)

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

        @self.app.get("/state")
        async def state():
            if not self._state:
                return {"status": "error", "message": "State not configured"}
            return {"status": "ok", "state": self._state.snapshot()}

        @self.app.post("/command")
        async def command(payload: CommandRequest):
            if not self._command_handler:
                return {"status": "error", "message": "Command handler not configured"}
            if self._debug:
                self._logger.debug("API command: %s", payload.text)
            self._command_handler(payload.text, source="api")
            return {"status": "accepted", "text": payload.text}

    def get_app(self) -> FastAPI:
        return self.app

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn
        uvicorn.run(self.get_app(), host=host, port=port)
