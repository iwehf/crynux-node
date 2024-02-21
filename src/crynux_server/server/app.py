from functools import partial
from typing import Optional

from anyio import Event, create_task_group
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from hypercorn.asyncio import serve
from hypercorn.config import Config

from . import v1
from .middleware import add_middleware


class Server(object):
    def __init__(self, headless: bool, web_dist: str = "") -> None:
        self._app = FastAPI()
        v1.include_router(self._app, headless=headless, prefix="/manager")
        if not headless and web_dist != "":
            self._app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
        add_middleware(self._app)

        self._shutdown_event: Optional[Event] = None

    async def start(self, host: str, port: int, access_log: bool = True):
        assert self._shutdown_event is None, "Server has already been started."

        self._shutdown_event = Event()
        config = Config()
        config.bind = [f"{host}:{port}"]
        if access_log:
            config.accesslog = "-"
        config.errorlog = "-"

        try:
            await serve(self._app, config=config, shutdown_trigger=self._shutdown_event.wait)  # type: ignore
        finally:
            self._shutdown_event = None

    def stop(self):
        assert self._shutdown_event is not None, "Server has not been started."
        self._shutdown_event.set()

    @property
    def app(self):
        return self._app
