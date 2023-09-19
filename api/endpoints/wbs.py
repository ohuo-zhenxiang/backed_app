import json

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket

import schemas
import crud
import models
from api import deps
from datetime import datetime
import time
from settings import REDIS_URL
from loguru import logger
import aioredis

router = APIRouter()
wbs_logger = logger.bind(name="WBS")


@router.websocket("/get_task/{task_token}")
async def websocket_endpoint(websocket: WebSocket, task_token: str) -> None:
    """
    Websocket for getting task progress updates
    :param websocket: WebSocket object.
    :param task_token: Task token
    :return:
    url: ws://localhost:9527/api/wbs/get_task/{task_token}
    """
    try:
        await websocket.accept()
        channel = task_token

        # logger.debug(f"Listening to channel {channel}")
        rds = await aioredis.from_url(REDIS_URL)
        sub = rds.pubsub()
        await sub.subscribe(channel)

        async for message in sub.listen():
            if message and isinstance(message, dict):
                data = message.get("data")
                if isinstance(data, bytes):
                    print(type(data))
                    await websocket.send_bytes(data)

        logger.debug(f"Unsubscribing from channel {channel}")
        await sub.unsubscribe(channel)
        await sub.close()
        rds.close()
    except Exception as e:
        # logger.error(e)
        pass
