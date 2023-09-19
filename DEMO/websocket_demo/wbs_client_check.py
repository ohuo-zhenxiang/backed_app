import json

import websocket
from loguru import logger

websocket_url = "ws://127.0.0.1:9527/api/wbs/get_task/test1"


# websocket_url = "ws://127.0.0.1:8000/items/foo/ws?token=some-key-token"

def on_message(ws, message):
    message = json.loads(message)
    print(message, type(message))

def on_error(ws, error):
    logger.error(error)


def on_close(ws, close_status_code, close_msg):
    logger.info(f"Closed with status code {close_status_code}: {close_msg}")


def on_open(ws):
    ws.send("Hello, WebSocket!")


if __name__ == "__main__":
    ws = websocket.WebSocketApp(websocket_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close, )
    ws.on_open = on_open

    ws.run_forever()
