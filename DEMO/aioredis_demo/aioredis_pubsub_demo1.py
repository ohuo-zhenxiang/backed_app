import asyncio
import async_timeout
import aioredis
from loguru import logger

STOPWORD = "STOP"


async def reader(channel: aioredis.client.PubSub):
    while True:
        try:
            async with async_timeout.timeout(1):
                message = await channel.get_message(ignore_subscribe_messages=True)
                if message is not None:
                    if message["channel"].decode() == "test1":
                        logger.success(f"(Reader) Message Received: {message['data']}")
                    else:
                        logger.error(f"(Reader) Message Received: {message['data']}")
                    if message["data"].decode() == STOPWORD:
                        print("(Reader) STOP")
                        break
                await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            pass


async def main():
    redis = aioredis.from_url("redis://:redis@localhost:6379/6")
    pubsub = redis.pubsub()
    # await pubsub.subscribe("channel:1", "channel:2")
    await pubsub.subscribe("test1", "test2")

    future = asyncio.create_task(reader(pubsub))

    # await redis.publish("channel:1", "Hello")
    # await redis.publish("channel:2", "World")

    await future


if __name__ == "__main__":
    asyncio.run(main())
