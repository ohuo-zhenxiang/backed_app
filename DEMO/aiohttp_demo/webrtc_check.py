import aiohttp
import asyncio
from urllib.parse import urlparse, urlunparse


async def check_webrtc_connection(webrtc_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(webrtc_url) as response:
                if response.status == 200:
                    return True
                else:
                    return False
    except Exception as e:
        print(f"error: {e}")
        return False


async def main():
    webrtc_url = "webrtc://192.168.199.18/live/2c9280826cfb0ed5016cfb10556c001c"
    parsed_url = urlparse(webrtc_url)
    modified_url = urlunparse((parsed_url.scheme, f"{parsed_url.hostname}:{8080}", parsed_url.path, parsed_url.params,
                               parsed_url.query, parsed_url.fragment))
    print(modified_url)
    result = await check_webrtc_connection(modified_url)
    print(result)
    if result:
        print("webrtc connection is ok")
    else:
        print("webrtc connection is not ok")


if __name__ == "__main__":
    asyncio.run(main())
