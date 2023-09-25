import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer

async def check_webrtc_stream(url, timeout=5):
    try:
        pc = RTCPeerConnection()

        @pc.on("datachannel")
        def on_datachannel(channel):
            pass

        pc.createDataChannel("dummy")

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # 等待连接状态变为 "connected"
        while pc.connectionState != "connected":
            await asyncio.sleep(1)

        return True, "WebRTC流可用"
    except Exception as e:
        return False, f"WebRTC流错误：{str(e)}"

if __name__ == "__main__":
    webrtc_url = "webrtc://192.168.199.18/live/2c9280826cfb0ed5016cfb10556c001c"  # 替换为实际的WebRTC流地址
    loop = asyncio.get_event_loop()
    success, message = loop.run_until_complete(check_webrtc_stream(webrtc_url))
    if success:
        print(message)
    else:
        print(f"检测失败：{message}")

