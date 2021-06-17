import asyncio
import websockets
import json
import cv2
import numpy as np
import base64

from PIL import Image

host = "localhost"
port = 3000

lock = asyncio.Lock()
clients = set()
#processes = []


async def register(websocket):
    global lock
    global clients
    async with lock:
        clients.add(websocket)
        #remote_ip = websocket.remote_address[0]
        #msg='[{ip}] connected'.format(ip=remote_ip)
        #print(msg)


async def unregister(websocket):
    global lock
    global clients
    async with lock:
        clients.remove(websocket)
        #remote_ip = websocket.remote_address[0]
        #msg='[{ip}] disconnected'.format(ip=remote_ip)
        #print(msg)


async def thread(websocket, path):
    await register(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            remote_ip = websocket.remote_address[0]
            # save image (for testing)
            image = base64.b64decode(data['image'])
            image_np = np.frombuffer(image, dtype=np.uint8)
            cv2.imwrite('image.jpg', cv2.imdecode(image_np, flags=1))
            # test response status
            send = json.dumps({'status': 'success'})
            await websocket.send(send)
    finally:
        await unregister(websocket)


print('run verification server')
start_server = websockets.serve(thread, host, port)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()