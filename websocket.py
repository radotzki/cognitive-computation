import asyncio
import datetime
import random
import websockets


async def broadcast(websocket, path):
    while True:
        data_file = open('data.json', 'r')
        data = data_file.read()
        data_file.close()
        await websocket.send(data)
        await asyncio.sleep(1)

start_server = websockets.serve(broadcast, '127.0.0.1', 5678)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
