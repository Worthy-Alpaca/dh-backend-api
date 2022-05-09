import os

from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import uvicorn

from simulation.dataloader import DataLoader
from simulation.machine import Machine
from simulation.manufacturing import Manufacturing
from schemas import DummyMachine

app = FastAPI()

class Item(BaseModel):
    name: str


@app.put("/")
def root(item: Item):
    return item


@app.put("/simulate/coating/{product_id}")
def start_simulation(product_id: str, dummyMachine: DummyMachine, response: Response):
    # replace this with Database lookup
    path = Path(os.getcwd() + os.path.normpath('/data/' + product_id))
    #machine = Machine(dummyMachine.machineName, dummyMachine.cph, dummyMachine.nozHeads, dummyMachine.SMD, dummyMachine.offsets)
    data = DataLoader(path)
    machine = Machine(dummyMachine)
    manufacturing = Manufacturing(data(), machine)
    simulationData = manufacturing.coating()
    return {"time": simulationData}

@app.put("/simulate/manufacturing/{product_id}")
def start_simulation(product_id: str, dummyMachine: DummyMachine, response: Response):
    # replace this with Database lookup
    path = Path(os.getcwd() + os.path.normpath('/data/' + product_id))
    
    try:
        data = DataLoader(path)
    except:
        response.status_code = status.HTTP_404_NOT_FOUND
        return response
    machine = Machine(dummyMachine)
    manufacturing = Manufacturing(data(), machine)    
    simulationData = manufacturing(plotPCB=True)
    return simulationData

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="debug", reload=True)