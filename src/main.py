import configparser
import os
from os.path import exists

from pathlib import Path

from fastapi import FastAPI, Response, status
import uvicorn

from data.dataloader import DataLoader
from simulation.machine import Machine
from simulation.manufacturing import Manufacturing
from schemas import DummyMachine

app = FastAPI()
config = configparser.ConfigParser()
configPath = os.getcwd() + os.path.normpath("/src/settings.ini")
if exists(configPath):
    config.read(configPath)
else:
    config.add_section("default")
    config.add_section("network")
    config.set("network", "port", "5000")
    config.set("network", "host", "127.0.0.1")


@app.put("/")
def root(response: Response):
    """check base status of API"""
    response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if exists(os.getcwd() + os.path.normpath("/data")):
        response.status_code = status.HTTP_200_OK
    body = {
        "API Version": config.get("default", "version"),
        "Status": response.status_code,
    }
    return body


@app.put("/simulate/coating/{productId}")
def startSimulation(productId: str, dummyMachine: DummyMachine, response: Response):
    """endpoint to calculate the coating time for a given product"""
    # replace this with Database lookup
    path = Path(os.getcwd() + os.path.normpath("/data/" + productId))
    data = DataLoader(path)
    machine = Machine(dummyMachine)
    manufacturing = Manufacturing(data(), machine)
    simulationData = manufacturing.coating()
    return {"time": simulationData}


@app.put("/simulate/manufacturing/{productId}")
def startSimulation(productId: str, dummyMachine: DummyMachine, response: Response):
    """endpoint to calculate the manufacturing time for a given product"""
    # replace this with Database lookup
    path = Path(os.getcwd() + os.path.normpath("/data/" + productId))

    try:
        data = DataLoader(path)
    except:
        response.status_code = status.HTTP_404_NOT_FOUND
        return response
    machine = Machine(dummyMachine)
    manufacturing = Manufacturing(data(), machine)
    simulationData = manufacturing(plotPCB=True)
    return simulationData


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.get("network", "host"),
        port=config.getint("network", "port"),
        log_level="debug",
        reload=True,
    )
