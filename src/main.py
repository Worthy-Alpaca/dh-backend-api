import configparser
import os
from os.path import exists

from pathlib import Path

from fastapi import FastAPI, Response, status, Request
import uvicorn

from data.dataloader import DataLoader
from data.datawrangler import MachineDataLoader
from data.datawrangler import DataWrangler
from simulation.cartsetup import CartSetup
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
    config.set("network", "basepath", "/api/v1/")


@app.put(config.get("network", "basepath") + "/")
def root(response: Response, request: Request):
    """check base status of API"""
    response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if exists(os.getcwd() + os.path.normpath("/data")):
        response.status_code = status.HTTP_200_OK
    body = {
        "API Version": config.get("default", "version"),
        "Status": response.status_code,
        "Headers": request.headers,
    }
    return body


@app.put(config.get("network", "basepath") + "/simulate/coating/")
def startSimulation(productId: str, dummyMachine: DummyMachine, response: Response):
    """endpoint to calculate the coating time for a given product"""
    # replace this with Database lookup
    path = Path(os.getcwd() + os.path.normpath("/data/programms/" + productId + "/" + "/m20"))
    data = DataLoader(path)
    machine = Machine(dummyMachine)
    manufacturing = Manufacturing(data(), machine)
    simulationData = manufacturing.coating()
    return {"time": simulationData}


@app.put(config.get("network", "basepath") + "/simulate/manufacturing/")
def startSimulation(productId: str, dummyMachine: DummyMachine, response: Response):
    """endpoint to calculate the manufacturing time for a given product"""
    # replace this with Database lookup
    path = Path(
        os.getcwd() + os.path.normpath("/data/programms/" + productId + "/" + dummyMachine.machine)
    )
    data = DataLoader(path)
    """try:
        data = DataLoader(path)
    except:
        response.status_code = status.HTTP_404_NOT_FOUND
        return response"""
    machine = Machine(dummyMachine)
    try:
        manufacturing = Manufacturing(data(), machine)
        simulationData = manufacturing(plotPCB=True)
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        body = {"error": e}
        return body
    return simulationData


@app.get(config.get("network", "basepath") + "/simulate/setup")
def setupSimulation(productId: str, machine: str, randomInterMin: int, randomInterMax: int):
    path = Path(os.getcwd() + os.path.normpath("/data/programms/" + productId + machine))

    data = DataLoader(path)
    setupM20 = CartSetup(data(), randomInterMin, randomInterMax)
    timeM20 = setupM20()
    return {machine: timeM20}


@app.get(config.get("network", "basepath") + "/predict/order/")
async def predictOrder(request: Request, response: Response):
    request.body()
    print(request.query_params.get("startdate"))
    print(request.query_params.get("enddate"))


@app.get(config.get("network", "basepath") + "/data/machinedata")
def getMachineData(response: Response):
    try:
        data = MachineDataLoader()
    except:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return response

    return data.returnData()


@app.get(config.get("network", "basepath") + "/data/options")
def getOptions():
    # replace with DB lookup for all possible programms
    path = Path(os.getcwd() + os.path.normpath("/data/programms"))

    return {"programms": os.listdir(path)}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.get("network", "host"),
        port=config.getint("network", "port"),
        log_level="debug",
        reload=True,
    )
