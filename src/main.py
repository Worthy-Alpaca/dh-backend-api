import configparser
import os
import sys
import uvicorn
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Response, status, Request
from os.path import exists

PACKAGE_PARENT = "../"
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

try:
    from data.dataloader import DataLoader
    from data.datawrangler import MachineDataLoader
    from data.datawrangler import DataWrangler
    from simulation.cartsetup import CartSetup
    from simulation.machine import Machine
    from simulation.manufacturing import Manufacturing
    from models.deploy import DeployModel
    from schemas import DummyMachine
except:
    from src.data.dataloader import DataLoader
    from src.data.datawrangler import MachineDataLoader
    from src.data.datawrangler import DataWrangler
    from src.simulation.cartsetup import CartSetup
    from src.simulation.machine import Machine
    from src.simulation.manufacturing import Manufacturing
    from src.models.deploy import DeployModel
    from src.schemas import DummyMachine


app = FastAPI()
config = configparser.ConfigParser()
configPath = os.getcwd() + os.path.normpath("/src/settings.ini")
if exists(configPath):
    config.read(configPath)
else:
    config.add_section("default")
    config.set("default", "version", "1.0")
    config.add_section("network")
    config.set("network", "port", "5000")
    config.set("network", "host", "127.0.0.1")
    config.set("network", "basepath", "/api/v7/")


@app.get("/")
def root(response: Response, request: Request):
    """endpoint to check base status of API"""
    response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if exists(os.getcwd() + os.path.normpath("/data")):
        response.status_code = status.HTTP_200_OK
    body = {
        "API Version": config.get("default", "version"),
        "Status": response.status_code,
        "Headers": request.headers,
        "Basepath": config.get("network", "basepath"),
    }
    return body


@app.put(config.get("network", "basepath") + "/simulate/coating/")
def startSimulation(productId: str, dummyMachine: DummyMachine, response: Response):
    """endpoint to calculate the coating time for a given product"""
    # replace this with Database lookup
    path = Path(
        os.getcwd() + os.path.normpath("/data/programms/" + productId + "/" + "/m20")
    )
    data = DataLoader(path)
    machine = Machine(dummyMachine)
    manufacturing = Manufacturing(data(), machine)
    simulationData = manufacturing.coating()
    return {"time": simulationData}


@app.put(config.get("network", "basepath") + "/simulate/manufacturing/")
def startSimulation(
    productId: str, useIdealState: bool, dummyMachine: DummyMachine, response: Response
):
    """endpoint to calculate the manufacturing time for a given product"""
    # replace this with Database lookup
    path = Path(
        os.getcwd()
        + os.path.normpath("/data/programms/" + productId + "/" + dummyMachine.machine)
    )
    data = DataLoader(path)
    machine = Machine(dummyMachine)
    try:
        manufacturing = Manufacturing(data(), machine)
        simulationData = manufacturing(plotPCB=True, useIdealState=useIdealState)
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        body = {"error": e}
        return body
    return simulationData


@app.put(config.get("network", "basepath") + "/simulate/AI/")
def startSimulation(
    productId: str, useIdealState: bool, dummyMachine: DummyMachine, response: Response
):
    """endpoint to calculate the manufacturing time for a given product"""
    # replace this with Database lookup
    path = Path(
        os.getcwd()
        + os.path.normpath("/data/programms/" + productId + "/" + dummyMachine.machine)
    )
    data = DataLoader(path)
    machine = Machine(dummyMachine)
    try:
        data = data()
        offsets = max(data[2])
        manufacturing = Manufacturing(data, machine)
        plot_x, plot_y = manufacturing.getPlots()
        model = DeployModel(
            Path(os.getcwd() + os.path.normpath("/data/models/FINAL MODEL"))
        )
        predArray = np.array(
            [
                len(data[0] * len(data[2])),
                0 if machine.machineName == "m10" else 1,
                data[0]["X"].max() + offsets[0],
                data[0]["Y"].max() + offsets[1],
            ]
        )
        predictedData = model.predict(predArray)[0][0]
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        body = {"error": e}
        return body
    return {"time": float(predictedData), "plot_x": plot_x, "plot_y": plot_y}


@app.get(config.get("network", "basepath") + "/simulate/setup/")
def setupSimulation(
    productId: str,
    machine: str,
    randomInterMin: int,
    randomInterMax: int,
    response: Response,
):
    """endpoint to calculate the setup time for a given product"""
    path = Path(
        os.getcwd() + os.path.normpath("/data/programms/" + productId + "/" + machine)
    )
    try:
        data = DataLoader(path)
    except:
        return {"time": 420}

    setupM20 = CartSetup(data(), randomInterMin, randomInterMax)
    timeM20 = setupM20()
    return timeM20


@app.get(config.get("network", "basepath") + "/predict/order/")
async def predictOrder(request: Request, response: Response):
    """endpoint to predict the best order of products between two given times"""
    request.body()
    print(request.query_params.get("startdate"))
    print(request.query_params.get("enddate"))


@app.get(config.get("network", "basepath") + "/data/machinedata/")
def getMachineData(response: Response):
    """endpoint to receive machinedata"""
    try:
        data = MachineDataLoader()
    except:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return response

    return data.returnData()


@app.get(config.get("network", "basepath") + "/data/options/")
def getOptions():
    """endpoint to get all available programms"""
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
