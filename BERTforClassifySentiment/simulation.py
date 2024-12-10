from flwr.simulation import run_simulation
from flowerSimulation.client_app import app as client_app
from flowerSimulation.server_app import app as server_app

if __name__ == "__main__":
    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=100
    )

