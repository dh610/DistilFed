from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flowerSimulation.task import get_params, get_model

def evaluate_metrics_aggregation_fn(metrics):
    """
    간단한 evaluate 단계의 메트릭 집계 함수.
    모든 메트릭의 총합을 계산하여 반환합니다.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    # Construct ServerConfig
    num_rounds = 3
    config = ServerConfig(num_rounds=num_rounds)

    # Set global model initialization
    model_name = "prajjwal1/bert-tiny"
    ndarrays = get_params(get_model(model_name))
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy
    fraction_fit = 0.02
    fraction_evaluate = 0.02
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=global_model_init,
        # min_fit_clients=2,
        # min_evaluate_clients=2,
        # fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)