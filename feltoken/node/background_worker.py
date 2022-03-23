import argparse
import os
import time
from email.policy import default
from getpass import getpass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from web3 import Web3

from feltoken.core.average import average_models
from feltoken.core.contracts import to_dict
from feltoken.core.data import load_data
from feltoken.core.node import check_node_state, get_node, get_node_secret
from feltoken.core.storage import (
    export_model,
    ipfs_download_file,
    load_model,
    upload_encrypted_model,
    upload_final_model,
)
from feltoken.core.web3 import get_current_secret, get_project_contract, get_web3
from feltoken.node.training import train_model

# Load dotenv at the beginning of the program
load_dotenv()

# Path for saving logs and models during training
LOGS = Path(__file__).parent / "logs" / f"{time.time()}"

KEYS = {
    "main": os.getenv("PRIVATE_KEY"),
    "node1": os.getenv("NODE1_PRIVATE_KEY"),
    "node2": os.getenv("NODE2_PRIVATE_KEY"),
}


def get_plan(project_contract):
    """Get latest running plan else return None."""
    if project_contract.functions.isPlanRunning().call():
        length = project_contract.functions.numPlans().call()
        plan = project_contract.functions.plans(length - 1).call()
        return to_dict(plan, "TrainingPlan")
    return None


def execute_rounds(
    data, model, plan, plan_dir, secret, account, project_contract, w3, config
):
    """Perform training rounds according to the training plan.

    Args:
        ...

    Returns:
        (object): scikit-learn model
    """
    num_rounds = plan["numRounds"]
    for i in range(num_rounds):
        print(f"\nRunning round {i}")
        round_dir = plan_dir / f"round_{i}"
        round_dir.mkdir(exist_ok=True)

        # 2. Execute training
        print("Training")
        model = train_model(model, data, config)

        # 3. Encrypt the model
        model_path = round_dir / "node_model.joblib"
        cid = upload_encrypted_model(model, model_path, secret)

        # 5. Send model to the contract (current round)
        tx = project_contract.functions.submitModel(cid).transact(
            {"from": account.address, "gasPrice": w3.eth.gas_price}
        )
        w3.eth.wait_for_transaction_receipt(tx)

        # 6. Download models and wait for round finished
        models = [model]
        downloaded = set()
        print("Waiting for other nodes to finish round.")
        while len(models) < plan["numNodes"]:
            length = project_contract.functions.getNodesLength().call()
            for node_idx in range(length):
                node = project_contract.functions.nodesArray(node_idx).call()
                node = to_dict(node, "Node")
                if (
                    node_idx in downloaded
                    or not node["activated"]
                    or node["_address"] == account.address
                ):
                    continue

                cid = project_contract.functions.getRoundModel(
                    i, node["_address"]
                ).call()
                if len(cid) < 5:
                    continue
                print(f"Downloading CID from node {node_idx}", cid)

                m_path = round_dir / f"model_node_{node_idx}.joblib"
                ipfs_download_file(cid, m_path, secret)
                models.append(load_model(m_path))
                downloaded.add(node_idx)

        print("Averaging models.", len(models))
        # 7. Average models
        model = average_models(models)
    return model


def watch_for_plan(project_contract):
    """Wait until new plan created."""
    # TODO: Use contract emiting events
    print("Waiting for a plan.")
    while True:
        plan = get_plan(project_contract)
        if plan is not None:
            return plan
        time.sleep(3)


def task(data, config):
    account = Web3().eth.account.from_key(config.account)
    w3 = get_web3(account, config.chain)
    print("Worker connected to chain id: ", w3.eth.chain_id)

    project_contract = get_project_contract(w3, config.contract)
    if not check_node_state(w3, project_contract, account):
        print("Script stoped.")
        return
    print("Node is ready for training.")

    # Obtain secret from the contract
    node = get_node(project_contract, account)
    SECRET = get_node_secret(project_contract, account)

    while True:
        plan = watch_for_plan(project_contract)
        print("Executing a plan!")
        # Use random seed from contract
        np.random.seed(plan["randomSeed"])

        secret = get_current_secret(SECRET, node["entryKeyTurn"], plan["keyTurn"])

        # Creat directory for storing plan
        plan_index = project_contract.functions.numPlans().call()
        plan_dir = LOGS / f"plan_{plan_index}"
        plan_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download model by CID
        base_model_path = plan_dir / "base_model.joblib"
        ipfs_download_file(plan["baseModelCID"], output_path=base_model_path)
        model = load_model(base_model_path)

        final_model = execute_rounds(
            data,
            model,
            plan,
            plan_dir,
            secret,
            account,
            project_contract,
            w3,
            config,
        )
        print("Creating final model.")
        final_model_path = plan_dir / "final_model.joblib"
        export_model(model, final_model_path)

        # 8. Upload final model if coordinator
        if plan["finalNode"] == account.address:
            print("Node selected as a final one.")
            # Generate builder secret based on random seed and secret
            # So it is same for all nodes
            builder = project_contract.functions.builders(plan["creator"]).call()
            builder = to_dict(builder, "Builder")

            cid = upload_final_model(
                final_model, final_model_path, builder["publicKey"]
            )

            tx = project_contract.functions.finishPlan(cid).transact(
                {"from": account.address, "gasPrice": w3.eth.gas_price}
            )
            w3.eth.wait_for_transaction_receipt(tx)
            print("Final model uploaded and encrypted for a builder.")

        time.sleep(30)
        print("Plan finished!")


def parse_args(args_str=None):
    """Parse and partially validate arguments form command line.
    Arguments are parsed from string args_str or command line if args_str is None

    Args:
        args_str (str): string with arguments or None if using command line

    Returns:
        Parsed args object
    """
    parser = argparse.ArgumentParser(
        description="Data provider worker script managing the trainig."
    )
    parser.add_argument(
        "--chain",
        type=int,
        help="Chain Id of chain to which should be the worker connected.",
    )
    parser.add_argument("--contract", type=str, help="Contract address")
    parser.add_argument(
        "--account",
        type=str,
        default="main",
        help="Name of account to use as specified in .env (main, node1, node2)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        help=(
            "Path to CSV file with data. Last column is considered as Y."
            "Or Ocean protocol dataset DID."
        ),
    )
    # OCEAN protocol related
    parser.add_argument(
        "--ocean",
        type=bool,
        action="store_true",
        help="Indicates if the dataset is compute-to-data dataset on ocean.",
    )
    parser.add_argument(
        "--algorithm_did",
        type=str,
        default=None,
        help=(
            "DID of published algorithm which is allowed for training on data."
            "Only required if --ocean is set."
        ),
    )
    args = parser.parse_args(args_str)

    assert args.chain in [
        1337,
        80001,
        137,
    ], "Invalid chain id or chain id is not supported (suppoerted: 1337, 137, 80001)"
    assert len(args.contract) == 42, "The contract address has invalid length."
    assert (
        not args.ocean or args.algorithm_did
    ), "algorithm_did must be set if ocean is True."

    args.account = KEYS.get(args.account, None)
    return args


def main(args_str=None):
    """Parse arguments and run worker task (watching contract and training models)."""
    config = parse_args(args_str)
    data = load_data(config.data)

    # Check for valid key and valid web3 token
    if not config.account:
        config.account = getpass(
            "Please provide your private key (exported from MetaMask):"
        )

    if "WEB3_STORAGE_TOKEN" not in os.environ or not os.getenv("WEB3_STORAGE_TOKEN"):
        os.environ["WEB3_STORAGE_TOKEN"] = getpass(
            "Please input your web3.storage API token:"
        )

    task(data, config)


if __name__ == "__main__":
    main()
