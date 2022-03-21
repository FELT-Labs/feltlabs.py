import os

from dotenv import load_dotenv

load_dotenv()


ALG_metadata = {
    "main": {
        "type": "algorithm",
        "algorithm": {
            "language": "python",
            "format": "docker-image",
            "version": "0.1",
            "container": {
                "entrypoint": "python -m pip install git+https://github.com/FELToken/feltoken.py/tree/ocean-integration; feltoken-train --model $ALGO --data test --output_model /data/outputs/result",
                "image": "oceanprotocol/algo_dockers",
                "tag": "python-branin",
            },
        },
        "files": [
            {
                "url": f"{os.getenv('FILE_PROVIDER_URL')}/model",
                "index": 0,
                "contentType": "text/text",
            }
        ],
        "name": "gpr",
        "author": "Trent",
        "license": "CC0",
        "dateCreated": "2020-01-28T10:55:11Z",
    }
}


def get_attributes(address):
    return {
        "main": {
            "name": "ALG_dataAssetAccessServiceAgreement",
            "creator": address,
            "timeout": 3600 * 24,
            "datePublished": "2020-01-28T10:55:11Z",
            "cost": 1.0,  # <don't change, this is obsolete>
        }
    }
