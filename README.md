# feltoken.py
Python library for FELToken. It is available at [PyPI](https://pypi.org/project/feltoken/):
```bash
pip install feltoken
```


# Ocean demo flow
1. First run demo ocean instance `barge` as documented here:
    - <https://github.com/oceanprotocol/ocean.py/blob/main/READMEs/c2d-flow.md#run-barge-services>
    - Run `git checkout v3` for the barge repository (for now)
    - Use account from barge by setting `PRIVATE_KEY` in `.env` to `0x5d75837394b078ce97bc289fa8d75e21000573520bfa7784a9d28ccaae602b`

2. Then run file provider server (for exchanging models):
    ```bash
    uvicorn feltoken.ocean.file_provider:app --reload
    ```
    Tunnel file provider to public (for example using ngrok):
    ```bash
    ngrok http 8000
    ```
    In `.env` update `FILE_PROVIDER_URL` with public file provider URL.

3. Publish dataset. Copy `data_did` provided in terminal for later use:
    ```
    python feltoken/ocean/dataset.py
    ```

4. Publish algorithm. Copy `alg_did` provided in terminal for later use:
    ```bash
    python feltoken/ocean/algorithm.py
    ```

5. Allow algorithm for dataset. Replace `data_did` and `alg_did` with previously obtained values:
    ```bash
    python feltoken/ocean/allow_algorithm.py <data_did> <alg_did>
    ```

6. Register node into smart-contract as normal and run node as:
    ```bash
    felt-node-worker --chain <id> --contract <address> --data <data_did> --algorithm_did <alg_did> --ocean
    ```




