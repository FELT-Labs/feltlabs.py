"""FELToken python package intended for running data provider code.

This code connects to the specified smart contract and trains the specified
models on provided data.

Entry command:

```bash
felt-node-worker --chain <80001> --contract <address> --account main --data <data_path.csv>
```
"""
import re
from pathlib import Path
from urllib import request

from setuptools import find_packages, setup

PATH = Path(__file__).parent.absolute()
DOCLINES = (__doc__ or "").split("\n")

CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3 :: Only
Topic :: Scientific/Engineering
"""

# Download project contract artifacts
artifacts = PATH / "feltoken/artifacts"
artifacts.mkdir(parents=True, exist_ok=True)

# TODO: Add supported chains/remove contracts
remote_url = "https://raw.githubusercontent.com/FELToken/smart-contracts/main/build/deployments/80001/ProjectContract.json"
request.urlretrieve(remote_url, artifacts / "ProjectContract.json")


def parse_requirements(file_name):
    """
    from:
        http://cburgmer.posterous.com/pip-requirementstxt-and-setuppy
    """
    requirements = []
    with open(file_name, "r") as f:
        for line in f:
            if re.match(r"(\s*#)|(\s*$)", line):
                continue
            if re.match(r"\s*-e\s+", line):
                requirements.append(
                    re.sub(r"\s*-e\s+.*#egg=(.*)$", r"\1", line).strip()
                )
            elif re.match(r"\s*-f\s+", line):
                pass
            else:
                requirements.append(line.strip())
    return requirements


requirements = parse_requirements(PATH / "requirements.txt")


setup(
    name="feltoken",
    version="0.2.0",
    packages=find_packages(),
    maintainer="FELToken",
    maintainer_email="info@bretahajek.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    keywords=["Federated Learning", "Web3", "Machine Learning"],
    url="https://feltoken.ai/",
    author="FELToken",
    project_urls={
        "Bug Tracker": "https://github.com/FELToken/feltoken.py/issues",
        "Documentation": "https://docs.feltoken.ai/",
        "Source Code": "https://github.com/FELToken/feltoken.py",
    },
    license="GPL-3.0 License",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    python_requires=">=3.8",
    install_requires=requirements,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "felt-node-worker = feltoken.node.background_worker:main",
            "feltoken-train = feltoken.node.training:main",
        ],
    },
    package_data={
        "feltoken": ["artifacts/*.json", "artifacts/contracts/*.json"],
    },
)
