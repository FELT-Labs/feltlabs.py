"""FELT python package intended for running federated learning on Ocean protocol.

This code is intended to work closely with Ocean protocol. Algorithms from this code
should run on ocean provider. Training local models and aggregating them into global
model.

Entry commands:

```bash
felt-train
felt-aggregate
felt-export
```

## Common Usage

After installing this library you can load models trained using FELT as:
```python
from feltlabs.model import load_model

# Load scikit-learn model
model = load_model("final-model.json")

# Data shape must be: (number_of_samples, number_of_features)
data = [
  [1980, 2, 2, 2, 0, 0],
  [1700, 3, 2, 3, 1, 1],
  [2100, 3, 2, 3, 1, 0],
]

result = model.predict(data)
print(result)
# Use following line for analytics as mean, std...
# result = model.predict(None)
```

### Command: felt-export
You can use `felt-export` for converting trained models into pickle object:
Resulting file will then contain a pickled object of scikit-learn model.

```bash
felt-export --input "final-model-House Prices.json" --output "model.pkl"
```

Then you can use the created file as follows:

```python
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(object, f)
    
# See the above code example for data definition
model.predict(data)
```
"""
import re
from pathlib import Path

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
    name="feltlabs",
    version="0.5.1",
    packages=find_packages(),
    maintainer="FELT Labs",
    maintainer_email="info@bretahajek.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    keywords=["Federated Learning", "Web3", "Machine Learning"],
    url="https://feltlabs.ai/",
    author="FELT Labs",
    project_urls={
        "Bug Tracker": "https://github.com/FELT-Labs/feltlabs.py/issues",
        "Documentation": "https://docs.feltlabs.ai/",
        "Source Code": "https://github.com/FELT-Labs/feltlabs.py",
    },
    license="GPL-3.0 License",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    python_requires=">=3.8",
    install_requires=requirements,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "felt-train = feltlabs.algorithm.train:main",
            "felt-aggregate = feltlabs.algorithm.aggregate:main",
            "felt-export = feltlabs.model:main",
        ],
    },
)
