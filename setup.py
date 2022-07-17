"""FELT python package intended for running federated learning on Ocean protocol.

This code is intended to work closely with Ocean protocol. Algorithms from this code
should run on ocean provider. Training local models and aggregating them into global
model.

Entry commands:

```bash
felt-train
felt-aggregate
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
    version="0.2.5",
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
