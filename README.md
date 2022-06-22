# feltoken.py
Python library for FELToken. It is available at [PyPI](https://pypi.org/project/feltoken/):
```bash
pip install feltoken
```


## Development
### Install
```
pip install -e .
pip install -r requirements-dev.txt
pre-commit install
```

### Test
```
pytest --cov=feltoken --cov-report=term
```

### Versioning
Do following steps when updating to new version:

1. Update version number in [`setup.py`](./setup.py)
2. Run following commands (replace `0.0.0` with correct version number):
   ```
   git add -A
   git commit -m "New version 0.0.0"
   git tag v0.0.0
   git push origin v0.0.0
   ```
3. Create new release on Github based on tag
