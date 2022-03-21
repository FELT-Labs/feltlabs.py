from numpy import genfromtxt


def load_data(path):
    """Load data and return them in (X, y) formta."""
    if path == "test":
        # Demo data for testing
        X, y = datasets.load_diabetes(return_X_y=True)
        subset = np.random.choice(X.shape[0], 100, replace=False)
        return X[subset], y[subset]
    else:
        try:
            data = genfromtxt(path, delimiter=",")
            return data[:-1], data[-1]

            X, y = load_data(args.data)
        except Exception as e:
            raise Exception(f"Unable to load {path}\n{e}")
