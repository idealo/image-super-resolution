## class Predictor
The predictor class handles prediction, given an input model.

Reads input files from the folder specified in config.json. Saves results in output folder specified in config.json.

Can receive a path for the weights or can let the user browse through the weights directory for the desired weights.
### \_\_init\_\_
```python
def __init__(input_dir, output_dir)
```

### get\_predictions
```python
def get_predictions(model, weights_path)
```
Runs the prediction.



