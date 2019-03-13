### \_get\_parser
```python
def _get_parser()
```

### parse\_args
```python
def parse_args()
```
Parse CLI arguments.



### get\_config\_from\_weights
```python
def get_config_from_weights(w_path, arch_params, name)
```
Extracts architecture parameters from the file name of the weights. Only works with standardized weights name.



### select\_option
```python
def select_option(options, message, val)
```
CLI selection given options.



### select\_positive
```python
def select_positive(message, value)
```
CLI non-negative integer selection.



### browse\_weights
```python
def browse_weights(weights_dir, model)
```
Weights selection from cl.



### setup
```python
def setup(config_file, default, training, prediction)
```
CLI interface to set up the training or prediction session.

Takes as input the configuration file path (minus the '.py' extection) and arguments parse from CLI.

### select\_dataset
```python
def select_dataset(session_type, conf)
```
CLI snippet for selection the dataset for training.



