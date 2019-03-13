### PSNR
```python
def PSNR(y_true, y_pred, MAXp)
```
Evaluates the PSNR value: PSNR = 20 * log10(MAXp) - 10 * log10(MSE).


##### Args
* **y_true**: ground truth.

* **y_pred**: predicted value.

* **MAXp**: maximum value of the pixel range (default=1).


