# Documentation for hz_utilities

## class `Timer`

``` hz_utilities.Timer() ```


- `checkpoint(text, display=True)`
  - First time called will initialize the timer. 
  - Then create checkpoint and document the time elapsed form the last checkpoint. 
  - If `display` is set to `True`, display the text at the checkpoint. 
  - Stores all the time at the checkpoint when called. 

- `display()`
  - Display all the times at the checkpoints. 



