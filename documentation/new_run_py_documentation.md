# new_run.py Documentation

This Python script provides functionality to manage a tree-like structure of run files. It allows you to add, delete, and view relationships between different runs, which are defined by their parent and child relationships. This is especially useful for managing complex project structures.

## Functions

- `print_tree(run_tree)`: This function prints the entire run tree. The `run_tree` parameter is a dictionary that represents the run tree.

- `print_subtree(run_tree, parent_run)`: This function prints a subtree rooted at a given parent run number. If the specified parent run does not exist in the tree, an error message will be printed.

- `add_run(run_tree, parent_run, child_run)`: This function adds a new run to the tree. It requires three parameters:
    - `run_tree`: The current run tree.
    - `parent_run`: The parent run number, for example, `7011`.
    - `child_run`: The child run number, for example, `7012`. If this argument is not provided, it will be automatically set to one more than the largest runID in the tree.
    The function first checks if the necessary files exist for the parent run. If they do not, an error message is printed, and the tree is not modified. If the files do exist, it creates new files for the child run by copying the parent files and making necessary adjustments. Finally, it adds the child run to the tree under the specified parent run.

- `delete_run(run_tree, run)`: This function deletes a run from the run tree. If the run does not exist in the tree, an error message will be printed. Before a run can be deleted, the user will be asked for confirmation.

## How to Use

1. Ensure you have Python installed and the necessary permissions to read/write files in your project directory.
2. Run the script via the command line, providing the necessary arguments depending on the action you wish to perform. Here are some examples:
    - To add a new run with specified child run number: `python new_run.py my_run_tree add_run 7011 7012`
    - To add a new run without specifying child run number (the child run number will be one more than the largest runID in the tree): `python new_run.py my_run_tree add_run 7011`
    - To print the entire tree: `python new_run.py my_run_tree print_tree`
    - To print a subtree: `python new_run.py my_run_tree print_tree 7011`
    - To delete a run: `python new_run.py my_run_tree delete_run 7011`
    When you try to delete a run, you will be asked for confirmation. If you agree, both the specified run and its associated files will be deleted.


## Notes

Hongyu's runs are documented in `run_tree.json`. When referring to it in the program, it should be just `run_tree`, without the suffix.