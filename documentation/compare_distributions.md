# `plot_energy_distribution_heatmap` Function

The `plot_energy_distribution_heatmap` function is used to generate and plot the energy distribution heatmap for a given list of data. The function generates two types of plots:

1. A plot that bins the data according to the user-specified ranges (`rho_ranges`), which allows for custom binning.
2. A raw plot without binning.

## Parameters

- `dd_list` (list): A list of dictionaries where each dictionary represents a file's data. The keys in the dictionary should include 'density_rho_kev', 'etot_array_kev', and 'rho'. The 'density_rho_kev' value should be a 2D numpy array, 'etot_array_kev' is a 1D array that represents the total energy in keV, and 'rho' is a 1D array that represents the density values.

- `file_list` (list): A list of file names corresponding to the datasets in `dd_list`. This is used for plot titles.

- `rho_ranges` (list of tuples): Each tuple represents a bin's lower and upper boundaries, with the format (lower, upper).

- `pdf` (matplotlib.backends.backend_pdf.PdfPages object): An object to manage the output to a PDF file.

- `fn_profile` (str): Not used in the current function, but it's retained for potential future enhancements.

- `fn_geq` (str): Not used in the current function, but it's retained for potential future enhancements.

## Returns

This function does not return any value. Instead, it creates and saves the plots directly to the `pdf` object passed as a parameter.

## Example

Here's an example of how to use the function:

```python
rho_ranges = [(0., 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.75, 0.95)]
plot_energy_distribution_heatmap(dd_list, file_list, rho_ranges, pdf, fn_profile, fn_geq)
```


The above example will generate two types of plots for each file in file_list: a binned version and a raw version. The plots are saved to the pdf object.


Remember, for the function to work correctly, your data should be properly formatted, with the 'density_rho_kev' as a 2D numpy array, and 'etot_array_kev' and 'rho' as 1D arrays.
