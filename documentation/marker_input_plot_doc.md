# Documentation for marker_input_plot.py

``` python
plot_contour(x_data, y_data, contour_data, contour_levels, contour_colors,additional_points=None, additional_lines=None, xlim=None, ylim=None, xlabel='', ylabel='', title='', rasterized=True, pdf=None, additional_contour_data=None, additional_contour_levels=None, additional_contour_colors=None, additional_point_weights=None, addtional_point_weights_normalized=False, true_to_ratio=False, graph_label=None, cbar_label = None, additional_points_size=2, **kwargs)
```
- Parameters
  - `x_data`: the data to be plotted in the x direction
  - `y_data`: the data to be plotted in the y direction
  - `contour_data`: the data to be plotted in the z direction in countour
  - `contour_colors`: the colar you want your contour lines to show
  - `additional_points = None`: the possible markers you want to add to the $x-y$ plane. The format is a tuple of two arrays. For example `(R,z)` where `R` and `z` are `np.array`s with same dimension. 
  - `additional_lines`: when you want to plot add additional lines. Currently only suppor the last closed flux surface lines read from the geqdsk file. Format list of tuples with x and y arrays for the lines: `[(x_arr_line1, y_arr_line1),(x_arr_line2,y_arr_line2),...]`
  - `xlim = None`: set the x boundary for the plot in a list of two numbers: `[xmin, xmax]`
  - `ylim = None`: set the y boundary for the plot in a list of two numerbs: `[ymin, ymax]`
  - `xlabel = ''`: set the label for the x axis
  - `ylabel = ''`: set the label for the y axis
  - `rasterized = Ture`: set if the figure is rasterized
  - `pdf = None`: plot if the figure on a pdf if specified
  - `addtional_contour_data = None`: input the same format as `contour data` to plot extra contour plots. One use case is to plot the contour value outside of the last closed flux surface and with different color as set below. 
  - `addtional_contour_colors = None`: set the color for the `additional_contour_data` in the same format as the `contour_colors`. Often differnt than the `contour_data` for clearity
  - 

- First time called will initialize the timer. 
- Then create checkpoint and document the time elapsed form the last checkpoint. 
- If `display` is set to `True`, display the text at the checkpoint. 
- Stores all the time at the checkpoint when called. 

- `display()`
  - Display all the times at the checkpoints. 



