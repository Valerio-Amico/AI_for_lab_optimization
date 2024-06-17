import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def create_colormap(data_points, data_values, coords_to_plot=(0, 1), grid_size=100):
    """
    Creates a colormap of the cost versus two coordinates, interpolating points for smoothness.
    
    Parameters:
    - data_points: Array of shape (n, D) with n data points and D coordinates each.
    - data_values: Array of shape (n) with values corresponding to the data points.
    - coords_to_plot: Tuple of two integers specifying which coordinates to use for plotting.
    - grid_size: Integer specifying the size of the grid for interpolation.
    
    Returns:
    - None. Displays the colormap.
    """
    # Extract the two coordinates for plotting
    x = data_points[:, coords_to_plot[0]]
    y = data_points[:, coords_to_plot[1]]
    
    # Create a grid
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate data on the grid
    zi = griddata((x, y), data_values, (xi, yi), method='cubic')
    
    # Plot the colormap
    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, zi, levels=15, cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel(f'Coordinate {coords_to_plot[0]}')
    plt.ylabel(f'Coordinate {coords_to_plot[1]}')
    plt.title('Colormap of Cost vs Two Coordinates')
    plt.show()

def create_colormap_max(data_points, data_values, coords_to_plot=(0, 1), grid_size=100):
    """
    Creates a colormap of the cost versus two coordinates, interpolating points for smoothness.
    Maximizes the value over other parameters for each point in the grid.
    
    Parameters:
    - data_points: Array of shape (n, D) with n data points and D coordinates each.
    - data_values: Array of shape (n) with values corresponding to the data points.
    - coords_to_plot: Tuple of two integers specifying which coordinates to use for plotting.
    - grid_size: Integer specifying the size of the grid for interpolation.
    
    Returns:
    - None. Displays the colormap.
    """
    # Extract the two coordinates for plotting
    x = data_points[:, coords_to_plot[0]]
    y = data_points[:, coords_to_plot[1]]
    
    # Create a grid
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate data on the grid with maximization over other parameters
    zi = np.full(xi.shape, -np.inf)
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (np.abs(x - xi[i, j]) < (x.max() - x.min()) / grid_size) & (np.abs(y - yi[i, j]) < (y.max() - y.min()) / grid_size)
            if np.any(mask):
                zi[i, j] = np.max(data_values[mask])
    
    # Plot the colormap
    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, zi, levels=15, cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel(f'Coordinate {coords_to_plot[0]}')
    plt.ylabel(f'Coordinate {coords_to_plot[1]}')
    plt.title('Colormap of Cost vs Two Coordinates (Maximized Over Other Parameters)')
    plt.show()

# Example usage
n, D = 100, 4  # 100 data points, each with 3 coordinates
data_points = np.random.rand(n, D)
data_values = np.random.rand(n)

create_colormap(data_points, data_values)


create_colormap_max(data_points, data_values)
