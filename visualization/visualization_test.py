import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QLabel, QHBoxLayout, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QPalette, QColor
from scipy.interpolate import griddata
import numpy as np
from scipy.spatial import KDTree

def read_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Color Plot Visualization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_layout = QHBoxLayout()

        # Left side: Plot area
        plot_widget = QWidget(self)
        plot_layout = QVBoxLayout()
        plot_widget.setLayout(plot_layout)
        
        self.figure = Figure(figsize=(10,8))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        main_layout.addWidget(plot_widget)

        # Right side: Control panel
        control_panel = QWidget(self)
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        self.param_labels = []
        self.param_combos_x = []
        self.param_combos_y = []

        for i in range(4):
            label_x = QLabel(f"Plot {i+1}", self)
            control_layout.addWidget(label_x)
            combo_x = QComboBox(self)
            control_layout.addWidget(combo_x)

            # label_y = QLabel(f"{i+1}, Y-axis:", self)
            # control_layout.addWidget(label_y)
            combo_y = QComboBox(self)
            control_layout.addWidget(combo_y)

            control_layout.addWidget(Color('white'))

            self.param_labels.append((label_x))
            self.param_combos_x.append(combo_x)
            self.param_combos_y.append(combo_y)

        main_layout.addWidget(control_panel)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Load data
        self.data = read_data('mydata.json')
        self.parameters = list(self.data["parameters"].keys())

        # Populate combo boxes
        for combo_x, combo_y in zip(self.param_combos_x, self.param_combos_y):
            combo_x.addItems(self.parameters)
            combo_y.addItems(self.parameters)

        # Connect signals to slot
        for combo_x, combo_y in zip(self.param_combos_x, self.param_combos_y):
            combo_x.currentIndexChanged.connect(self.update_plot)
            combo_y.currentIndexChanged.connect(self.update_plot)

        self.plot()


    def plot(self):
        self.figure.clear()
        axes = [self.figure.add_subplot(2, 2, i+1) for i in range(4)]

        for i, (combo_x, combo_y, ax) in enumerate(zip(self.param_combos_x, self.param_combos_y, axes)):
            param_x = combo_x.currentText()
            param_y = combo_y.currentText()

            if param_x and param_y:
                ax.autoscale(True)
                parameter_x = self.data["parameters"][param_x]
                parameter_y = self.data["parameters"][param_y]
                cost = np.array(self.data["results"]['cost'])

                scatter = ax.scatter(parameter_x, parameter_y, c=cost, cmap='viridis', s=4)
                self.figure.colorbar(scatter, ax=ax, label='Cost')
                ax.set_xlabel(param_x)
                ax.set_ylabel(param_y)

                # create the begraund colormap
                k_nearest = 40
                N_size_imview = 30

                enl_x, enl_y = (max(parameter_x) - min(parameter_x))*0.1, (max(parameter_y) - min(parameter_y))*0.1
                extent=[min(parameter_x)-enl_x,max(parameter_x)+enl_x,min(parameter_y)-enl_y,max(parameter_y)+enl_y]
                x, y = np.meshgrid(np.linspace(extent[0], extent[1], N_size_imview), np.linspace(extent[3], extent[2], N_size_imview), indexing="xy")
                xy = np.array([x.flatten(), y.flatten()]).T

                if k_nearest == 1:
                    XY = np.array([np.array(parameter_x), np.array(parameter_y)]).T
                    resampled = griddata(XY, np.array(cost), xy, method='nearest').reshape([N_size_imview,N_size_imview])
                elif k_nearest > 1:
                    # prepare KDTree
                    _KDTree = KDTree(np.array([parameter_x, parameter_y]).T)
                    
                    _, indexes = _KDTree.query(xy, k_nearest)
                    resampled = np.max(cost[indexes], axis=1).reshape([N_size_imview, N_size_imview])

                ax.imshow(resampled, extent=extent, interpolation="lanczos", aspect="auto") 
                

        
        self.canvas.draw()

    def update_plot(self):
        self.plot()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
