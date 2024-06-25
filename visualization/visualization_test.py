import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QLabel, QHBoxLayout, QGridLayout, QDoubleSpinBox
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
        self.param_combos_kn = []
        self.param_doubespin_rmax = []

        for i in range(3):
            label_x = QLabel(f"Plot {i+1}", self)
            control_layout.addWidget(label_x)

            combo_x = QComboBox(self)
            control_layout.addWidget(combo_x)

            combo_y = QComboBox(self)
            control_layout.addWidget(combo_y)

            label_kn = QLabel(f"select k-neibours", self)
            control_layout.addWidget(label_kn)
            combo_kn = QComboBox(self)
            control_layout.addWidget(combo_kn)

            label_rmax = QLabel(f"insert max radius", self)
            control_layout.addWidget(label_rmax)
            double_rmax = QDoubleSpinBox(self)
            double_rmax.setValue(1)
            control_layout.addWidget(double_rmax)

            control_layout.addWidget(Color('white'))

            self.param_labels.append((label_x))
            self.param_combos_x.append(combo_x)
            self.param_combos_y.append(combo_y)
            self.param_combos_kn.append(combo_kn)
            self.param_doubespin_rmax.append(double_rmax)

        main_layout.addWidget(control_panel)

        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Load data
        self.data = read_data('mydata_100.json')
        self.parameters = list(self.data["parameters"].keys())
        self.cost = np.array(self.data["results"]['cost'])
        self.times = np.array(self.data['time'])
        self.n_datapoints = len(self.cost)

        # Populate combo boxes
        for combo_x, combo_y, combo_kn, double_rmax in zip(self.param_combos_x, self.param_combos_y, self.param_combos_kn, self.param_doubespin_rmax):
            combo_x.addItems(self.parameters)
            combo_y.addItems(self.parameters)
            combo_kn.addItems([str(k) for k in np.arange(1, 10, step=1)])

        # Connect signals to slot
        for combo_x, combo_y, combo_kn, double_rmax in zip(self.param_combos_x, self.param_combos_y, self.param_combos_kn, self.param_doubespin_rmax):
            combo_x.currentIndexChanged.connect(self.update_plot)
            combo_y.currentIndexChanged.connect(self.update_plot)
            combo_kn.currentIndexChanged.connect(self.update_plot)
            double_rmax.valueChanged.connect(self.update_plot)

        self.plot()

    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def plot(self):
        self.figure.clear()
        axes = [self.figure.add_subplot(2, 2, i+1) for i in range(4)]

        for i, (combo_x, combo_y, combo_kn, double_rmax, ax) in enumerate(zip(self.param_combos_x, self.param_combos_y, self.param_combos_kn, self.param_doubespin_rmax, axes[0:-1])):
            self.imview_single_axis(combo_x, combo_y, combo_kn, double_rmax, ax)
            
        data = []
        for par in self.parameters:
            data.append(self.data["parameters"][par])
        data = np.array(data).T
        ax_time = axes[-1]
        distance_by_step = np.array([self.distance(data[i,:], data[i+1,:]) for i in range(self.n_datapoints-1)])
        ax_time.plot(self.times[0:-1], distance_by_step, label="distance from previous point")
        best_index = np.argmax(self.cost)
        distance_from_best = np.array([self.distance(data[i,:], data[best_index,:]) for i in range(self.n_datapoints)])
        ax_time.plot(self.times, distance_from_best, label="distance from best point")
        ax_time.legend()
        ax_time.set_xlabel("time")
        ax_time.set_ylabel("euclidean distance")

        self.canvas.draw()

    def imview_single_axis(self, combo_x, combo_y, combo_kn, double_rmax, ax):
        param_x = combo_x.currentText()
        param_y = combo_y.currentText()
        param_kn = combo_kn.currentText()
        param_rmax = double_rmax.value()

        if param_x and param_y:
            ax.autoscale(True)
            parameter_x = self.data["parameters"][param_x]
            parameter_y = self.data["parameters"][param_y]
            parameter_kn = int(param_kn)
            parameter_rmax = float(param_rmax)

            scatter = ax.scatter(parameter_x, parameter_y, c=self.cost, cmap='viridis', s=0)
            #scatter = ax.scatter(parameter_x, parameter_y, c="red", cmap='viridis', s=2)
            self.figure.colorbar(scatter, ax=ax, label='Cost')
            ax.set_xlabel(param_x)
            ax.set_ylabel(param_y)

            # create the background colormap
            k_nearest = parameter_kn
            radius_ball = parameter_rmax
            N_size_imview = 300

            enl_x, enl_y = (max(parameter_x) - min(parameter_x))*0.1, (max(parameter_y) - min(parameter_y))*0.1
            extent=[min(parameter_x)-enl_x,max(parameter_x)+enl_x,min(parameter_y)-enl_y,max(parameter_y)+enl_y]
            x, y = np.meshgrid(np.linspace(extent[0], extent[1], N_size_imview), np.linspace(extent[3], extent[2], N_size_imview), indexing="xy")
            xy = np.array([x.flatten(), y.flatten()]).T

            if k_nearest == 1:
                XY = np.array([np.array(parameter_x), np.array(parameter_y)]).T
                resampled = griddata(XY, self.cost, xy, method='nearest').reshape([N_size_imview,N_size_imview])
            elif k_nearest > 1:
                # prepare KDTree
                _KDTree = KDTree(np.array([parameter_x, parameter_y]).T)
                
                _, indexes = _KDTree.query(xy, k=k_nearest, p=2, distance_upper_bound=radius_ball)
                
                #resampled = np.zeros(N_size_imview**2)
                
                #for i, indexes_in_ball in enumerate(indexes):
                    
                    #indexes_in_ball = list(filter(lambda a: a != self.n_datapoints, list(indexes_in_ball)))
                    
                    #if list(indexes_in_ball) == []:
                    #    resampled[i] = np.nan
                    #else:
                self._cost_aus_nan = np.array(list(self.cost) + [np.nan])
                #resampled = np.nanmax(self._cost_aus_nan[indexes], axis=1).reshape([N_size_imview,N_size_imview])
                weigths = np.linspace(0,1,k_nearest)
                print(self._cost_aus_nan[indexes])
                print(np.sort(self._cost_aus_nan[indexes], axis = 1))
                resampled = np.average(np.sort(self._cost_aus_nan[indexes], axis = 1), axis=1, weights=weigths).reshape([N_size_imview,N_size_imview])
                
                #resampled = resampled.reshape([N_size_imview,N_size_imview])
                # else:
                #     resampled = np.max(self.cost[indexes], axis=1).reshape([N_size_imview, N_size_imview])
            ax.imshow(resampled, extent=extent, interpolation="lanczos", aspect="auto") 
                
    def update_plot(self):
        self.plot()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
