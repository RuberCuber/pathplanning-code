import sys

# Setting the Qt bindings for QtPy
import os
os.environ["QT_API"] = "pyqt5"

import numpy as np
import pyvista as pv
from qtpy import QtWidgets
from pyvistaqt import MainWindow, BackgroundPlotter
import abc

import my_utils
#import PH_curves



class BaseWindow(MainWindow):
  def __init__(self, parent=None, show=True): 
    QtWidgets.QMainWindow.__init__(self, parent)

    # create the frame
    self.frame = QtWidgets.QFrame()
    vlayout = QtWidgets.QVBoxLayout()
    self.plotter = BackgroundPlotter(show=True)     # add the pyvista interactor

  def run(self, **kwargs):
    # world frame (W-eiejek)
    self.world_frame = my_utils.world_frame()

    # start body frame (B-exeyez) w.r.t. the world frame
    self.compute_start_frame()
    self.compute_goal_frame()
    self.compute_contact_points()
    self.compute_goal_e1_dydx1()
    self.plot_arena()
    self.compute_ph_curve()
    self.generate_meshes_need_updating()

    if kwargs['rolling_on_path']:
      self.rolling_on_path()
    else:
      print(f"kwargs[rolling_on_path]: {kwargs['rolling_on_path']}")

    self.curr_frame = np.identity(4)  # for animations
    self.next_frame = np.identity(4)
    self.index = 0

    if kwargs['callback']:
      self.call_back(interval = 200, count = len(self.path_points))

  def call_back(self, interval, count):  # interval: time interval, count: number of times
    self.plotter.add_callback(self.update, interval=interval, count = count)

  def update(self):
    if self.path_points is None:
      self.plotter.update()
      return;
      
    if self.index < len(self.path_points-1):
      self.curr_frame = my_utils.homogeneous_frame(
        self.B_arr[self.index], 
        self.ex_arr[self.index], 
        self.ey_arr[self.index], 
        self.ez_arr[self.index]
      )

      self.next_frame = my_utils.homogeneous_frame(
        self.B_arr[self.index+1], 
        self.ex_arr[self.index+1], 
        self.ey_arr[self.index+1], 
        self.ez_arr[self.index+1]
      ) 

      self.lamina_mesh.points = my_utils.update_points(
          self.curr_frame, self.next_frame, self.lamina_mesh.points)

      self.grid_object.points = my_utils.update_points(
          self.curr_frame, self.next_frame, self.grid_object.points)

      self.ex_mesh.points = np.vstack((
          self.B_arr[self.index+1], 
          self.B_arr[self.index+1] + self.ex_arr[self.index+1]))

      self.ey_mesh.points = np.vstack((
          self.B_arr[self.index+1], 
          self.B_arr[self.index+1] + self.ey_arr[self.index+1]))

      self.ez_mesh.points = np.vstack((
          self.B_arr[self.index+1], 
          self.B_arr[self.index+1] + self.ez_arr[self.index+1]))

      my_utils.plot_point(self.M_arr[self.index + 1], self.plotter)

    self.plotter.update()
    self.index += 1

  @abc.abstractmethod
  def compute_start_frame(self):
    pass
  
  @abc.abstractmethod
  def compute_goal_frame(self):
    pass
  
  @abc.abstractmethod
  def compute_contact_points(self):
    pass
  
  @abc.abstractmethod
  def compute_goal_e1_dydx1(self):
    pass 

  @abc.abstractmethod
  def plot_arena(self):
    pass
  
  @abc.abstractmethod
  def compute_ph_curve(self):
    pass

  @abc.abstractmethod
  def generate_meshes_need_updating(self):
    pass
  
  @abc.abstractmethod
  def rolling_on_path(self):
    pass


