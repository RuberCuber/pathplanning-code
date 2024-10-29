import os
os.environ["QT_API"] = "pyqt5"

import my_utils
import grid_utils
import search_utils

import copy
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from qtpy import QtWidgets
import time

def translate_points(direction, points):
    for point in points:
        point += direction
    return points

PHI = (1 + np.sqrt(5))/2 #phi
DA_ICO = np.arccos((5**0.5)/3) #dihedral angle of icosahedron
IA_TRI = np.pi/3 #interior angle of triangle
CIR_RAD = 1 #circumscribed sphere radius
ICO_EDGE_LEN = 2*CIR_RAD/(np.sqrt(PHI*np.sqrt(5))) #icosahedron edge length

#vertices of icosahedron
points = np.matrix([
    [PHI,1,0],
    [PHI,-1,0],
    [-PHI,-1,0],
    [-PHI,1,0],
    [1,0,PHI],
    [-1,0,PHI],
    [-1,0,-PHI],
    [1,0,-PHI],
    [0,PHI,1],
    [0,PHI,-1],
    [0,-PHI,-1],
    [0,-PHI,1]
])

grid_width = 20
grid_height = 20

start = (2,2)
#end = (6,6)
end = (15,17)

neighbour_list = []

#wall blocker coords
wall_coords = [
    (0,5),
    (1,5),
    (2,5),
    (3,5),
    (4,5),
    (5,5),
    (6,5),
    (7,5),
    (8,5),
    (9,5),
    (10,5),
    (11,5),
    (12,5),
    (13,5),
    (14,5),
    (15,5),
    (16,5),
    (5,9),
    (6,9),
    (7,9),
    (8,9),
    (9,9),
    (10,9),
    (11,9),
    (12,9),
    (13,9),
    (14,9),
    (15,9),
    (16,9),
    (17,9),
    (18,9),
    (19,9),
    (10,13),
    (11,13),
    (12,13),
    (13,13),
    (14,13),
    (15,13),
    (16,13),
]

#starting ico
ico = pv.Icosahedron(radius=1.098185,center=(0,0,0))

#rotate to starting orientation
ico.points = my_utils.points_rotate_about_a_line([0,0,0], [0,1,0], DA_ICO/2, ico.points)
ico.points = my_utils.points_rotate_about_a_line([0,0,0], [0,0,1], IA_TRI*1.5, ico.points)
ico.points = np.around(ico.points, 4)

#create triangle grid and hex grid and get neighbour list
mesh, tri_grid, neighbour_list = grid_utils.create_grid(grid_width, grid_height, wall_coords)
mesh.points = np.around(mesh.points, 4)

#list of tuples containing all hex grid points as tuple coordinates ((0,0), (4,8) etc)
mesh_coords = grid_utils.list_labels_to_tuples(mesh["labels"])

#initial transformations
ico_pos = grid_utils.label2point(start)
ico.points = translate_points([ico_pos[0],ico_pos[1],0], ico.points)

#starting frame
ico_frame = my_utils.world_frame()
ico_frame[0, 3] = ico_pos[0]
ico_frame[1, 3] = ico_pos[1]

#trender frame
ico_frame_render = my_utils.world_frame()
ico_frame_render[0, 3] = ico_pos[0]
ico_frame_render[1, 3] = ico_pos[1]

#target frame
target_frame = my_utils.world_frame()
target_point = grid_utils.label2point(end)
target_frame[0][3] = target_point[0]
target_frame[1][3] = target_point[1]

#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (2,1))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (2,2))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (3,2))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (4,2))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (4,1))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (3,1))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (2,1))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (2,2))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (1,2))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (0,2))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (0,1))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (1,1))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (2,1))
#virtual_frame_pos, target_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(grid_utils.get_frame_label(target_frame)), target_frame, (2,2))

end = grid_utils.get_frame_label(target_frame)

ico_start = copy.deepcopy(ico)
ico_render = copy.deepcopy(ico)
ico_pos_render = copy.deepcopy(ico_pos)


print("Starting frame: \n{}".format(np.around(ico_frame,3)))
print("Target frame: \n{}".format(np.around(target_frame,3)))


do_coarse = True
do_fine = True
do_rot = True

path_coarse = []
path_fine = []
path_rot = []

if do_coarse:
    #start coarse position pathfinding
    print("Starting coarse pathing")

    start_time = time.time()
    visited_list, target_pos = search_utils.astar_position(start, end, neighbour_list)
    end_time = time.time()

    print("Calculation Time: {} seconds".format(end_time-start_time))
    print("Visited List Coarse Length: {}".format(len(visited_list)))

    path_coarse = search_utils.GetPathPos(start, target_pos, visited_list)[:-5]
    #loop through path
    print("coarse path: {}".format(path_coarse))
    for pos in path_coarse[1:]:
        ico_pos, ico_frame = grid_utils.perform_virtual_roll(ico_pos, ico_frame, pos)

if do_fine:
    #start fine position pathfinding to knight location
    print("Starting fine pathing")

    start_time = time.time()
    visited_list, end_frame = search_utils.astar(ico_frame, target_frame, neighbour_list, mode = "Knight")
    end_time = time.time()

    print("Calculation Time: {} seconds".format(end_time-start_time))
    print("Visited List Fine Length: {}".format(len(visited_list)))

    path_fine = search_utils.GetPath(ico_frame, end_frame, visited_list)
    #loop through path
    print("fine path: {}".format(path_fine))
    for pos in path_fine[1:]:
        ico_pos, ico_frame = grid_utils.perform_virtual_roll(ico_pos, ico_frame, pos)

if grid_utils.point2label(ico_pos) != end and do_rot:
    #start rotation pathfinding
    print("Starting rotation pathing")

    start_time = time.time()
    visited_list, end_frame = search_utils.astar(ico_frame, target_frame, neighbour_list, mode = "Default")
    end_time = time.time()

    print("Calculation Time: {} seconds".format(end_time-start_time))
    print("Visited List Rotation Length: {}".format(len(visited_list)))

    path_rot = search_utils.GetPath(ico_frame, end_frame, visited_list)
    print("rot path: {}".format(path_rot))


path_final = path_coarse[:-1] + path_fine[:-1] + path_rot[:-1] + [end]

#plotter
plotter = BackgroundPlotter(show=True)
plotter.open_gif("pathfinding_animation.gif", fps=4)


grid_utils.plot_grid(mesh, plotter, tri_grid, show_labels=False)
grid_utils.plot_walls(wall_coords, plotter)

my_utils.plot_ico(ico_start, plotter)
my_utils.plot_frame(ico_frame_render, plotter)

my_utils.plot_ico(ico_render, plotter)
e1, e2, e3, a1, a2, a3 = my_utils.plot_frame(target_frame, plotter)

#loop through path
print("path final: {}".format(path_final))
print("path length: {}".format(len(path_final)))
for pos in path_final[1:]:
    ico_pos_render, ico_render.points, ico_frame_render = grid_utils.perform_roll(mesh, ico_pos_render, ico_render.points, ico_frame_render, plotter, direction=None, position=pos)
    plotter.remove_actor(a1)
    plotter.remove_actor(a2)
    plotter.remove_actor(a3)
    e1, e2, e3, a1, a2, a3 = my_utils.plot_frame(ico_frame_render, plotter)
    plotter.write_frame()
    #
    #my_utils.plot_ico(ico, plotter)
    #time.sleep(0.2)
my_utils.plot_frame(ico_frame_render, plotter)
print("Ending frame: \n{}".format(np.around(ico_frame_render, 3)))

#stay on last frame for a while
for i in range(8):
    plotter.write_frame()

#plotter.close()
#exit()

plotter.app.exec_()
