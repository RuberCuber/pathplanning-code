from collections import deque, defaultdict
from typing import Tuple
import numpy as np
import pyvista as pv
import my_utils

Location = Tuple[int, int]
PHI = (1 + np.sqrt(5))/2 #phi
CIR_RAD = 1 #circumscribed sphere radius
ICO_EDGE_LEN = 2*CIR_RAD/(np.sqrt(PHI*np.sqrt(5))) #icosahedron edge length
PLANE_OFFSET = -PHI/2 - 0.0636606
DA_ICO = np.arccos((5**0.5)/3)

class TriGrid:
    def __init__(self, width, height) -> None:
        self.neighbour_list = defaultdict(list)
        self.width = width
        self.height = height

        self.PopulateNeighbours()

    #add a new neighbour to the neighbour list
    def add_neighbour(self, cur: Location, new: Location):
        self.neighbour_list[cur].append(new)

    #populate neighbour array
    def PopulateNeighbours(self):
        for j in range(self.height):
            for i in range(self.width):
                cur: Location = (i, j)
                new_list = []

                if (i % 2 == 0 and j % 2 == 0):
                    new_list = [
                        (i, j-1),
                        (i+1, j),
                        (i-1, j)
                    ]
                elif (i % 2 == 1 and j % 2 == 1):
                    new_list = [
                        (i, j-1),
                        (i+1, j),
                        (i-1, j)
                    ]

                #check if each position is in bounds, then add to neighbour list
                for pos in new_list:
                    if self.inBounds(pos) and self.inBounds(cur):
                        self.add_neighbour(cur, pos)


    #check if position is in bounds
    def inBounds(self, pos: Location):
        (x, y) = pos
        if x in range(0, self.width) and y in range(0, self.height):
            return True
        return False

    #create array of points, dependant on width and height, and use neighbour array to create lines
    def CreatePointArr(self):
        point_arr = np.full((self.height, self.width, 2), 0)
        line_arr = np.array([], dtype=int)

        for j in range(self.height):
            for i in range(self.width):
                pos = (i, j)
                
                point_arr[j,i] = pos
        
        #reshape into polydata format
        point_arr = np.reshape(point_arr, (-1, 2))

        p_index_visited = []

        for idx, point in enumerate(point_arr):
            p_index = idx
            for neighbour in self.neighbour_list[tuple(point)]:
                n_index = point_arr.tolist().index(list(neighbour))
                if p_index not in p_index_visited:
                    #add lines in polydata format
                    line_arr = np.append(line_arr, [2, p_index, n_index])
            
            p_index_visited.append(p_index)
        return point_arr, line_arr

#check if position is in bounds
def inBounds(pos, width, height):
    (x, y) = pos
    if x in range(0, width) and y in range(0, height):
        return True
    return False

def append_neighbours(neighbour_list, pos, width, height, walls):
    (x, y) = pos
    new_list = [
        (x+1, y),
        (x-1, y)
    ]
    if (x+y) % 2 == 0:
        new_list.append((x, y-1))
    else:
        new_list.append((x, y+1))

    #check if each position is in bounds, then add to neighbour list
    for loc in new_list:
        if inBounds(loc, width, height) and inBounds(pos, width, height) and ((x,y) not in walls) and (loc not in walls):
            neighbour_list[tuple(pos)].append(loc)
    return neighbour_list

#create mesh of hex vertexes using pyvista
def create_mesh(points, lines, width, height, walls):
    v_arr = []
    neighbour_list = defaultdict(list)

    for pos in points:
        (x, y) = pos

        if (x, y) not in walls:
            neighbour_list = append_neighbours(neighbour_list, pos, width, height, walls)

        vert_y = y + 1/3 -((x+y) % 2 == 0)*(1/3)

        vertex = np.array([x, vert_y, PLANE_OFFSET])
        v_arr = np.append(v_arr, vertex)
    v_arr = np.reshape(v_arr, (-1, 3))

    mesh = pv.PolyData(var_inp=v_arr, lines=lines)
    mesh["labels"] = [f"({i}, {j})" for (i, j) in points]

    return mesh, neighbour_list

#convert grid to equilateral triangle grid
def TriGridPoints(points):
    new_points = []
    #line_arr = np.array([], dtype=float)
    for pos in points:
        (x, y) = pos
        h = 1
        w = 1
        new_y = y*h
        new_x = x*w

        if (x+y) % 2 == 1:
            new_y_mod = 1
        else:
            new_y_mod = -1
        
        tri_points = [
            (new_x, new_y-(new_y_mod*h/2)+1/6),
            (new_x-w, new_y+(new_y_mod*h/2)+1/6),
            (new_x+w, new_y+(new_y_mod*h/2)+1/6)
        ]
        
        for point in tri_points:
            if len(new_points) == 0 or point not in new_points:
                new_points.append(point)
    return new_points

#converts list of tuples into polydata format
def TupleToPv(tup_arr):
    new_arr = np.array([], dtype=float)
    for pos in tup_arr:
        (x, y) = pos
        new_arr = np.append(new_arr, [x, y, PLANE_OFFSET])
    return np.reshape(new_arr, (-1, 3))

def init_grid(width, height):
    #initialise grid
    grid = TriGrid(width=width, height=height)
    return grid

def hex_mesh(grid, width, height, walls):
    #hexagon travel grid mesh
    point_arr, line_arr = grid.CreatePointArr()
    mesh, neighbour_list = create_mesh(point_arr, line_arr, width, height, walls)
    mesh.points[:, 0] *= np.sqrt(3)/3
    return point_arr, mesh, neighbour_list

def tri_grid(point_arr):
    #triangle grid, for visualisation only
    tri_points = TriGridPoints(point_arr)
    tri_points_pv = pv.PolyData(TupleToPv(tri_points))
    tri_plane = tri_points_pv.delaunay_2d()
    tri_plane.points[:, 0] *= np.sqrt(3)/3
    #print(f"tri mesh points: {tri_points_pv}")
    return tri_plane

def create_grid(width, height, walls):
    grid = init_grid(width, height)
    point_arr, mesh, neighbour_list = hex_mesh(grid, width, height, walls)
    tri_plane = tri_grid(point_arr)
    
    return mesh, tri_plane, neighbour_list

def plot_grid(mesh, plotter, tri_plane, show_labels=False):
    plotter.add_mesh(mesh, point_size=10, show_edges=True, line_width=2, color="blue")
    if show_labels:
        plotter.add_point_labels(mesh, "labels", font_size=20)
    plotter.add_mesh(tri_plane, show_edges=True)
    return

def get_grid_neighbour_points(point):
    [x,y,z] = point
    a = np.sqrt(3)/3
    b = 1/3
    neighbour_arr = [
        [x+a,y+b,z],
        [x-a,y+b,z],
        [x,y-(2*b),z]
    ]
    '''
    for neighbour in neighbour_arr:
        if point not in mesh_points:
            neighbour_arr = np.delete(neighbour_arr, np.where(neighbour_arr==neighbour)[0][0])
    '''
    return np.around(neighbour_arr, 4)

#get rotation line vector between 2 adjacent grid triangle positions
def get_rotation_vector(start, end):
    vect = end - start
    perp_vect = [vect[1], -vect[0], vect[2]]
    midpoint = [(start[0]+end[0])/2, (start[1]+end[1])/2, (start[2]+end[2])/2]
    return midpoint, perp_vect


def perform_roll(mesh, start, ico, frame, plotter, direction=None, position=None):
    if direction != None:
        neighbour = get_grid_neighbour_points(start)[direction]
    elif position != None:
        neighbour = label2point(position)

    vect_point, vect_dir = get_rotation_vector(start, neighbour)
    #my_utils.plot_arrow(vect_point, vect_dir, plotter)
    #my_utils.plot_arrow(start, [0,0,-1], plotter)
    #my_utils.plot_arrow(neighbour, [0,0,1], plotter)
    my_utils.plot_arrow(start, neighbour-start, plotter)
    points = my_utils.points_rotate_about_a_line(vect_point, vect_dir, -DA_ICO, ico)
    frame = my_utils.frame_rotate_about_a_line(vect_point, vect_dir, -DA_ICO, frame)
    return neighbour, points, frame

#used for pathplanning
def perform_virtual_roll(start, frame, position):
    neighbour = label2point(position)
    vect_point, vect_dir = get_rotation_vector(start, neighbour)
    frame = my_utils.frame_rotate_about_a_line(vect_point, vect_dir, -DA_ICO, frame)
    return neighbour, frame

#convert label coords to position coords
def label_to_point(mesh, compare_label):
    index = None
    for idx, label in enumerate(mesh["labels"]):
        if str(compare_label) == str(label):
            index = idx
            break
    return mesh.points[index]


def list_labels_to_tuples(label_list):
    tuple_list = []
    for label in label_list:
        tuple_list.append(label)
    return tuple_list

def plot_walls(wall_coords, plotter):
    for wall in wall_coords:
        cyl = pv.Cylinder(
            center = label2point(wall),
            direction = (0,0,1),
            radius = 0.2,
            height = 1,
            resolution = 10,
        )
        plotter.add_mesh(cyl, color='red')

def frame_to_pose(frame):
    return np.around(frame[:-1, :-1], 2)

#gets frame position in label form
def get_frame_label(frame):
    x = frame[0][3]
    y = frame[1][3]
    point = np.around([x, y, PLANE_OFFSET], 4)
    label = point2label(point)
    
    if label == None:
        return None
    else:
        return label
    
#convert position coords to label coords
def point_to_label(mesh, compare_point):
    index = None
    for idx, point in enumerate(mesh.points):
        if np.allclose(point, compare_point, rtol=0, atol=0.01):
            index = idx
            break
    return mesh["labels"][index]

def point2label(point):
    point_x = point[0]
    point_y = point[1]
    
    label_x = int(round(point_x/(np.sqrt(3)/3)))
    
    if (int(point_y) + label_x) % 2 == 1:
        label_y = point_y - 1/3
    else:
        label_y = point_y
    return (int(label_x), int(round(label_y)))

def label2point(label):
    (label_x, label_y) =  label
    if (label_x + label_y) % 2 == 1:
        point_y = label_y + 1/3
    else:
        point_y = label_y
    point_x = label_x*(np.sqrt(3)/3)

    arr = np.asarray([round(point_x, 4), round(point_y, 4), -0.8727])
    return arr