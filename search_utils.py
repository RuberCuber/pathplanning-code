from collections import deque, defaultdict
from typing import Tuple
import numpy as np
import pyvista as pv
from queue import PriorityQueue
import time
import grid_utils
import my_utils

Location = Tuple[int, int]
DA_ICO = np.arccos((5**0.5)/3) #dihedral angle of icosahedron

#add a new neighbour to the neighbour list
def add_neighbour(neighbour_list, cur: Location, new: Location):
    neighbour_list[cur].append(new)
    return neighbour_list

#breadth first search
def bfs(start, end, neighbour_list, wall_list):
    queue = deque()
    visited_list = defaultdict(list)
    
    #add start to visited list
    visited_list[start] = None
    queue.append(start)

    #loop until queue is empty
    while queue:
        #dequeue and get node
        cur: Location = queue.popleft()
        #break if reached end
        if cur == end:
            break
        
        #loop through neighbours and check if visited already and not in walls
        for neighbour in neighbour_list[cur]:
            if neighbour not in visited_list and neighbour not in wall_list:
                queue.append(neighbour)
                visited_list[neighbour] = cur
    return visited_list

def get_neighbour_list_frame(neighbour_list, frame):
    x = frame[0][3]
    y = frame[1][3]
    pos = grid_utils.get_frame_label(frame)
    frame_list = []
    
    for neighbour in neighbour_list[pos]:
        new_neighbour, new_frame = grid_utils.perform_virtual_roll(grid_utils.label2point(pos), frame, neighbour)
        frame_list.append(new_frame)
    return frame_list

#breadth first search using frames
def bfs2(mesh, start_frame, end_frame, neighbour_list, plotter):
    queue = deque()
    visited_list = defaultdict(list)
    
    #add start to visited list
    visited_list[encode_frame(start_frame)] = None
    queue.append(start_frame)

    #loop until queue is empty
    while queue:
        cur_frame = queue.popleft()
        
        #break if reached end
        if np.allclose(cur_frame, end_frame, rtol=0, atol=0.01):
            print('reached end!')
            break

        #loop through neighbours and check if visited already
        neighbour_list_frame = get_neighbour_list_frame(mesh, neighbour_list, cur_frame)
        for neighbour in neighbour_list_frame:
            if not np.isin(visited_list, neighbour):
                queue.append(neighbour)
                visited_list[encode_frame(neighbour)] = cur_frame
    
    return visited_list

#astar using frames
def astar(start_frame, target_frame, neighbour_list, mode = "Default"):
    queue = PriorityQueue()
    queue.put((0, encode_frame(start_frame)))

    visited_list = defaultdict(list)
    visited_list[encode_frame(start_frame)] = None

    cost_list = defaultdict(list)
    cost_list[encode_frame(start_frame)] = 0    

    end_frame = target_frame
    end_pos = grid_utils.get_frame_label(end_frame)
    end_rots = GetFrameRotations(end_frame)

    iter_count = 0

    h_cost_best = 1000000
    best_frame = start_frame
    best_iter = 0

    #loop until queue is empty
    while queue:
        iter_count += 1
        (cur_cost, cur_frame_encoded) = queue.get()
        cur_frame = decode_frame(cur_frame_encoded)

        #debug print stuff
        if iter_count % 20000 == 0:
            print("iter count: {}".format(iter_count))
            print("target frame: \n{}".format(target_frame))
            print("best frame: \n{}".format(np.around(best_frame, 2)))
            print("best h_cost: \n{}".format(h_cost_best))
            print("best iter: \n{}".format(best_iter))

        #break if reached end
        if np.allclose(cur_frame, target_frame, rtol=0, atol=0.1):
            print('reached end!')
            end_frame = cur_frame
            break

        #temp code, will break if ico lands on the target triangle but not necesarrily the right pose
        #elif grid_utils.get_frame_label(cur_frame) == end_pos and check_rot_tri(cur_frame, end_rots) == True:
        #    print('reached end, location only')
        #    end_frame = cur_frame
        #    break

        #temp code, check for knight movement conditions
        if mode == "Knight" and check_knight_movement(cur_frame, end_frame, end_pos) == True:
            print('reached knight position location')
            end_frame = cur_frame
            break

        #loop through neighbours and check if visited already
        neighbour_list_frame = get_neighbour_list_frame(neighbour_list, cur_frame)
        for neighbour in neighbour_list_frame:
            neighbour_encoded = encode_frame(neighbour)
            new_cost = cost_list[cur_frame_encoded] + 0.001

            #if neighbour_encoded in cost_list:
            #    print('yeah!')
            if not np.isin(visited_list, neighbour_encoded) or new_cost < cost_list[neighbour_encoded]:
                h_cost = heuristic(start_frame, neighbour, target_frame)
                cost_list[neighbour_encoded] = new_cost + h_cost
                #print(new_cost + heuristic(neighbour, target_frame))
                queue.put((new_cost, neighbour_encoded))
                visited_list[neighbour_encoded] = cur_frame

                if h_cost < h_cost_best:
                    best_frame = cur_frame
                    h_cost_best = h_cost
                    best_iter = iter_count

        
    return visited_list, end_frame

#astar search, position only
def astar_position(start_pos, target_pos, neighbour_list):
    queue = PriorityQueue()
    queue.put((0, start_pos))

    visited_list = defaultdict(list)
    visited_list[start_pos] = None

    end_pos = target_pos

    cost_list = defaultdict(list)
    cost_list[start_pos] = 0

    iter_count = 0

    #loop until queue is empty
    while queue:
        iter_count += 1
        (cur_cost, cur_pos) = queue.get()

        #break if reached end
        #if cur_pos == target_pos:
        if heuristic_pos(cur_pos, target_pos) == 0:
            print('reached close enough position!')
            end_pos = cur_pos
            break

        #loop through neighbours and check if visited already
        for neighbour_pos in neighbour_list[cur_pos]:
            new_cost = cost_list[cur_pos] + 0.001

            if neighbour_pos not in visited_list or new_cost < cost_list[neighbour_pos]:
                cost_list[neighbour_pos] = new_cost
                prio_cost = new_cost + heuristic_pos(cur_pos, target_pos)
                queue.put((prio_cost, neighbour_pos))
                visited_list[neighbour_pos] = cur_pos
        
    return visited_list, end_pos

def heuristic(start_frame, cur_frame, end_frame):
    start_loc = grid_utils.get_frame_label(start_frame)
    cur_loc = grid_utils.get_frame_label(cur_frame)
    end_loc = grid_utils.get_frame_label(end_frame)
    ax, ay, az = frame_angles(cur_frame, end_frame)

    (x1, y1) = cur_loc
    (x2, y2) = end_loc
    (x3, y3) = start_loc
    total_dist = abs(x3 - x2) + abs(y3 - y2)

    cur_dist = abs(x1 - x2) + abs(y1 - y2)
    rot = ax + ay + az

    #proportion of current distance to end from total distance, range: 0 - 1, with closer to 1 being same distance at starting, and 0 being right on top of end pos.
    #If proportion >1, means cur is further away than starting
    if total_dist == 0:
        dist_proportion = 1
    else:
        dist_proportion = (cur_dist/total_dist)
    if dist_proportion > 1: dist_proportion = 1

    dist_scaling = 4
    dist_proportion = dist_proportion**dist_scaling
    
    rot_weight = 1 - dist_proportion
    dist_weight = dist_proportion

    dist_cost = cur_dist*dist_weight
    rot_cost = rot*rot_weight
    #print(rot_cost + dist_cost)
    return rot_cost + dist_cost
    #return ax + ay + az
    #return 0

def heuristic_pos(cur_pos, target_pos):
    (x1, y1) = cur_pos
    (x2, y2) = target_pos

    cur_dist = abs(x1 - x2) + abs(y1 - y2)
    return cur_dist

def frame_angles(f1, f2):
    ex1, ey1, ez1, B1 = my_utils.decomp_frame(f1)
    ex2, ey2, ez2, B2 = my_utils.decomp_frame(f2)
    angle_x = my_utils.angle_between(ex1, ex2)
    angle_y = my_utils.angle_between(ey1, ey2)
    angle_z = my_utils.angle_between(ez1, ez2)
    return angle_x, angle_y, angle_z

def GetPath(start, end, visited_list):
    path = []
    cur = end
    
    #print("debug1: {}".format(grid_utils.get_frame_label(visited_list[str(end)])))

    while cur is not None and visited_list[encode_frame(cur)] is not None:
        path.append(grid_utils.get_frame_label(cur))
        cur = visited_list[encode_frame(cur)]
        #print("debug2: {}".format(cur))
    path.append(grid_utils.get_frame_label(start))
    
    return list(reversed(path))

def GetPathPos(start, end, visited_list):
    path = []
    cur = end

    while cur is not None and visited_list[cur] is not None:
        path.append(cur)
        cur = visited_list[cur]
    path.append(start)
    return list(reversed(path))

#get 2 other frame rotations of current location
def GetFrameRotations(frame):
    x = frame[0][3]
    y = frame[1][3]
    frame_1 = my_utils.frame_rotate_about_a_line([x, y, 0], [0, 0, 1], np.deg2rad(120), frame)
    frame_2 = my_utils.frame_rotate_about_a_line([x, y, 0], [0, 0, 1], np.deg2rad(240), frame)
    return [frame, frame_1, frame_2]

#check if the pose is a rotation of current pos (ie, on the same grid triangle)
def check_rot_tri(cur_frame, target_rots):
    for frame in target_rots:
        if np.allclose(cur_frame, frame, rtol=0, atol=0.01):
            return True
    return False

#check if frame is in position to perform 'knight movement'
def check_knight_movement(cur_frame, target_frame, target_pos):
    (cur_x, cur_y) = grid_utils.get_frame_label(cur_frame)
    (target_x, target_y) = target_pos
    cur_rot = grid_utils.frame_to_pose(cur_frame)
    target_rot = grid_utils.frame_to_pose(target_frame)

    if abs(cur_x-target_x) == 1 and abs(cur_y-target_y) == 1:
        if np.allclose(cur_rot, target_rot, rtol=0, atol=0.01):
            return True
    return False
        

def encode_frame(frame):
    return frame.tobytes()

def decode_frame(bytes):
    return np.frombuffer(bytes).reshape(4, 4)
    