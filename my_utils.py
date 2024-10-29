import numpy as np
import pyvista as pv

#import PH_curves

def world_frame():
  world_frame = np.identity(4)
  return world_frame

def homogeneous_frame(ex, ey, ez, B): 
  frame = np.identity(4)
  frame[0:3, 0] = ex
  frame[0:3, 1] = ey
  frame[0:3, 2] = ez
  frame[0:3, 3] = B   

  return frame

def decomp_frame(frame):
  ex = frame[0:3, 0]
  ey = frame[0:3, 1]
  ez = frame[0:3, 2]
  B = frame[0:3, 3]  

  return ex, ey, ez, B

def vector_after_rotation(rot_vector, t, vector):
  rot_matrix = rot_mat(rot_vector, t)
  return np.matmul(rot_matrix, vector)

def vectors_after_rotation(rot_vector, t, vectors):
  rot_matrix = rot_mat(rot_vector, t)

  for idx in range(len(vectors)):
      vectors[idx] = np.matmul(rot_matrix, vectors[idx].T)
  
  return vectors

def rot_mat(rot_vector, t):
  rot_vector = vector_normalize(rot_vector) # unit vector
  ux = rot_vector[0]
  uy = rot_vector[1]
  uz = rot_vector[2]

  rot_mat = np.zeros([3, 3])
  rot_mat[0, 0] = np.cos(t) + ux ** 2 * (1 - np.cos(t))
  rot_mat[1, 0] = uy * ux * (1 - np.cos(t)) + uz * np.sin(t)
  rot_mat[2, 0] = uz * ux * (1 - np.cos(t)) - uy * np.sin(t)
  rot_mat[0, 1] = ux * uy * (1 - np.cos(t)) - uz * np.sin(t)
  rot_mat[1, 1] = np.cos(t) + uy ** 2 * (1 - np.cos(t))
  rot_mat[2, 1] = uz * uy * (1 - np.cos(t)) + ux * np.sin(t)
  rot_mat[0, 2] = ux * uz * (1 - np.cos(t)) + uy * np.sin(t)
  rot_mat[1, 2] = uy * uz * (1 - np.cos(t)) - ux * np.sin(t)
  rot_mat[2, 2] = np.cos(t) + uz ** 2 * (1 - np.cos(t))

  return rot_mat

def rotate_about_a_line(point, direction, angle):
    # https://www.engr.uvic.ca/~mech410/lectures/4_2_RotateArbi.pdf
  direction = vector_normalize(direction)
  AA = direction[0]
  BB = direction[1]
  CC = direction[2]
  V = np.sqrt(BB ** 2 + CC ** 2)   # L = 1 = norm(direction)

  T_P = np.identity(4)
  T_P[0:3, 3] = -np.asarray(point, dtype=np.float32)

  T_R_X = np.identity(4)

  if not V <= 1e-7:
      T_R_X[1:3, 1:3] = np.array([[CC / V, -BB / V], [BB / V, CC / V]]) 

  T_R_Y = np.identity(4)
  T_R_Y[0:3, 0:3] = np.array([[V, 0, -AA], [0, 1, 0], [AA, 0, V]])

  T_R_Z = np.identity(4)
  T_R_Z[0:2, 0:2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
  
  T_P_inv = np.identity(4)
  T_P_inv[0:3, 3] = point
  T_R_X_inv = np.transpose(T_R_X)
  T_R_Y_inv = np.transpose(T_R_Y)

  T=np.matmul(T_P_inv, np.matmul(T_R_X_inv, np.matmul(T_R_Y_inv, np.matmul(
      T_R_Z, np.matmul(T_R_Y, np.matmul(T_R_X, T_P))))))  

  return T

def frame_rotate_about_a_line(point, direction, angle, frame):
  T = rotate_about_a_line(point, direction, angle)
  ex, ey, ez, B = decomp_frame(frame)
  B = np.matmul(T[0:3, 0:3], B) + T[0:3, 3]
  ex = np.matmul(T[0:3, 0:3], ex)
  ey = np.matmul(T[0:3, 0:3], ey)
  ez = np.matmul(T[0:3, 0:3], ez)

  return homogeneous_frame(ex, ey, ez, B)

def points_rotate_about_a_line(point, direction, angle, points):
  T = rotate_about_a_line(point, direction, angle)
  return points_under_SE3(T, points)

def update_body_axes(rot_vector, t, ex, ey, ez):
  ex = vector_after_rotation(rot_vector, t, ex)
  ey = vector_after_rotation(rot_vector, t, ey)
  ez = vector_after_rotation(rot_vector, t, ez)

  return ex, ey, ez

def get_du_dv_from_ds(x_u, x_v, e1, ds):
	# x_u, x_v, and e1 are w.r.t. the body frame
	# x_u: u-coordinate curve tangent
	# x_v: v-coordinate curve tangent
	# e1: unit vector of the rolling direction
	# ds: rolling distance

  E = np.dot(x_u, x_u)
  F = np.dot(x_u, x_v)
  G = np.dot(x_v, x_v)

  e_u = x_u / np.linalg.norm(x_u)
  e_v = x_v / np.linalg.norm(x_v)

  cos_theta = np.dot(e1, e_u)  
  sin_theta = np.dot(e1, e_v)
  theta = np.arctan2(sin_theta, cos_theta)

  # ds^2 = E*da^2 + G*db^2, but db = tan(theta) * da
  du = np.sqrt(ds * ds/(E + G * np.square(np.tan(theta))))

  if np.abs(theta)>np.pi/2:
      du = -du

  dv = np.tan(theta) * du

  return du, dv, theta

def rotation_angle(v1, v2, v3):
  # find the rotation angle from v1 to v2 about v3
  # v3 is parallel to np.cross(v1, v2)
  v1 = v1 / np.linalg.norm(v1)
  v2 = v2 / np.linalg.norm(v2)
  v3 = v3 / np.linalg.norm(v3)
  cos_theta = np.dot(v1, v2)
  sin_theta = np.dot(np.cross(v1, v2), v3)

  return np.arctan2(sin_theta, cos_theta)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def vector_projection_on_plane(v, n):
  # v: vector; n: unit normal vector of the plane
  return v - np.dot(v, n) * n

def points_under_SE3(T, points):
  # param T: SE(3)
  # param points: Nx3 ndarray
  rot_mat = T[0:3, 0:3]
  origin = T[0:3, 3]
  return origin + (np.matmul(rot_mat, points.T)).T
  
def get_transform_matrix(frame1, frame2):
  # return T: T * frame1 = frame2
  # frame1: world; frame2: body; T * v_world = v_body
  T = np.matmul(frame2, np.linalg.inv(frame1))
  return T

def vector_normalize(v):
  return v / np.linalg.norm(v)

def normal_3_points_plane(point1, point2, B):
  # follow the convention in disk rolling: ey is the normal to the circle plane
  v1 = point1 - B
  v2 = point2 - B
  v_normal = np.cross(v1, v2)
  v_normal = v_normal / np.linalg.norm(v_normal)

  return v_normal

def solve_linear_trig(C0, C1, C2):
  # C0 * cos(beta) + C1 * sin(beta) = C2 
  # sin(alpha) * cos(beta) + cos(alpha) * sin(beta) = C2 / hypotenuse
  # sin(alpha + beta) = C2 / hypotenuse
  hypotenuse = np.sqrt(np.square(C0) + np.square(C1))
  sin_alpha = C0 / hypotenuse
  cos_alpha = C1 / hypotenuse

  if np.abs(C2) <= hypotenuse:
    beta1 = np.arcsin(C2 / hypotenuse) - np.arctan2(sin_alpha, cos_alpha)
    beta2 = np.pi - np.arcsin(C2 / hypotenuse) - np.arctan2(sin_alpha, cos_alpha)
    return (beta1, beta2)
  else:
    return (np.nan, np.nan)

def arc_length_points(points):
  arc_length = 0
  for i in range(len(points) - 1):
      arc_length += np.linalg.norm(points[i] - points[i + 1])
  
  return arc_length

def update_points(frame1, frame2, points_wrt_frame1):
  T = get_transform_matrix(frame1, frame2)
  return points_under_SE3(T, points_wrt_frame1)

def update_vector(frame1, frame2, vector_wrt_frame1):
  T = get_transform_matrix(frame1, frame2)
  rot_mat = T[0:3, 0:3]
  return (np.matmul(rot_mat, vector_wrt_frame1.T)).T

def find_lowest_point(points):
  # lowest point in terms of z coordinates
  # points: nx3 ndarray
  index = np.argmin(points[:, 2])
  return points[index, 0:3]

def find_e1_goal(points):
  index = np.argmin(points[:, 2])
  point = points[index, 0:3]
  direction = points[index+1, 0:3] - point
  direction[2] = 0
  direction = vector_normalize(direction)
  return point, direction


def darboux_frame_as_global_for_planning(
              world_frame, darboux_frame_start, M_start, M_goal, e1_goal
):
  M_start_new = update_points(world_frame, darboux_frame_start, M_start)
  M_goal_new = update_points(world_frame, darboux_frame_start, M_goal)
  e1_goal_new = update_vector(world_frame, darboux_frame_start, e1_goal)
  return M_start_new, M_goal_new, e1_goal_new


def plot_line(start, end, plotter, line_width = 1, color = 'blue'):
  mesh = pv.Line(start, end)

  actor = plotter.add_mesh(
      mesh, 
      color = color,
      line_width = line_width,
      show_edges = True,
      opacity = 1.0)

  return mesh, actor

def plot_arrow(start, direction, plotter, color = 'red'):
  mesh = pv.Arrow(start, direction, 
    tip_length=0.25, tip_radius=0.1, tip_resolution=10, shaft_radius=0.05, 
    shaft_resolution=10, scale=0.7)
  
  plotter.add_mesh(
      mesh, 
      color = color,
      show_edges = False,
      opacity = 1.0)

  return mesh

def plot_plane(xlimit, ylimit, normal, origin, plotter, num_grids = 2):
    u = np.linspace(xlimit[0], xlimit[1], num_grids)
    v = np.linspace(ylimit[0], ylimit[1], num_grids)
    u, v = np.meshgrid(u, v)

    # z = ax + by + c from n dot (x - origin) = 0
    a = -normal[0] / normal[2]
    b = -normal[0] / normal[2]
    c = np.dot(normal, origin) / normal[2]
    z = a * u + b * v + c 

    grid_plane =  pv.StructuredGrid(u, v, z)

    plotter.add_mesh(
        grid_plane, 
        color = 'white',
        show_edges = True,
        opacity = 1.0)

def plot_3d_object(points, plotter):
  # Create and plot structured grid
  grid_object = pv.StructuredGrid(points[:, 0], points[:, 1], points[:, 2])

  plotter.add_mesh(
      grid_object, 
      show_edges=True, 
      opacity=0.6)

  return grid_object

def plot_bean(bu, body_frame, plotter):
    points = bu.bean_points(body_frame)
    grid_bean = pv.StructuredGrid(points[:, 0], points[:, 1], points[:, 2])

    plotter.add_mesh(
        grid_bean, 
        show_edges=True, 
        color = 'white',
        opacity=0.4)

    return grid_bean

def plot_disk(origin, ex, ey, ez, radius, plotter, opacity = 1.0, num_points = 50):
    # ey: normal vector to the disk plane
    
    t = np.linspace(0, 2*np.pi, num_points)
    points = np.empty(shape=[0, 3])

    for e in t:
        point = origin + radius * (np.cos(e) * ex + 0 * ey + np.sin(e) * ez)
        points = np.vstack((points, point))
    
    return plot_curve(points, plotter, opacity)

def plot_curve(points, plotter, opacity = 1.0):
    spline = pv.Spline(points)
    spline_mesh = spline.tube(radius = 0.015)

    plotter.add_mesh(
        spline_mesh,
        color = 'black',
        show_edges = True,
        opacity = opacity)

    return spline_mesh

def plot_point(point, plotter):
    plotter.add_mesh(
            point, 
            color = 'red',
            #render_points_as_spheres = True,
            point_size = 10.0)

def plot_ico(ico, plotter):
    plotter.add_mesh(
            ico, 
            color = 'red',
            show_edges=True,
            line_width=5,
            opacity=0.5,
            #render_points_as_spheres = True,
            point_size = 10.0)
   
          
def plot_frame(frame, plotter):
    # frame: a HomoFrame
    # plotter: pyvista.Plotter() 
    origin = frame[0:3, 3]
    e1 = frame[0:3, 0]
    e2 = frame[0:3, 1]
    e3 = frame[0:3, 2]
    
    e1_mesh, e1_actor = plot_line(origin, origin + e1, plotter)
    e2_mesh, e2_actor = plot_line(origin, origin + e2, plotter)
    e3_mesh, e3_actor = plot_line(origin, origin + e3, plotter)

    plotter.add_mesh(
        e1_mesh,
        color = 'red',
        name="mesh1",
        line_width = 5)

    plotter.add_mesh(
        e2_mesh,
        color = 'green',
        name="mesh2",
        line_width = 5)

    plotter.add_mesh(
        e3_mesh,
        color = 'blue',
        name="mesh3",
        line_width = 5)

    return e1_mesh, e2_mesh, e3_mesh, e1_actor, e2_actor, e3_actor


# def plot_3D_curve(points, ax, line_idxes = None):
#     ax.plot(points[:, 0], points[:, 1], points[:, 2], label='parametric 3D curve')
#     colors = ['red', 'orange']

#     if line_idxes is not None:
#         for idx in range(len(line_idxes) - 1):
#             ax.plot([points[line_idxes[idx], 0], points[line_idxes[idx + 1], 0]], 
#                 [points[line_idxes[idx], 1], points[line_idxes[idx + 1], 1]], 
#                 [points[line_idxes[idx], 2], points[line_idxes[idx + 1], 2]],
#                 color = colors[(idx) % 2])

# def plot_2D_curve(points):
#     plt.figure()
#     plt.plot(points[:, 0], points[:, 1], label = 'parametric 2d curve')


