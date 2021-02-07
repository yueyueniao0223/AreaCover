import math 
from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from gridMap import GridMap
import matplotlib.pyplot as plt

do_animation = True

class Cover:
    class Direction(IntEnum):
        Up = 1
        DOWN = -1
    
    class MovingDirection(IntEnum):
        RIGHT = 1
        LEFT = -1
    
    def __init__(self, moving_direction, direction, x_inds_gol_y,goal_y):
        self.moving_direction = moving_direction
        self.direction = direction
        self.turning_window = []
        self.update_turning_window()
        self.x_indexes_goal_y = x_inds_gol_y
        self.goal_y = goal_y
    
    def find_safe_turning_grid(self, c_x_index, c_y_index, grid_map):
        for (d_x_ind, d_y_ind) in self.turning_window:
            next_x_ind = d_x_ind + c_x_index
            next_y_ind = d_y_ind + c_y_index

            if not grid_map.check_occupied_from_xy_index(next_x_ind,next_y_ind,occupied_val = 0.5):
                return next_x_ind, next_y_ind
        
        return None, None

    def update_turning_window(self):
        self.turning_window = [
            (self.moving_direction,0.0),
            (self.moving_direction, self.direction),
            # (0, self.direction),
            (-self.moving_direction, self.direction),
        ]

    def swap_moving_direction(self):
        self.moving_direction *= -1
        self.update_turning_window()

    
    def move_target_grid(self, c_x_index, c_y_index, grid_map):
        n_x_index = self.moving_direction + c_x_index
        n_y_index = c_y_index

        if not grid_map.check_occupied_from_xy_index(n_x_index,n_y_index,occupied_val=0.5):
            return n_x_index,n_y_index
        else:
            next_c_x_index, next_c_y_index = self.find_safe_turning_grid(c_x_index,c_y_index,grid_map)
            #退一格
            if(next_c_x_index is None) and (next_c_y_index is None):
                next_c_x_index = -self.moving_direction + c_x_index
                next_c_y_index = c_y_index
                if grid_map.check_occupied_from_xy_index(next_c_x_index,next_c_y_index):
                    return None,None
            else:
                while not grid_map.check_occupied_from_xy_index(
                        next_c_x_index + self.moving_direction, 
                        next_c_y_index,occupied_val=0.5):
                    next_c_x_index += self.moving_direction
                self.swap_moving_direction()

            return next_c_x_index,next_c_y_index
    
    def is_search_done(self, grid_map):
        for ix in self.x_indexes_goal_y:
            if not grid_map.check_occupied_from_xy_index(ix, self.goal_y,occupied_val = 0.5):
                return False
        return True

    def search_start_grid(self, grid_map):
        x_inds = []
        y_ind = 0
        if self.direction == self.Direction.DOWN:
            x_inds, y_ind = search_free_grid_index_at_edge_y(grid_map, from_upper=True)
        elif self.direction == self.Direction.Up:
            x_inds, y_ind = search_free_grid_index_at_edge_y(grid_map, from_upper=False)

        if self.moving_direction == self.MovingDirection.RIGHT:
            return min(x_inds), y_ind
        elif self.moving_direction == self.MovingDirection.LEFT:
            return max(x_inds), y_ind
        
        return ValueError("self.moving direction is invalid ")

    def is_search_done(self, grid_map):
        for ix in self.x_indexes_goal_y:
            if not grid_map.check_occupied_from_xy_index(ix, self.goal_y, occupied_val = 0.5):
                return False
        return True

def find_direction_and_start_pos(ox, oy):
    max_dist = 0.0
    vec = [0.0, 0.0]
    start_pos = [0.0, 0.0]
    #找到最长的那条边
    for i in range(len(ox) - 1):
        dx = ox[i + 1] - ox[i]
        dy = oy[i + 1] - oy[i]
        d = np.hypot(dx, dy)

        if d > max_dist:
            max_dist = d
            vec = [dx, dy]
            start_pos = [ox[i], oy[i]] 
    return vec,start_pos

def convert_grid_coordinate(ox, oy, vec, start_position):
    tx = [ix - start_position[0] for ix in ox]
    ty = [iy - start_position[1] for iy in oy]
    th = math.atan2(vec[1], vec[0])
    rot = Rot.from_euler('z', th).as_matrix()[0:2, 0:2]
    converted_xy = np.stack([tx, ty]).T @ rot

    return converted_xy[:, 0], converted_xy[:, 1]

def search_free_grid_index_at_edge_y(grid_map, from_upper=False):
    y_index = None
    x_indexes = []

    if from_upper:
        x_range = range(grid_map.height)[::-1]
        y_range = range(grid_map.width)[::-1]
    else:
        x_range = range(grid_map.height)
        y_range = range(grid_map.width)
    
    for iy in x_range:
        for ix in y_range:
            if not grid_map.check_occupied_from_xy_index(ix, iy):
                y_index = iy
                x_indexes.append(ix)
        if y_index:
            break
    return x_indexes, y_index

def setup_grid_map(ox, oy, resolution, direction, offset_grid=10):
    width = math.ceil(max(ox) - min(ox) / resolution) + offset_grid
    height = math.ceil(max(oy) - min(oy) / resolution) + offset_grid
    center_x = (np.max(ox) + np.min(ox)) / 2.0
    center_y = (np.max(oy) + np.min(oy)) / 2.0

    grid_map = GridMap(width,height,resolution,center_x,center_y)
    grid_map.print_grid_map_info()
    grid_map.set_value_from_polygon(ox, oy, 1.0, inside=False)
    grid_map.expand_grid()

    x_inds_goal_y = []
    goal_y = 0
    if direction == Cover.Direction.Up:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(grid_map, from_upper=True)
    elif direction == Cover.Direction.DOWN:
        x_inds_goal_y, goal_y = search_free_grid_index_at_edge_y(grid_map, from_upper=False)
    
    return grid_map, x_inds_goal_y, goal_y



def path_search(searcher, grid_map, grid_search_animation=False):
    c_x_index, c_y_index = searcher.search_start_grid(grid_map)
    if not grid_map.set_value_from_xy_index(c_x_index, c_y_index, 0.5):
        print("Cannot find start grid")
        return [],[]
    
    x,y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index, c_y_index)

    px,py = [x], [y]
    fig, ax = None, None
    if grid_search_animation:
        fig, ax =plt.subplots()
        fig.canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
    
    while True:
        c_x_index, c_y_index = searcher.move_target_grid(c_x_index,c_y_index,grid_map)
        if searcher.is_search_done(grid_map) or (c_x_index is None or c_y_index is None):
            print("Done")
            break
        
        x,y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index, c_y_index)

        px.append(x)
        py.append(y)

        grid_map.set_value_from_xy_index(c_x_index,c_y_index, 0.5)

        if grid_search_animation:
            grid_map.plot_grid_map(ax = ax)
            plt.pause(0.5)
    
    return px, py

def convert_global_coordinate(x, y, vec, start_position):
    th = math.atan2(vec[1], vec[0])
    rot = Rot.from_euler('z', -th).as_matrix()[0:2, 0:2]
    converted_xy = np.stack([x, y]).T @ rot
    rx = [ix + start_position[0] for ix in converted_xy[:, 0]]
    ry = [iy + start_position[1] for iy in converted_xy[:, 1]]
    return rx, ry

def planning(ox, oy, resolution, moving_direction = Cover.MovingDirection.RIGHT,
             direction = Cover.Direction.Up):
    vec, start_position = find_direction_and_start_pos(ox, oy)
    rox, roy = convert_grid_coordinate(ox, oy, vec, start_position)
    grid_map, x_inds_goal_y, goal_y = setup_grid_map(rox, roy, resolution, direction)

    searcher = Cover(moving_direction, direction, x_inds_goal_y, goal_y)
    px, py = path_search(searcher, grid_map)
    rx, ry = convert_global_coordinate(px, py, vec, start_position)

    print("Path length:", len(rx))

    return rx, ry

def planning_animation(ox, oy, resolution):
    px, py = planning(ox, oy, resolution, direction=Cover.Direction.Up)
    
    for ipx,ipy in zip(px, py):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event:[exit(0) if event.key == 'escape' else None])
        plt.plot(ox,oy, '-xb')
        plt.plot(px,py,'-r')
        plt.plot(ipx,ipy,'or')
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)

    plt.cla()
    plt.plot(ox, oy, "-xb")
    plt.plot(px, py, "-r")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.1)
    # plt.close()

def main():
    print("start!!!")
    #长方形区域
    # ox = [0.0,200.0,200.0,0.0,0.0]
    # oy = [0.0, 0.0, 60.0, 60.0, 0.0]

    # ox = [0.0, 20.0, 50.0, 100.0, 130.0, 40.0, 0.0]
    # oy = [0.0, -20.0, 0.0, 30.0, 60.0, 80.0, 0.0]

    #正凸形    
    ox = [0.0, 200.0, 200.0, 150.0, 150.0, 50.0, 50.0, 0.0, 0.0]
    oy = [0.0, 0.0, 60.0, 60.0, 120.0, 120.0, 60.0, 60.0, 0.0]

    #倒凸形
    # ox = [0.0, 150.0, 150.0, 100.0, 100.0, 50.0, 50.0, 0.0, 0.0] 
    # oy = [120.0, 120.0, 60.0, 60.0, 0.0, 0.0, 60.0, 60.0, 120.0]

    #此情况下无法全覆盖
    # ox = [0.0, 150.0, 150.0, 100.0, 100.0, 50.0, 50.0, 0.0, 0.0]
    # oy = [0.0, 0.0, 120.0, 120.0, 60.0, 60.0, 120.0, 120.0, 0.0]

    resolution = 5.0
    planning_animation(ox, oy, resolution)
    plt.show()
    print("done!!!")

if __name__ == "__main__":
    main()