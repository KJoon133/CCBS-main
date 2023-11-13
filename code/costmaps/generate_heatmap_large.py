import sys
import cv2
import os
import argparse
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, '/home/railab/Workspace/CCBS/code/')

from pathlib import Path
from core.utils.visualize_heatmap import visualize_heatmap

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

from core.func.cbs_basic import CBSSolver, CBSSolver_Cost

def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        # print(path)
        rst += len(path) - 1
        if(len(path)>1):
            assert path[-1] != path[-2]
    return rst

def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    
    # #_agents
    line = f.readline()
    num_agents = int(line)

    # #_agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))

    return my_map, starts, goals

def min_vertex_cover(graph):
    """Returns the minimum vertex cover of the given graph."""
    # Make a copy of the graph to avoid modifying the original
    graph = graph.copy()

    # Initialize the vertex cover to be empty
    vertex_cover = list()

    # Greedily add nodes to the vertex cover until all edges are covered
    while len(graph.edges()) != 0:
        degrees = dict(graph.degree())
        nodes = sorted(degrees, key=lambda n: degrees[n], reverse=True)
        node = nodes[0]

        if graph.has_node(node):
            vertex_cover.append(node)
            print("Node {} added to vertex cover: {}".format(node, vertex_cover))

            if graph.has_node(node):
                graph.remove_node(node)

    return vertex_cover

def make_graph_from_collision_data(collision_data):
    collision_graph = nx.Graph()

    for collision in collision_data:
        a1 = collision['a1']
        a2 = collision['a2']

        nodes = [a1, a2]

        collision_graph.add_nodes_from(nodes)
        collision_graph.add_edge(a1, a2)

    return collision_graph

def compute_gaussian(alpha, beta, x, y, sigma):
    # print("x-alpha: {}".format(x-alpha))
    # print("y-beta: {}".format(y-beta))
    return np.round(np.exp(-1 * (((x-alpha)**2 + (y-beta)**2) / (2*(sigma**2)))), 3)

def make_heatmap(collision_map_array, sigma):
    heatmap_list = []
    agent_num, map_size_x, map_size_y = collision_map_array.shape[0], collision_map_array.shape[1], collision_map_array.shape[2]

    for collision_map in collision_map_array:
        heatmap = np.zeros_like(collision_map)
        if np.count_nonzero(collision_map) > 0:
            non_zero_list = np.nonzero(collision_map)
            # collision one point
            if len(non_zero_list[0]) == 1:
                alpha, beta = non_zero_list[0], non_zero_list[1]
                for x in range(map_size_x):
                    for y in range(map_size_y):
                        heatmap[x][y] = compute_gaussian(alpha, beta, x, y, sigma)
                # print(heatmap)
                heatmap_list.append(heatmap)
            else:
                for loc in non_zero_list:
                    alpha, beta = loc[0], loc[1]
                    for x in range(map_size_x):
                        for y in range(map_size_y):
                            heatmap[x][y] = compute_gaussian(alpha, beta, x, y, sigma)
                # print(heatmap)
                heatmap_list.append(heatmap)
        else:
            heatmap_list.append(heatmap)

    print(np.shape(heatmap_list))
    return np.array(heatmap_list).reshape(agent_num, 1, map_size_x, map_size_y)

def print_result(instance_name, cbs_results, ccbs_results):
    text = "Instance: {}\nCBS results!\nCost: {}, Gen_nodes: {}, Exp_nodes: {}, Total CT: {}\nCCBS results!\nCost: {}, Gen_nodes: {}, Exp_nodes: {}, Total CT: {}\n".format(instance_name, cbs_results[0], cbs_results[1], cbs_results[2], cbs_results[3], ccbs_results[0],ccbs_results[1],ccbs_results[2], ccbs_results[3])
    return text

def save_result(test_name, instance_name, text, case):
    if case == "cost_error":
        f = open("/home/railab/Workspace/CCBS/Results/Making_heatmap/{}/bigger_heatmap/cost_error/{}_result.txt".format(test_name, instance_name), 'w')
        f.write(text)
    elif case == "nodes_inc":
        f = open("/home/railab/Workspace/CCBS/Results/Making_heatmap/{}/bigger_heatmap/nodes_inc/{}_result.txt".format(test_name, instance_name), 'w')
        f.write(text)
    elif case == "nodes_dec":
        f = open("/home/railab/Workspace/CCBS/Results/Making_heatmap/{}/bigger_heatmap/nodes_dec/{}_result.txt".format(test_name, instance_name), 'w')
        f.write(text)
    else:
        print("CASE ERROR!")
        return

    f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, default=None,
                        help='test_$number$')
    parser.add_argument('--generator', type=int, default=None,
                        help='generator number')
    args = parser.parse_args()

    mapf_path_list = sorted(glob.glob("/home/railab/Workspace/CCBS/code/instances/generated/{}/test*".format(args.test_name)))
    if args.generator == None:
        print("Generator number is None!!")
        raise ValueError(args.generator)
    else:
        if args.generator == 1:
            mapf_path_list = mapf_path_list[:187]
        elif args.generator == 2:
            mapf_path_list = mapf_path_list[187:375]
        elif args.generator == 3:
            mapf_path_list = mapf_path_list[375:562]
        elif args.generator == 4:
            mapf_path_list = mapf_path_list[562:750]
        elif args.generator == 5:
            mapf_path_list = mapf_path_list[750:937]
        elif args.generator == 6:
            mapf_path_list = mapf_path_list[937:1125]
        elif args.generator == 7:
            mapf_path_list = mapf_path_list[1125:1312]
        elif args.generator == 8:
            mapf_path_list = mapf_path_list[1312:1500]
        elif args.generator == 9:
            mapf_path_list = mapf_path_list[1500:1687]
        elif args.generator == 10:
            mapf_path_list = mapf_path_list[1687:1875]
        elif args.generator == 11:
            mapf_path_list = mapf_path_list[1875:2062]
        elif args.generator == 12:
            mapf_path_list = mapf_path_list[2062:2250]
        elif args.generator == 13:
            mapf_path_list = mapf_path_list[2250:2437]
        elif args.generator == 14:
            mapf_path_list = mapf_path_list[2437:2625]
        elif args.generator == 15:
            mapf_path_list = mapf_path_list[2625:2812]
        elif args.generator == 16:
            mapf_path_list = mapf_path_list[2812:]

    costmap_file_list = sorted(os.listdir("/home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/{}_heatmap/bigger_heatmap/".format(args.test_name)))

    # /home/railab/Workspace/CCBS/Results/Making_heatmap/test_70/cost_error
    cost_error_file_list = sorted(os.listdir("/home/railab/Workspace/CCBS/Results/Making_heatmap/{}/bigger_heatmap/cost_error/".format(args.test_name)))
    nodes_dec_file_list = sorted(os.listdir("/home/railab/Workspace/CCBS/Results/Making_heatmap/{}/bigger_heatmap/nodes_dec/".format(args.test_name)))
    nodes_inc_file_list = sorted(os.listdir("/home/railab/Workspace/CCBS/Results/Making_heatmap/{}/bigger_heatmap/nodes_inc/".format(args.test_name)))

    result_file_list = cost_error_file_list + nodes_dec_file_list + nodes_inc_file_list
    # print(result_file_list)
    # return

    random.shuffle(mapf_path_list)

    for mapf_path in mapf_path_list:
        test_name = args.test_name
        print(mapf_path)
        scene_name = mapf_path.split("/")[-1][:-4]
        
        if scene_name + '.npy' in costmap_file_list or scene_name + "_result.txt" in result_file_list:
            print("Collision heatmap or result file alread exist!")
            continue

        mapf_scene_name = "/home/railab/Workspace/CCBS/code/instances/generated/{}/{}.txt".format(args.test_name, scene_name)

        map, starts, goals = import_mapf_instance(mapf_scene_name)
        agent_num = len(starts)
        
        collision_cbs = CBSSolver(map,starts,goals)
        root = collision_cbs.make_root()

        astar_path_list, collision_data = root['paths'], root['collisions']

        # collision_graph = make_graph_from_collision_data(collision_data)
        # vertex_cover = min_vertex_cover(collision_graph)
        map_size = np.shape(map)

        collision_map_array = np.zeros((agent_num, map_size[0], map_size[1]))
        # for collision_agent in vertex_cover:
        for _ in range(len(collision_data)):
            collision = collision_data[0]
            a1, a2, loc, timestep = collision['a1'], collision['a2'], collision['loc'], collision['timestep']

            # if a1 == collision_agent or a2 == collision_agent:
            x, y = loc[0]
            collision_map_array[a1][x][y] = 1
            collision_map_array[a2][x][y] = 1
            collision_data.remove(collision)

        # Make Gaussian heatmap
        ######### Choose the parameter by map size ###################
        heatmap_list = []
        for costmap in collision_map_array:
            heatmap_list.append(cv2.GaussianBlur(costmap, (25,25), 5))
        ##############################################################
        
        # # Normalize
        normalized_heatmap_list = []
        for heatmap in heatmap_list:
            if np.count_nonzero(heatmap) > 0:
                normalized_heatmap_list.append((heatmap-np.min(heatmap))/(np.max(heatmap)-np.min(heatmap)))
            else:
                normalized_heatmap_list.append(heatmap)

        # normalized_heatmap_array = np.array(normalized_heatmap_list)

        # for heatmap_row in normalized_heatmap_array[0]:
        #     for heatmap_col in heatmap_row:
        #         if heatmap_col != 0:
        #             print(heatmap_col)

        # /home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/test_70_heatmap/bigger_heatmap/test_70_14.npy 64
        # /home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/test_69_heatmap/bigger_heatmap/test_69_59.npy 128
        # /home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/test_62_heatmap/bigger_heatmap/test_62_13.npy 256
        # base_heatmap = np.load("/home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/test_62_heatmap/bigger_heatmap/test_62_13.npy")
        # visualize_heatmap(base_heatmap, "base")

        normalized_heatmap_array = np.array(normalized_heatmap_list).reshape(agent_num, 1, map_size[0], map_size[1])
        # visualize_heatmap(normalized_heatmap_array, 'heatmap')

        # continue

        gen_test_path = "/home/railab/Workspace/CCBS/code/instances/gen_test/{}/{}.txt".format(test_name, scene_name)
        print(gen_test_path)

        f = open(gen_test_path, 'r')
        contents = f.readlines()
        f.close()

        cbs_result = contents[2]

        cbs_cost = int(cbs_result.split(", ")[0].split(": ")[-1])
        cbs_gen_nodes = int(cbs_result.split(", ")[1].split(": ")[-1])
        cbs_exp_nodes = int(cbs_result.split(", ")[2].split(": ")[-1])
        cbs_total_CT = float(cbs_result.split(", ")[3].split(": ")[-1])

        # print(cbs_cost)
        # print(cbs_gen_nodes)
        # print(cbs_exp_nodes)
        # print(cbs_total_CT)

        ccbs_solver = CBSSolver_Cost(map, starts, goals, normalized_heatmap_array)
        ccbs_path, ccbs_gen_nodes, ccbs_exp_nodes, ccbs_total_CT = ccbs_solver.find_solution_heatmap(cbs_gen_nodes, cbs_exp_nodes)

        if ccbs_path == None or ccbs_gen_nodes == None or ccbs_exp_nodes == None:
            print("==============================================================================\n")
            case = "nodes_inc"
            error_txt = "CCBS calc long"
            save_result(test_name, scene_name, error_txt, case)
            continue

        ccbs_cost = get_sum_of_cost(ccbs_path)

        ccbs_result = "CCBS Result!\nCost: {}, Gen nodes: {}, Exp nodes: {}, total CT: {}".format(ccbs_cost, ccbs_gen_nodes, ccbs_exp_nodes, ccbs_total_CT)
        # print(ccbs_result)

        final_result = "".join(contents[:-2]) + ccbs_result

        print(final_result)

        if cbs_cost < ccbs_cost:
            case = "cost_error"
        elif cbs_gen_nodes <= ccbs_gen_nodes or cbs_exp_nodes <= ccbs_exp_nodes:
            case = "nodes_inc"
        elif cbs_gen_nodes > ccbs_gen_nodes and cbs_exp_nodes > ccbs_exp_nodes:
            case = "nodes_dec"
        else:
            case = "Unknown"
            continue

        # print(case)

        save_result(test_name, scene_name, final_result, case)
        # /home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/test_68_heatmap/normalized
        if case != "nodes_dec":
            continue
        elif case == "nodes_dec":
            np.save("/home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/{}_heatmap/bigger_heatmap/{}.npy".format(test_name, scene_name), normalized_heatmap_array, allow_pickle=True)

    # for mapf_path in mapf_path_list:
    #     test_name = mapf_path.split("/")[-2]
    #     instance_name = mapf_path.split("/")[-1][:-4]
    #     print("==============================================================================\n")
    #     print("Instance: {}\n".format(instance_name))

    #     if instance_name + '.npy' in costmap_file_list or instance_name + "_result.txt" in result_file_list:
    #         print("Collision heatmap or result file alread exist!")
    #         continue

    #     map, starts, goals = import_mapf_instance(mapf_path)
    #     agent_num = len(starts)
        
    #     collision_cbs = CBSSolver(map,starts,goals)
    #     root = collision_cbs.make_root()

    #     astar_path_list, collision_data = root['paths'], root['collisions']

    #     # collision_graph = make_graph_from_collision_data(collision_data)
    #     # vertex_cover = min_vertex_cover(collision_graph)
    #     map_size = np.shape(map)

    #     collision_map_array = np.zeros((agent_num, map_size[0], map_size[1]))
    #     # for collision_agent in vertex_cover:
    #     for _ in range(len(collision_data)):
    #         collision = collision_data[0]
    #         a1, a2, loc, timestep = collision['a1'], collision['a2'], collision['loc'], collision['timestep']

    #         # if a1 == collision_agent or a2 == collision_agent:
    #         x, y = loc[0]
    #         collision_map_array[a1][x][y] = 1
    #         collision_map_array[a2][x][y] = 1
    #         collision_data.remove(collision)
    
    #     # heatmap_array = make_heatmap(collision_map_array, 2.0)
    #     # visualize_heatmap(collision_map_array.resize(agent_num, 1, map_size[0], map_size[1]), "sample")
    #     # break

    #     # # Make Gaussian heatmap
    #     heatmap_list = []
    #     for costmap in collision_map_array:
    #         heatmap_list.append(cv2.GaussianBlur(costmap, (9,9), 2.0))
        
    #     # # Normalize
    #     normalized_heatmap_list = []
    #     for heatmap in heatmap_list:
    #         if np.count_nonzero(heatmap) > 0:
    #             normalized_heatmap_list.append((heatmap-np.min(heatmap))/(np.max(heatmap)-np.min(heatmap)))
    #         else:
    #             normalized_heatmap_list.append(heatmap)
    #     normalized_heatmap_array = np.array(normalized_heatmap_list).reshape(agent_num, 1, map_size[0], map_size[1])

    #     visualize_heatmap(normalized_heatmap_array, "sample")
        # break
        # print(normalized_heatmap_array)
        # visualize_heatmap(normalized_heatmap_array, "normalized_heatmap")

        # Compare results of CBS and Cost+CBS
        # print()
        # cbs_solver = CBSSolver(map, starts, goals)
        # # print(cbs_solver.heuristics)
        # cbs_path, cbs_gen_nodes, cbs_exp_nodes, cbs_total_CT = cbs_solver.find_solution()

        # if cbs_path == None or cbs_gen_nodes == None or cbs_exp_nodes == None:
        #     print("==============================================================================\n")
        #     case = "nodes_inc"
        #     error_txt = "CBS calc long"
        #     save_result(test_name, instance_name, error_txt, case)
        #     continue

        # print("CBS Done!\n")

        # # print(normalized_heatmap_array)
        # ccbs_solver = CBSSolver_Cost(map, starts, goals, normalized_heatmap_array)
        # ccbs_path, ccbs_gen_nodes, ccbs_exp_nodes, ccbs_total_CT = ccbs_solver.find_solution_heatmap(cbs_gen_nodes, cbs_exp_nodes)

        # if ccbs_path == None or ccbs_gen_nodes == None or ccbs_exp_nodes == None:
        #     print("==============================================================================\n")
        #     case = "nodes_inc"
        #     error_txt = "CCBS calc long"
        #     save_result(test_name, instance_name, error_txt, case)
        #     continue

        # print("CCBS Done!\n")

        # if cbs_path is not None and ccbs_path is not None:
        #     cbs_cost = get_sum_of_cost(cbs_path)
        #     ccbs_cost = get_sum_of_cost(ccbs_path)

        #     text = print_result(instance_name, [cbs_cost, cbs_gen_nodes, cbs_exp_nodes, cbs_total_CT], [ccbs_cost, ccbs_gen_nodes, ccbs_exp_nodes, ccbs_total_CT])
        #     print(text)
        #     print("==============================================================================\n")

        #     case = None
        #     # /home/railab/Workspace/CCBS/Results/Making_heatmap/cost_error
        #     if cbs_cost < ccbs_cost:
        #         case = "cost_error"
        #     elif cbs_gen_nodes <= ccbs_gen_nodes or cbs_exp_nodes <= ccbs_exp_nodes:
        #         case = "nodes_inc"
        #     else:
        #         case = "nodes_dec"
            
        #     save_result(test_name, instance_name, text, case)
        #     # # /home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/test_68_heatmap/normalized
        #     np.save("/home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/{}_heatmap/sorted_normalized_nonvertex/{}.npy".format(test_name, instance_name), normalized_heatmap_array, allow_pickle=True)

if __name__ == "__main__":
    main()