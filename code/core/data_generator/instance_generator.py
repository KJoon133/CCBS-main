#!/usr/bin/python
import argparse, os, sys, glob, time, random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from pathlib import Path

sys.path.append("/home/railab/Workspace/CCBS/code/")

from core.func.cbs_basic import CBSSolver
from core.func.icbs_complete import ICBS_Solver_Compare

from core.utils.utils import get_sum_of_cost, import_mapf_instance

HLSOLVER = "CBS"
LLSOLVER = "a_star"

def timeout(signum, frame):
    raise Exception('end of time')

def print_mapf_instance(my_map, starts, goals):
    print('Start locations')
    print_locations(my_map, starts)
    print('Goal locations')
    print_locations(my_map, goals)

def print_locations(my_map, locations):
    starts_map = [[-1 for _ in range(len(my_map[0]))] for _ in range(len(my_map))]
    for i in range(len(locations)):
        starts_map[locations[i][0]][locations[i][1]] = i
    to_print = ''
    for x in range(len(my_map)):
        for y in range(len(my_map[0])):
            if starts_map[x][y] >= 0:
                to_print += str(starts_map[x][y]) + ' '
            elif my_map[x][y]:
                to_print += '@ '
            else:
                to_print += '. '
        to_print += '\n'
    print(to_print)

# Make test_*.txt files
def generate_instance(map, starts, goals, save_path, test_name):
    
    grid_num = np.where(np.asarray(map)==False)
    vacant_grid = [(int(x), int(y)) for x, y in zip(grid_num[0], grid_num[1])]

    agent_num = len(starts)

    text_map = ""
    for row in range(len(map)):
        for column in range(len(map[row])):
            if map[row][column] == False:
                text_map += '. '
            else:
                text_map += '@ '
        text_map += "\n"

    flag = 85001

    while True:
        sampling_grid = vacant_grid
        random_goal = random.sample(sampling_grid, agent_num*2)

        text_data = "{} {}\n".format(len(map), len(map[0]))
        text_data += text_map

        text_data +=  str(agent_num) + "\n"

        agent_list = [[a, b] for a, b in zip(random_goal[agent_num:], random_goal[:agent_num])]

        agent_text = ""

        for i in range(len(agent_list)):
            for j in range(len(agent_list[i])):
                agent_text += " ".join(str(a) for a in agent_list[i][j])
                if j == 0:
                    agent_text += " "
            agent_text += "\n"

        text_data += agent_text

        f = open(save_path + test_name + "_" + str(flag) + ".txt", 'w')
        f.write(text_data)
        f.close()

        if flag == 90000:
            break
    
        flag += 1

    return


def main():
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('--generation', action='store_true', default=False,
                        help='For data generation')
    parser.add_argument('--gen_test', action='store_true', default=False,
                        help='Test with generated data')
    parser.add_argument('--gen_test_num', type=int, default=None)
    parser.add_argument('--test', type=str, default=None)
    args = parser.parse_args()

    standard_instance = "/home/railab/Workspace/CCBS/code/instances/{}.txt".format(args.test)

    test_name = args.test
    save_path = "/home/railab/Workspace/CCBS/code/instances/generated/{}/".format(test_name)

    if args.generation:
        map, starts, goals = import_mapf_instance(standard_instance)

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError:
            print("Error: Failed to create the directory.")

        # generated_instance_list = glob.glob(save_path + "*")

        generate_instance(map, starts, goals, save_path, test_name)

    else:
        # Batch test
        scene_list = sorted(glob.glob(save_path + "*.txt"))

        if args.gen_test_num == 1:
            scene_list = scene_list[:312]
        elif args.gen_test_num == 2:
            scene_list = scene_list[312:625]
        elif args.gen_test_num == 3:
            scene_list = scene_list[625:937]
        elif args.gen_test_num == 4:
            scene_list = scene_list[937:1250]
        elif args.gen_test_num == 5:
            scene_list = scene_list[1250:1562]
        elif args.gen_test_num == 6:
            scene_list = scene_list[1562:1875]
        elif args.gen_test_num == 7:
            scene_list = scene_list[1875:2187]
        elif args.gen_test_num == 8:
            scene_list = scene_list[2187:2500]
        elif args.gen_test_num == 9:
            scene_list = scene_list[2500:2812]
        elif args.gen_test_num == 10:
            scene_list = scene_list[2812:3125]
        elif args.gen_test_num == 11:
            scene_list = scene_list[3125:3437]
        elif args.gen_test_num == 12:
            scene_list = scene_list[3437:3750]
        elif args.gen_test_num == 13:
            scene_list = scene_list[3750:4062]
        elif args.gen_test_num == 14:
            scene_list = scene_list[4062:4375]
        elif args.gen_test_num == 15:
            scene_list = scene_list[4375:4687]
        elif args.gen_test_num == 16:
            scene_list = scene_list[4687:]
        
        # 55 시작, 640, 930, 1220
        elif args.gen_test_num == 0:
            scene_list = scene_list
        
        # 282
        elif args.gen_test_num == None:
            print("gen test all!")
            pass

        print(len(scene_list))
        
        count = 0

        for scene in scene_list:
            scene_name = scene.split("/")[-1][:-4]
            test_name = scene.split("/")[-2]

            test_save_path = "/home/railab/Workspace/CCBS/code/instances/gen_test/{}/".format(test_name)

            print("Scene: {}".format(scene_name))
            print("COUNT: {}".format(count))
            count += 1
            
            if os.path.exists(test_save_path + scene_name + ".txt"):
                print("Already exist!")
                continue

            my_map, starts, goals = import_mapf_instance(scene)

            cbs = CBSSolver(my_map, starts, goals)
            cbs_paths, cbs_gen_nodes, cbs_exp_nodes, cbs_total_CT = cbs.find_solution()

            if cbs_paths == None:
                os.remove(scene)
            else:
                if cbs_gen_nodes < 10 or cbs_exp_nodes < 10:
                    print("Too short!")
                    print("=====================================================")
                    os.remove(scene)
                    continue
                
                icbs = ICBS_Solver_Compare(my_map, starts, goals)
                icbs_paths, icbs_gen_nodes, icbs_exp_nodes, icbs_total_CT = icbs.find_solution(False, "a_star")

                cbs_cost = get_sum_of_cost(cbs_paths)
                icbs_cost = get_sum_of_cost(icbs_paths)

                cbs_result = "{}\nCBS result!\nCost: {}, Gen nodes: {}, Exp nodes: {}, total CT: {}".format(scene_name, cbs_cost, cbs_gen_nodes, cbs_exp_nodes, cbs_total_CT)
                icbs_result = "\nICBS result!\nCost: {}, Gen nodes: {}, Exp nodes: {}, total CT: {}".format(icbs_cost, icbs_gen_nodes, icbs_exp_nodes, icbs_total_CT)

                test_result = cbs_result + icbs_result

                print(test_result)

                f = open(test_save_path + "{}.txt".format(scene_name), 'w')
                f.write(test_result)
                f.close()

            
            print("=====================================================")
    

if __name__=="__main__":
    main()