import sys, os, argparse
import numpy as np

sys.path.append("/home/railab/Workspace/CCBS/code")

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)

from core.func.cbs_basic import CBSSolver
from core.utils.utils import get_scene_list, import_mapf_instance, get_scene_80

def main(test_name):
    train_scene_list = get_scene_80(test_name, "bigger_heatmap", "train")
    test_scene_list = get_scene_80(test_name, "bigger_heatmap", "test")
    valid_scene_list = get_scene_80(test_name, "bigger_heatmap", "valid")

    scene_list = train_scene_list + test_scene_list + valid_scene_list

    for scene in scene_list:
        instance_name = scene.split("/")[-1][:-4]
        method = "bigger_heatmap"

        save_path = "/home/railab/Workspace/CCBS/code/h_value_maps/{}/{}/{}.npy".format(test_name, method, instance_name)

        if os.path.exists(save_path):
            print("Already exist!!")
        else:
            map, starts, goals = import_mapf_instance(scene)
            num_agents = len(starts)

            obs_map = np.array([[int(val) for val in sublist] for sublist in map])
            obs_map = np.repeat(obs_map[np.newaxis,:,:], num_agents, axis=0)

            cbs_solver = CBSSolver(map, starts, goals)
            h_map = cbs_solver.heuristics

            h_map_array = np.zeros_like(obs_map)
            for i, h_val in enumerate(h_map):
                for key, val in h_val.items():
                    h_map_array[i][key[0]][key[1]] = val
                h_map_array[i] = np.where(h_map_array[i][::] == 0, 100, h_map_array[i][::])
                h_map_array[i][goals[i][0]][goals[i][1]] = 0
                h_map_array[i][starts[i][0]][starts[i][1]] = 0

            # /home/railab/Workspace/CCBS/code/h_value_maps/sorted_normalized_nonvertex
            print(save_path)

            np.save(save_path, h_map_array)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default=None)
    args = parser.parse_args()

    if args.test != None:
        main(args.test)
    else:
        print("Please write argument!")
        raise ValueError(args.test)