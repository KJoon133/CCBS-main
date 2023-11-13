from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os, glob, sys

def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        # print(path)
        rst += len(path) - 1
        if(len(path)>1):
            assert path[-1] != path[-2]
    return rst

def print_result(instance_name, cbs_results, ccbs_results):
    text = "Instance: {}\nCBS results!\nCost: {}, Gen_nodes: {}, Exp_nodes: {}, Total CT: {}, High level CT: {}\nCCBS results!\nCost: {}, Gen_nodes: {}, Exp_nodes: {}, Total CT: {}, High level CT: {}\n".format(instance_name, cbs_results[0], cbs_results[1], cbs_results[2], cbs_results[3], cbs_results[4], ccbs_results[0],ccbs_results[1],ccbs_results[2], ccbs_results[3], ccbs_results[4])
    return text

def import_mapf_instance(filename):
    # print(filename)
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
    
    # #agents
    line = f.readline()
    num_agents = int(line)

    # #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx,gy))

    return my_map, starts, goals

def dense_input_data(starts, goals):
    input_data = []

    all_data = [(sx, sy, gx, gy) for (sx, sy), (gx, gy) in zip(starts, goals)]

    for i, (start, goal) in enumerate(zip(starts, goals)):
        sx, sy = start
        gx, gy = goal
        dx, dy = gx-sx, gy-sy
        mag = np.sqrt(dx ** 2 + dy ** 2)

        if mag != 0:
            dx = dx / mag
            dy = dy / mag

        except_agent_data = all_data[:i] + all_data[i+1:]

        own_data = [sx, sy, gx, gy, dx, dy, mag]

        for (other_sx, other_sy, other_gx, other_gy) in except_agent_data:
            other_dx, other_dy = other_gx-other_sx, other_gy-other_sy
            other_mag = np.sqrt(other_dx ** 2 + other_dy ** 2)

            if other_mag != 0:
                other_dx = other_dx / other_mag
                other_dy = other_dy / other_mag

            own_data += [other_sx-sx, other_sy-sy, other_dx, other_dy, other_mag]

        own_data = np.array(own_data)
        input_data.append(own_data)
    
    return np.array(input_data)

def visualize_heatmap(heatmap_arrray, fig_title):
    h_shape = heatmap_arrray.shape  
    heatmap_arrray = heatmap_arrray.reshape(h_shape[0], h_shape[2], h_shape[3])
    
    fig, axs = plt.subplots(4, 5, figsize=(21, 26))
    for i, (ax, heatmap) in enumerate(zip(axs.flat, heatmap_arrray)):
        # if np.count_nonzero(heatmap) > 0:
        #     print(heatmap)
        ax.pcolor(heatmap[::], cmap='gray', vmin=0., vmax=1.0)
        ax.set_title("{}th agent heatmap".format(i))
    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout()

    plt.show()

def save_result(instance_name, text, case, method):
    if method == None:
        raise ValueError(method)
    
    if case == "cost_error":
        f = open("/home/railab/Workspace/CCBS/Results/Making_heatmap/test_68/" + method + "/cost_error/{}_result.txt".format(instance_name), 'w')
        f.write(text)
    elif case == "nodes_inc":
        f = open("/home/railab/Workspace/CCBS/Results/Making_heatmap/test_68/" + method + "/nodes_inc/{}_result.txt".format(instance_name), 'w')
        f.write(text)
    elif case == "nodes_dec":
        f = open("/home/railab/Workspace/CCBS/Results/Making_heatmap/test_68/" + method + "/nodes_dec/{}_result.txt".format(instance_name), 'w')
        f.write(text)
    else:
        print("CASE ERROR!")

    f.close()

def get_scene_all(test_name, method):
    good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/all/test*")

    scenes = glob.glob("/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/*")
    good_result_num_list = [file.split("_")[-2] for file in good_result_file_list]
    scene_num_list = [scene.split("_")[-1][:-4] for scene in scenes]

    removed_num_list = []
    for scene_num in scene_num_list:
        if scene_num in good_result_num_list:
            removed_num_list.append(scene_num)
    removed_scenes_list = ["/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/" + test_name + "_" + scene + ".txt" for scene in removed_num_list]
    
    print("Total {} data loaded!".format(len(removed_scenes_list)))
    return removed_scenes_list

def get_scene_list(test_name, method, is_train):
    if is_train == True:
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/train_data/test*")
        mode = "Train"
    elif is_train == False:
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/test_data/test*")
        mode = "Test"
    else:
        raise ValueError(is_train)

    scenes = glob.glob("/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/*")

    good_result_num_list = [file.split("_")[-2] for file in good_result_file_list]
    scene_num_list = [scene.split("_")[-1][:-4] for scene in scenes]

    removed_num_list = []
    for scene_num in scene_num_list:
        if scene_num in good_result_num_list:
            removed_num_list.append(scene_num)
    removed_scenes_list = ["/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/" + test_name + "_" + scene + ".txt" for scene in removed_num_list]

    print("Total {} data loaded\nNow mode: {}!".format(len(removed_scenes_list), mode))

    return removed_scenes_list

def get_scene_list_with_dataset(test_name, method, dataset):
    if dataset == "train":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/train/test*")
    elif dataset == "valid":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/valid/test*")
    elif dataset == "test":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/test/test*")
    else:
        raise ValueError(dataset)

    scenes = glob.glob("/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/*")

    good_result_num_list = [file.split("_")[-2] for file in good_result_file_list]
    scene_num_list = [scene.split("_")[-1][:-4] for scene in scenes]

    removed_num_list = []
    for scene_num in scene_num_list:
        if scene_num in good_result_num_list:
            removed_num_list.append(scene_num)
    removed_scenes_list = ["/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/" + test_name + "_" + scene + ".txt" for scene in removed_num_list]

    print("Total {} data loaded\nNow mode: {}!".format(len(removed_scenes_list), dataset))

    return removed_scenes_list


def get_scene_19(test_name, method, dataset):
    if dataset == "train":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/train_19/test*")
    elif dataset == "valid":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/valid_1/test*")
    else:
        raise ValueError(dataset)

    scenes = glob.glob("/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/*")

    good_result_num_list = [file.split("_")[-2] for file in good_result_file_list]
    scene_num_list = [scene.split("_")[-1][:-4] for scene in scenes]

    removed_num_list = []
    for scene_num in scene_num_list:
        if scene_num in good_result_num_list:
            removed_num_list.append(scene_num)
    removed_scenes_list = ["/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/" + test_name + "_" + scene + ".txt" for scene in removed_num_list]

    print("Total {} data loaded\nNow mode: {}!".format(len(removed_scenes_list), dataset))

    return removed_scenes_list

def get_scene_80(test_name, method, dataset):
    if dataset == "train":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/train_80/test*")
    elif dataset == "valid":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/valid_10/test*")
    elif dataset == "test":
        good_result_file_list = glob.glob("/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/test_10/test*")
    else:
        raise ValueError(dataset)

    scenes = glob.glob("/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/*")

    good_result_num_list = [file.split("_")[-2] for file in good_result_file_list]
    scene_num_list = [scene.split("_")[-1][:-4] for scene in scenes]

    removed_num_list = []
    for scene_num in scene_num_list:
        if scene_num in good_result_num_list:
            removed_num_list.append(scene_num)
    removed_scenes_list = ["/home/railab/Workspace/CCBS/code/instances/generated/" + test_name + "/" + test_name + "_" + scene + ".txt" for scene in removed_num_list]

    print("Total {} data loaded for {}!".format(len(removed_scenes_list), dataset))

    return removed_scenes_list


def get_cbs_result(test_name, method, scene_name):
    result_file_path = "/home/railab/Workspace/CCBS/Results/Making_heatmap/" + test_name + "/" + method + "/nodes_dec/test_10/" + scene_name + ".txt"

    f = open(result_file_path, 'r')
    contents = f.readlines()
    print(contents)

def compare_target_pred(target, pred):

    if len(target.shape) != 3:
        target = target.reshape(target.shape[0], target.shape[2], target.shape[3])
    
    if len(pred.shape) != 3:
        pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[3])
    
    fig, axs = plt.subplots(4, 10, figsize=(21, 26))
    for i, ax, target_, pred_ in zip(range(0, len(axs.flat), 2), axs.flat, target, pred):
            agent_num = int(i/2 + 1)
            axs.flat[i].pcolor(target_[::], cmap='gray', vmin=0., vmax=1.0)
            axs.flat[i].set_title("{}th agent target".format(agent_num))

            axs.flat[i+1].pcolor(pred_[::], cmap='gray', vmin=0., vmax=1.0)
            axs.flat[i+1].set_title("{}th agent pred".format(agent_num))

    fig.tight_layout()

    plt.show()

def save_target_pred(target, pred, scene_name):

    if len(target.shape) != 3:
        target = target.reshape(target.shape[0], target.shape[2], target.shape[3])
    
    if len(pred.shape) != 3:
        pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[3])
    
    fig, axs = plt.subplots(4, 10, figsize=(21, 26))
    for i, ax, target_, pred_ in zip(range(0, len(axs.flat), 2), axs.flat, target, pred):
            agent_num = int(i/2 + 1)
            axs.flat[i].pcolor(target_[::], cmap='gray', vmin=0., vmax=1.0)
            axs.flat[i].set_title("Target")

            axs.flat[i+1].pcolor(pred_[::], cmap='gray', vmin=0., vmax=1.0)
            axs.flat[i+1].set_title("Prediction")

    fig.tight_layout()

    plt.savefig("/home/railab/Workspace/CCBS/Train_results/Overfitting/target_pred_result/"+ scene_name +".png", dpi=200)
    plt.close()


def compare_target_pred_paper(target, pred):

    if len(target.shape) != 3:
        target = target.reshape(target.shape[0], target.shape[2], target.shape[3])
    
    if len(pred.shape) != 3:
        pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[3])
    
    fig, axs = plt.subplots(2, 4, figsize=(21, 26))
    for i, ax, target_, pred_ in zip(range(0, len(axs.flat), 2), axs.flat, target, pred):
            if i == 8:
                break
            agent_num = int(i/2 + 1)
            axs.flat[i].pcolor(target_[::], cmap='gray', vmin=0., vmax=1.0)
            axs.flat[i].set_title("{}th agent target".format(agent_num))

            axs.flat[i+1].pcolor(pred_[::], cmap='gray', vmin=0., vmax=1.0)
            axs.flat[i+1].set_title("{}th agent pred".format(agent_num))

    fig.tight_layout()

    plt.show()



def dense_input_data(starts, goals):
    input_data = []

    all_data = [(sx, sy, gx, gy) for (sx, sy), (gx, gy) in zip(starts, goals)]

    for i, (start, goal) in enumerate(zip(starts, goals)):
        sx, sy = start
        gx, gy = goal
        dx, dy = gx-sx, gy-sy
        mag = np.sqrt(dx ** 2 + dy ** 2)

        if mag != 0:
            dx = dx / mag
            dy = dy / mag

        except_agent_data = all_data[:i] + all_data[i+1:]

        own_data = [sx, sy, gx, gy, dx, dy, mag]

        for (other_sx, other_sy, other_gx, other_gy) in except_agent_data:
            other_dx, other_dy = other_gx-other_sx, other_gy-other_sy
            other_mag = np.sqrt(other_dx ** 2 + other_dy ** 2)

            if other_mag != 0:
                other_dx = other_dx / other_mag
                other_dy = other_dy / other_mag

            own_data += [other_sx-sx, other_sy-sy, other_dx, other_dy, other_mag]

        own_data = np.array(own_data)
        input_data.append(own_data)
    
    return np.array(input_data)

def inverse_sigmoid(y):
    return np.log(y/(1-y))

def main():
    return

if __name__ =="__main__":
    main()
