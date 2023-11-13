import os, random, glob, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(linewidth=sys.maxsize)

from core.utils.utils import import_mapf_instance, inverse_sigmoid, dense_input_data, get_scene_80
from core.model.Unet import UNet_2D_256

device = 'cuda:0'# if torch.cuda.is_available() else 'cpu'

def train(train_scene_list, BATCH_SIZE, model, loss_f, optm):
    avg_train_loss = 0.

    for train_scene in train_scene_list:
        # Train Start
        # Get data
        test_name = train_scene.split("/")[-2]
        instance_name = train_scene.split("/")[-1][:-4]

        map, starts, goals = import_mapf_instance(train_scene)
        agent_num = len(starts)

        obs_map = np.array([[int(val) for val in sublist] for sublist in map])
        obs_map = np.repeat(obs_map[np.newaxis,:,:], agent_num, axis=0)

        heatmap_path = "/home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/{}_heatmap/bigger_heatmap/{}.npy".format(test_name, instance_name)
        heatmap_data = np.load(heatmap_path, allow_pickle=True)

        h_map_path = "/home/railab/Workspace/CCBS/code/h_value_maps/{}/bigger_heatmap/{}.npy".format(test_name, instance_name)
        h_map_array = np.load(h_map_path, allow_pickle=True)

        # sorting datas
        start_goals = [[(sx, sy), (gx, gy), heatmap, h_val_map] for ind, ((sx, sy), (gx, gy), heatmap, h_val_map) in enumerate(zip(starts, goals, heatmap_data, h_map_array))]
        start_goals.sort(key=lambda x:(x[0], x[1]))
        sorted_starts = np.array([(sx, sy) for [(sx, sy), (gx, gy), heatmap, h_val_map] in start_goals])
        sorted_goals = np.array([(gx, gy) for [(sx, sy), (gx, gy), heatmap, h_val_map] in start_goals])
        sorted_heatmap_array = np.array([heatmap for [(sx, sy), (gx, gy), heatmap, h_val_map] in start_goals])
        sorted_h_val_map_array = np.array([h_val_map for [(sx, sy), (gx, gy), heatmap, h_val_map] in start_goals])

        input_data = dense_input_data(sorted_starts, sorted_goals)

        # Shuffle input, target pair
        data_pair_list = [[i, input, target, h_map] for i, (input, target, h_map) in enumerate(zip(input_data, sorted_heatmap_array, sorted_h_val_map_array))]
        random.shuffle(data_pair_list)
        input_list = []
        target_list = []
        h_val_list = []
        
        for pair in data_pair_list:
            input_list.append(pair[1])
            target_list.append(pair[2])
            h_val_list.append(pair[3])

        input = np.array(input_list)
        h_val = np.array(h_val_list)
        target_ = np.array(target_list)
        target_ = np.where(target_==0, 1e-4, target_)
        target_ = np.where(target_>=1, 1-(1e-4), target_)

        inverse_target = inverse_sigmoid(target_)

        obs_map = torch.reshape(torch.Tensor(obs_map[:BATCH_SIZE]), heatmap_data[:BATCH_SIZE].shape).to(device)
        h_val = torch.reshape(torch.Tensor(h_val[:BATCH_SIZE]), heatmap_data[:BATCH_SIZE].shape).to(device)

        conv_data = torch.cat((obs_map, h_val), axis=1)

        input_data = torch.Tensor(input[:BATCH_SIZE]).to(device)
        target = torch.Tensor(inverse_target[:BATCH_SIZE]).to(device)

        # print(target.shape)
        # print("min target:", torch.min(target))
        # print("max target:", torch.max(target))
        # return

        pred = model(conv_data, input_data)
        # pred = torch.reshape(pred, target.shape)

        loss = loss_f(pred, target)
        
        optm.zero_grad()
        loss.backward()
        optm.step()

        avg_train_loss += loss.item()/len(train_scene_list)

    return avg_train_loss
    
def valid(valid_scene_list, BATCH_SIZE, model, loss_f):
    ## Train dataset End
    ## Valid Start
    # if epoch % INTERVAL == 0:
    avg_valid_loss = 0.

    with torch.no_grad():
        for valid_scene in valid_scene_list:
            test_name = valid_scene.split("/")[-2]
            instance_name = valid_scene.split("/")[-1][:-4]

            map, starts, goals = import_mapf_instance(valid_scene)
            num_agents = len(starts)

            obs_map = np.array([[int(val) for val in sublist] for sublist in map])
            obs_map = np.repeat(obs_map[np.newaxis,:,:], num_agents, axis=0)

            start_goals = [[(sx, sy), (gx, gy)] for ind, ((sx, sy), (gx, gy)) in enumerate(zip(starts, goals))]
            start_goals.sort(key=lambda x:(x[0], x[1]))
            sorted_starts = np.array([(sx, sy) for [(sx, sy), (gx, gy)] in start_goals])
            sorted_goals = np.array([(gx, gy) for [(sx, sy), (gx, gy)] in start_goals])
            input_data = dense_input_data(sorted_starts, sorted_goals)

            heatmap_path = "/home/railab/Workspace/CCBS/code/costmaps/Gaussian_Blur/{}_heatmap/bigger_heatmap/{}.npy".format(test_name, instance_name)
            heatmap_data = np.load(heatmap_path, allow_pickle=True)

            h_map_path = "/home/railab/Workspace/CCBS/code/h_value_maps/{}/bigger_heatmap/{}.npy".format(test_name, instance_name)
            h_map_array = np.load(h_map_path, allow_pickle=True)

            # Shuffle input, target pair
            data_pair_list = [[i, input, target, h_map] for i, (input, target, h_map) in enumerate(zip(input_data, heatmap_data, h_map_array))]
            random.shuffle(data_pair_list)
            input_list = []
            target_list = []
            h_val_list = []
            
            for pair in data_pair_list:
                input_list.append(pair[1])
                target_list.append(pair[2])
                h_val_list.append(pair[3])

            input = np.array(input_list)
            h_val = np.array(h_val_list)
            target_ = np.array(target_list)
            target_ = np.where(target_==0, 1e-4, target_)
            target_ = np.where(target_>=1, 1-(1e-4), target_)

            inverse_target = inverse_sigmoid(target_)

            obs_map = torch.reshape(torch.Tensor(obs_map[:BATCH_SIZE]), heatmap_data[:BATCH_SIZE].shape).to(device)
            h_val = torch.reshape(torch.Tensor(h_val[:BATCH_SIZE]), heatmap_data[:BATCH_SIZE].shape).to(device)

            conv_data = torch.cat((obs_map, h_val), axis=1)

            input_data = torch.Tensor(input[:BATCH_SIZE]).to(device)
            target = torch.Tensor(inverse_target[:BATCH_SIZE]).to(device)

            pred = model(conv_data, input_data)

            loss = loss_f(pred, target)
            avg_valid_loss += loss.item()/len(valid_scene_list)
    
    return avg_valid_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--l2", type=float, default=None)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.l2
    VALID_INTERVAL = 10

    train_scene_list = get_scene_80("test_62", "bigger_heatmap", dataset="train")
    valid_scene_list = get_scene_80("test_62", "bigger_heatmap", dataset="valid")

    if args.model == None:

        print("Model not selected!!")
        raise ValueError("'args.model' is None")
    
    else:
        
        # if args.model == "Large":
        #     model = cnn_mlp_large(202, 65536).to(device)
        #     model_name = "cnn_mlp_Large"
        
        if args.model == "unet_256":
            model = UNet_2D_256(2, 1, 202).to(device)
            model_name = "unet_256"

        else:
            raise ValueError("'args.model' Unknown model")
    
    if WEIGHT_DECAY is None:
        optm = torch.optim.Adam(model.parameters(), lr=LR)
        
    else:
        optm = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    optm_name = type(optm).__name__
    
    loss_function = nn.MSELoss()
    loss_name = loss_function._get_name()

    ### PLOT ##################
    print("\n############################################")
    print("Device:     {}".format(device))
    print("Model:      {}".format(model_name))
    print("Optimizer:  {}".format(optm_name))
    print("Loss func:  {}".format(loss_name))
    print("L2 lambda:  {}".format(WEIGHT_DECAY))
    print("Epoch:      {}".format(EPOCHS))
    print("Batch_size: {}".format(BATCH_SIZE))
    print("LR:         {}".format(LR))
    print("############################################\n")

    save_path = "/home/railab/Workspace/CCBS/code/model_params/{}/{}/L2_{}".format(model_name, loss_name, WEIGHT_DECAY)

    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    except OSError:
        print("Error: Failed to create the directory.")

    # Load model and get before loss list
    if os.path.exists(save_path):
        saved_model_list = glob.glob(save_path + "/*.pth")

        if len(saved_model_list) == 0:
            previous_epochs = 0
            train_loss_list = []
            valid_loss_list = []

            print("There is no trained model!")

        else:
            last_model_file_name = saved_model_list[-1]
            previous_epochs = int(last_model_file_name.split("_")[-1][:-4])

            train_loss_path_list = glob.glob(save_path + "/*train*.txt")[0]
            valid_loss_path_list = glob.glob(save_path + "/*valid*.txt")[0]
            train_loss_list = []
            valid_loss_list = []

            train_f = open(train_loss_path_list, "r")
            train_loss_lines = train_f.readlines()
            train_loss_list = [float(saved_loss.rstrip()) for saved_loss in train_loss_lines]

            train_f.close()

            valid_f = open(valid_loss_path_list, "r")
            valid_loss_lines = valid_f.readlines()
            valid_loss_list = [float(saved_loss.rstrip()) for saved_loss in valid_loss_lines]

            valid_f.close()

            min_loss_model = saved_model_list[-1]

            try:
                model.load_state_dict(torch.load(min_loss_model))
                # print("File name: {}".format(min_loss_model.split("/")[-1]))
                print("Model loaded")

            except:
                print("Model load Error!")
    else:
        previous_epochs = 0
        train_loss_list = []
        valid_loss_list = []

        print("There is no trained model!")

    min_loss = np.inf

    # Train/Valid Iteration
    for epoch in range(previous_epochs + 1, EPOCHS - previous_epochs + 1):
        ### TRAIN ##################
        random.shuffle(train_scene_list)

        model.train()
        avg_train_loss = train(train_scene_list, BATCH_SIZE, model, loss_function, optm)
        
        train_loss_list.append(avg_train_loss)
        
        print("Epoch: {} / {}\nTrain loss: {}".format(epoch, EPOCHS + previous_epochs, avg_train_loss))

        if min_loss > avg_train_loss:
            model_list = glob.glob(save_path + "/*.pth")
            if len(model_list) != 0:
                os.remove(model_list[0])
            torch.save(model.state_dict(), save_path + "/batch_{}_lr_{}_epoch_{}.pth".format(BATCH_SIZE, LR, epoch))
            min_loss = avg_train_loss

        ### VALID ###################
        model.eval()
        avg_valid_loss = valid(valid_scene_list, BATCH_SIZE, model, loss_function)

        valid_loss_list.append(avg_valid_loss)

        print("Epoch: {} / {}\nValid loss: {}".format(epoch, EPOCHS + previous_epochs, avg_valid_loss))

        np.savetxt(save_path + "/{}_train_loss.txt".format(model_name), train_loss_list, delimiter=',')

        if epoch % 10 == 0:

            np.savetxt(save_path + "/{}_valid_loss.txt".format(model_name), valid_loss_list, delimiter=',')


if __name__ == "__main__":
    main()