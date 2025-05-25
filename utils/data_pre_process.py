import numpy as np
import os
import dill
import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

class custom_subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The subset Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.targets = labels
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)

# split for federated settings
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


################################### data setup ########################################
def load_partition(args):
    dict_users = []
    # read dataset
    if args.dataset == 'generator_defect_classification':
        path = './data/dataset/generator_defect_classification'
        trans_custom = transforms.Compose([
            transforms.Resize((64, 64)),  # ResNet expects 224x224 input
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet pretrained stats
                                std=[0.229, 0.224, 0.225])
        ])
        train_path = os.path.join(path, 'train_data')
        test_path = os.path.join(path, 'test_data')
        dataset_train = datasets.ImageFolder(root=train_path, transform=trans_custom)
        dataset_train.samples = random.sample(dataset_train.samples, len(dataset_train.samples))
        dataset_test = datasets.ImageFolder(root=test_path, transform=trans_custom)

        # Set number of classes based on the training dataset
        args.num_classes = len(dataset_train.classes) # Should be 2 for 'clean_images' and 'damaged_images'
        print(f"Dataset: generator_defect_classification, Number of classes: {args.num_classes}")
        print(f"Training classes: {dataset_train.classes}")
        print(f"Test classes: {dataset_test.classes}")

        dict_users = {}
        pik_path = os.path.join(path, 'generator_defect_dict_users.pik') # Unique name for this dataset's pik file

        if os.path.isfile(pik_path):
            print(f"Attempting to load user data split from: {pik_path}")
            try:
                with open(pik_path, 'rb') as f:
                    loaded_dict = dill.load(f)
                    if isinstance(loaded_dict, dict):
                        dict_users = loaded_dict
                        if not dict_users:
                            print(f"Info: Loaded an empty dictionary from {pik_path}.")
                    else:
                        print(f"Warning: Data in {pik_path} is not a dictionary. Will regenerate.")
            except Exception as e:
                print(f"Warning: Could not load {pik_path} ({e}). Will regenerate.")
        
        if len(dict_users) < 1:
            print("Generating new user data split...")
            if args.iid:
                # Assuming iid function is defined elsewhere and takes (dataset, num_users)
                dict_users = iid(dataset_train, args.num_users)
                print("Generated IID data split.")
            else:
                # Assuming noniid function is defined elsewhere.
                # For noniid, class_num might be args.num_classes or a specific strategy parameter.
                # Using args.noniid_clsnum as in the fmnist example.
                dict_users = noniid(dataset_train, args.num_users, class_num=args.noniid_clsnum)
                print(f"Generated Non-IID data split with noniid_clsnum={args.noniid_clsnum}.")

            if args.freeze_datasplit:
                try:
                    with open(pik_path, 'wb') as f:
                        dill.dump(dict_users, f)
                    print(f"Saved new user data split to: {pik_path}")
                except Exception as e:
                    print(f"Error: Could not save data split to {pik_path} ({e}).")
        else:
            print(f"Successfully loaded user data split for {len(dict_users)} users.")

    elif args.dataset == 'ac_microgrid_timeseries':
        path = './data/dataset/ac_microgrid_timeseries'

        def norm_tabular(data, norm_type='z-score'):
            len_dim = data.size(0)
            client_dim = data.size(-2)
            attr_dim = data.size(-1)
            data = data.view(-1, attr_dim)

            if norm_type == 'min-max':
                # Min-Max Scaling
                min_vals = data[:, :-1].min(dim=0)[0]
                max_vals = data[:, :-1].max(dim=0)[0]
                x_train_scaled = (data[:, :-1] - min_vals) / (max_vals - min_vals)
            elif norm_type == 'z-score':
                # Perform Z-score Standardization
                mean_vals = data[:, :-1].mean(dim=0)
                std_vals = data[:, :-1].std(dim=0)
                x_train_scaled = (data[:, :-1] - mean_vals) / std_vals

            normed_data = torch.concatenate((x_train_scaled, data[:, -1].unsqueeze(-1)), dim=1)
            return normed_data

        class SGAttackDataset(torch.utils.data.Dataset):   
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return self.x.__len__()

            def __getitem__(self, index):
                raw_data = self.x[index]
                labels = self.y[index].type(torch.LongTensor)
                # for p in range(len(raw_data) - self.window):
                #     train_seq = raw_data[p:p+self.window, :, :]
                #     train_label = labels[p:p+self.window]
                #     seq_data.append(train_seq)
                #     seq_label.append(train_label)
                # return torch.stack(seq_data), torch.stack(seq_label)
                return raw_data, labels

        data = []
        for i in range(1, 4):
            df = pd.read_csv(os.path.join(path, 'dg'+str(i)+'_data_noisy9.csv'))
            data_array = df.to_numpy()
            data.append(data_array)
        data = torch.Tensor(np.concatenate(data, axis=0)).to(args.device)
        # normalization
        data = norm_tabular(data)

        train_indices = np.random.choice(np.arange(len(data)), size=int(len(data) * 0.8), replace=False)
        val_test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
        val_indices = np.random.choice(val_test_indices, size=int(len(data) * 0.1), replace=False)
        test_indices = np.setdiff1d(val_test_indices, val_indices)

        dataset_train = SGAttackDataset(data[train_indices, :-1], data[train_indices, -1])
        dataset_val = SGAttackDataset(data[val_indices, :-1], data[val_indices, -1])
        dataset_test = SGAttackDataset(data[test_indices, :-1], data[test_indices, -1])

        '''
        train_df = pd.read_csv(os.path.join(path, 'train_data.csv'))
        train_data = train_df.to_numpy()
        train_data = torch.Tensor(train_data).to(args.device)
        train_data = norm_tabular(train_data)

        train_selected = []
        test_selected = []
        for label in range(5):
            indices = (train_data[:, -1] == label).nonzero(as_tuple=True)[0]
            train_selected.append(train_data[indices[:20000]])
            test_selected.append(train_data[indices[20000:25000]])
        train_data = torch.cat(train_selected, dim=0)
        test_data = torch.cat(test_selected, dim=0)

        # Shuffle train_data
        perm = torch.randperm(train_data.size(0))
        train_data = train_data[perm]

        # Shuffle test_data
        perm = torch.randperm(test_data.size(0))
        test_data = test_data[perm]

        # test_df = pd.read_csv(os.path.join(path, 'test_data.csv'))
        # test_data = test_df.to_numpy()
        # test_data = torch.Tensor(test_data).to(args.device)
        # test_data = norm_tabular(test_data)

        # selected = []
        # for label in range(5):
        #     indices = (test_data[:, -1] == label).nonzero(as_tuple=True)[0]
        #     selected_indices = indices[:100]  
        #     selected.append(test_data[selected_indices])
        # test_data = torch.cat(selected, dim=0)

        dataset_train = SGAttackDataset(train_data[:, :-1], train_data[:, -1])
        dataset_val = SGAttackDataset(train_data[:, :-1], train_data[:, -1])
        dataset_test = SGAttackDataset(test_data[:, :-1], test_data[:, -1])
        '''
        
        args.num_classes = 5

        pik_path = os.path.join(path,'ac_microgrid_timeseries_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                dict_users = noniid(dataset_train, args.num_users, class_num=args.noniid_clsnum)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
    
    elif args.dataset == 'electricity_theft_detection':
        # Load dataset
        path = "data/dataset/electricity_theft_detection/balance_data.csv"
        data = pd.read_csv(path)

        y_df = data['label']
        x_df = data.drop(['label'], axis=1)

        x = np.array(x_df)
        y = np.array(y_df) # These are the class indices (0, 1, ...)
        print("Dataset loaded and processed.")
        print(f"x shape: {x.shape}, y shape: {y.shape}, Unique y labels: {np.unique(y)}")

        # Split dataset
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
        
        x_train_np = x_train_np.reshape(x_train_np.shape[0], x_train_np.shape[1], 1)
        x_test_np = x_test_np.reshape(x_test_np.shape[0], x_test_np.shape[1], 1)
        
        print(f"x_train shape: {x_train_np.shape}, y_train shape: {y_train_np.shape}")
        print(f"x_test shape: {x_test_np.shape}, y_test shape: {y_test_np.shape}")

        train_x = torch.from_numpy(x_train_np).float().to(args.device)
        train_y = torch.from_numpy(y_train_np).long().to(args.device) # Use long for CrossEntropyLoss
        test_x = torch.from_numpy(x_test_np).float().to(args.device)
        test_y = torch.from_numpy(y_test_np).long().to(args.device)

        dataset_train = TensorDataset(train_x, train_y)
        dataset_val = TensorDataset(test_x, test_y)
        dataset_test = TensorDataset(test_x, test_y)
        
        args.num_classes = 2

        pik_path = os.path.join(path,'electricity_theft_detection_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                dict_users = noniid(dataset_train, args.num_users, class_num=args.noniid_clsnum)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
    
    else:
        exit('Error: unrecognized dataset')
    
    ## extract 10% data from test set for validation, and the rest for testing
    print("Creating validation dataset from testing dataset...")
    dataset_test, dataset_val = torch.utils.data.random_split(dataset_test, [len(dataset_test)-int(0.1 * len(dataset_test)), int(0.1 * len(dataset_test))])
    ## generate a public dataset for DP-topk purpose from validation set
    dataset_test, dataset_val = dataset_test.dataset, dataset_val.dataset
    # print("Creating public dataset...")
    # dataset_public = public_iid(dataset_val, args) # make sure public set has every class
    ## make sure experiments with different sizes of public dataset use the same testing data and training data

    # return args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users
    return args, dataset_train, dataset_test, dataset_val, None, dict_users

###################### utils #################################################
## IID assign data samples for num_users (mnist, svhn, fmnist, emnist, cifar)
def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    print("Assigning training data samples (iid)")
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

## IID assign data samples for num_users (mnist, emnist, cifar); each user only has n(default:two) classes of data
def noniid(dataset, num_users, class_num=2):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: each user only has two classes of data
    """
    print("Assigning training data samples (non-iid)")
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, class_num, replace=False))
        if num_users <= num_shards:
            idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

## generate a iid public dataset from dataset. 
def public_iid(dataset, args):
    """
    Sample I.I.D. public data from fashion MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if args.dataset == 'fmnist':
        labels = dataset.train_labels.numpy()
    elif args.dataset == 'cifar':
        labels = np.array(dataset.targets)
    else:
        labels = dataset.labels
    pub_set_idx = set()
    if args.pub_set > 0:
        for i in list(set(labels)):
            pub_set_idx.update(
                set(
                np.random.choice(np.where(labels==i)[0],
                                          int(args.pub_set/len(list(set(labels)))), 
                                 replace=False)
                )
                )
    # test_set_idx = set(np.arange(len(labels)))
    # test_set_idx= test_set_idx.difference(val_set_idx)
    return DatasetSplit(dataset, pub_set_idx)

def sample_dirichlet_train_data(dataset, args, no_participants, alpha=0.9):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if ind in args.poison_images or ind in args.poison_images_test:
            continue
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]
    class_size = len(cifar_classes[0])
    per_participant_list = {}
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            if user in per_participant_list:
                per_participant_list[user].extend(sampled_list)
            else:
                per_participant_list[user] = sampled_list
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

    return per_participant_list
