# remove the "liveness_datasets." prefix for use without Jupyter notebooks
from liveness_datasets.liveness_dataset import LivenessDataset
# from liveness_dataset import LivenessDataset
from torch.utils.data import DataLoader, ConcatDataset

import os

def get_loaders(data_dir, protocol, batch_size=64, shuffle=True, num_workers=1,
        include_devel=False, transform_src=None, transform_depth=None,
        transform_target=None, debug=False):
    """
    returns a list of tuples with one train and one test loader each,
    corresponding to the chosen protocol
    """
    train_lists, test_lists = get_protocol_parameters(protocol)
    loaders = []

    for idx in range(len(train_lists)):
        train_list, test_list = train_lists[idx], test_lists[idx]

        train_ds, test_ds = get_liveness_datasets(train_list, test_list,
            batch_size, dsdir=data_dir, include_devel=False,
            transform=transform_src, depth_transform=transform_depth,
            target_transform=transform_target, shuffle=shuffle,
            num_workers=num_workers, debug=False)

        train_ldr = DataLoader(train_ds, batch_size=batch_size,
                shuffle=shuffle, num_workers=num_workers)
        test_ldr = DataLoader(test_ds, batch_size=batch_size,
                shuffle=shuffle, num_workers=num_workers)

        loaders.append((train_ldr, test_ldr))
    return loaders

def get_liveness_datasets(
        train_datasets, test_datasets, batch_size, dsdir, include_devel=False,
        transform=None, depth_transform=None, target_transform=None,
        shuffle=True, num_workers=1, debug=False):
    """
    get train and test datasets according to specified dataset names

    {train,test}_datasets: list of datasets
        supported datasets: casia-fasd, replay-attack, msu-mfsd, oulu-npu,
                            oulu-npu-[protocol number]-[protocol step]
    dsdir: directory where datasets can be found
    include_devel: include labels_devel.csv for training with replay-attack or
                   oulu-npu
    """
    def ds_directory(ds):
        """get actual directory name for each dataset"""
        if ds == "casia-fasd":
            return "casia-new"
        elif ds == "msu-mfsd":
            return "msu-new"
        elif ds == "replay-attack":
            return "replay-new"
        else: # oulu-npu
            return "oulu-new"
    def get_csv_name(ds, phase):
        """phase is dev[el], train or test"""
        name = f"labels_{phase}"
        if "oulu" in ds:
            protocol_number = ds.split('-')[2]
            protocol_step = ds.split('-')[3]
            name += f"{protocol_step}_p{protocol_number}"
        name += ".csv"
        return name
    ds_path = lambda ds : os.path.join(dsdir, ds_directory(ds))
    # LivenessDataset parameters
    train_img_dirs, train_annotation_files = [], []
    test_img_dirs, test_annotation_files = [], []
    # train dirs and csv files
    for ds in train_datasets:
        train_img_dirs.append(ds_path(ds))
        train_annotation_files.append(get_csv_name(ds, "train"))
    # add dev file for oulu-npu and replay-attack
    if include_devel:
        dev_file = {
            "oulu-npu": "dev",
            "replay-attack": "devel"
            }
        # all datasets in the train list that have dev/devel files
        intersection = [ds for ds in dev_file.keys() if ds in train_datasets]
        for ds in intersection:
            train_img_dirs.append(ds_path(ds))
            train_annotation_files.append(get_csv_name(ds, dev_file[ds]))
    # test dirs and csv files
    for ds in test_datasets:
        test_img_dirs.append(ds_path(ds))
        test_annotation_files.append(get_csv_name(ds, "test"))
    if debug: # returns LivenessDataset parameters just created
        return (train_img_dirs, train_annotation_files, test_img_dirs,
               test_annotation_files)
    train_ds, test_ds = [], []
    for img_dir, csv_file in zip(train_img_dirs, train_annotation_files):
        train_ds.append(LivenessDataset(
            img_dir, csv_file, transform=transform,
            depth_transform=depth_transform,
            target_transform=target_transform))
    for img_dir, csv_file in zip(test_img_dirs, test_annotation_files):
        test_ds.append(LivenessDataset(
            img_dir, csv_file, transform=transform,
            depth_transform=depth_transform,
            target_transform=target_transform))
    return ConcatDataset(train_ds), ConcatDataset(test_ds)

def get_protocol_parameters(protocol):
    """
    returns list of tuples (train_datasets, test_datasets) to be used as
    parameters to the get_liveness_datasets function according to the
    protocol in use

    protocol values:
    - intra-(dataset name): intra-dataset protocol for given dataset
        - notice that intra-oulu-npu will use the entire train and test (and
          possibly dev) sets in the oulu-npu dataset
    - intra-crm: intra-dataset on casia-fasd, replay-attack and msu-mfsd
    - oulu-npu-(protocol number): oulu-npu protocol
    - cross-fast: casia-fasd vs replay-attack
    - cross-full: NxN (pairs) on casia-fasd, replay-attack and msu-mfsd
    - loo-crm: Leave-One-Out on casia-fasd, replay-attack and msu-mfsd
    - loo-cimo: LOO on casia-fasd, replay-attack, msu-mfsd and oulu-npu
    """
    casia_fasd = "casia-fasd"
    msu_mfsd = "msu-mfsd"
    replay_attack = "replay-attack"
    oulu_npu = "oulu-npu"
    crm = [casia_fasd, replay_attack, msu_mfsd]
    cimo = crm + [oulu_npu]
    all_datasets = [casia_fasd, msu_mfsd, replay_attack, oulu_npu]
    train_datasets, test_datasets = [], []
    if protocol[:5] == "intra":
        if protocol == "intra-crm":
            for x in crm:
                train_datasets.append([x])
                test_datasets.append([x])
        else:
            ds = protocol[6:] # everything after "intra-"
            train_datasets.append([ds])
            test_datasets.append([ds])
    elif protocol == "cross-fast":
        train_datasets = [[casia_fasd], [replay_attack]]
        test_datasets = [[replay_attack], [casia_fasd]]
    elif protocol == "cross-full":
        for x in all_datasets:
            for y in all_datasets:
                if x == y:
                    continue
                train_datasets.append([x])
                test_datasets.append([y])
    elif protocol[:3] == "loo":
        loo_map = { "crm": crm, "cimo": cimo, "all": all_datasets }
        ds_list = loo_map[protocol[4:]]
        for leave in ds_list:
            rest = [ds for ds in ds_list if ds != leave]
            train_datasets.append(rest)
            test_datasets.append([leave])
    elif protocol[:8] == "oulu-npu":
        protocol_number = int(protocol.split('-')[-1])
        if protocol_number < 3:
            train_datasets.append([f"oulu-npu-{protocol_number}-1"])
            test_datasets.append([f"oulu-npu-{protocol_number}-1"])
        else:
            for i in range(6):
                train_datasets.append([f"oulu-npu-{protocol_number}-{i+1}"])
                test_datasets.append([f"oulu-npu-{protocol_number}-{i+1}"])
    return train_datasets, test_datasets

if __name__=="__main__":
    def print_results(train_img_dirs, train_annotation_files, test_img_dirs,
            test_annotation_files):
        print("train:")
        for i in range(len(train_img_dirs)):
            print(f"{train_img_dirs[i]}: {train_annotation_files[i]}")
        print("test:")
        for i in range(len(test_img_dirs)):
            print(f"{test_img_dirs[i]}: {test_annotation_files[i]}")

    protocols = ["intra-casia-fasd", "intra-msu-mfsd", "intra-replay-attack",
         "intra-oulu-npu", "intra-crm", "cross-fast", "cross-full",
         "loo-crm", "loo-cimo"]
    protocols = ["oulu-npu-1", "oulu-npu-2", "oulu-npu-3", "oulu-npu-4"]
    for p in protocols:
        print("---------------------------------------------")
        print(p)
        print("---------------------------------------------")
        train_list, test_list = get_protocol_parameters(p)
        for train_datasets, test_datasets in zip(train_list, test_list):
            print(f"========> {train_datasets} -> {test_datasets}")
            a, b, c, d = get_liveness_datasets(train_datasets, test_datasets,
                                               1, "/home/raul/liveness/newds",
                                               include_devel=True, debug=True)
            print_results(a, b, c, d)
