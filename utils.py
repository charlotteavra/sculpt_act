import torch
import numpy as np
import math
import os
import open3d as o3d
import h5py
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from constants import HARDWARE_CONFIGS


import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            qvel = root["/observations/qvel"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][
                    max(0, start_ts - 1) :
                ]  # hack, to make timesteps more aligned
                action_len = episode_len - max(
                    0, start_ts - 1
                )  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad


class ClayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_idxs,
        dataset_dir,
        n_datapoints,
        n_raw_trajectories,
        center_action=False,
        stopping_token=False,
    ):
        """
        The Dataloader for the clay sculpting dataset at the Trajectory level (compatible with ACT and Diffusion Policy).

        :param episode_idxs: list of indices of the episodes to load
        :param dataset_dir: directory where the dataset is stored
        :param n_datapoints: number of datapoints (i.e. desired number of final trajectories after augmentation)
        :param n_raw_trajectories: number of raw trajectories in the dataset
        :param center_action: whether to center the action before normalizing
        :param stopping_token: whether to add a stopping token to the action [not currently implemented TODO]
        """
        super(ClayDataset).__init__()
        self.dataset_dir = dataset_dir
        self.episode_idxs = episode_idxs
        self.max_len = 9  # maximum number of actions for X trajectory
        self.action_shape = (self.max_len, 5)
        self.n_datapoints = n_datapoints
        self.n_raw_trajectories = n_raw_trajectories
        self.center_action = center_action
        self.stopping_token = stopping_token
        self.hardware_config = HARDWARE_CONFIGS

        # determine the number of datapoints per trajectory - needs to be a round number
        self.n_datapoints_per_trajectory = int(
            self.n_datapoints / self.n_raw_trajectories
        )

        # deterime the augmentation interval
        self.aug_step = 360 / self.n_datapoints_per_trajectory

    def _center_pcl(self, pcl, center):
        centered_pcl = pcl - center
        centered_pcl = centered_pcl * 10
        return centered_pcl

    def _center_normalize_action(self, action, ctr):
        # center the action
        new_action = np.zeros(5)
        new_action[0:3] = action[0:3] - ctr
        new_action[3:5] = action[3:5]
        # normalize centered action
        mins = np.array([-0.15, -0.15, -0.05, -90, 0.005])
        maxs = np.array([0.15, 0.15, 0.05, 90, 0.05])
        norm_action = np.zeros(6)
        norm_action[0:5] = (action[0:5] - mins) / (maxs - mins)
        return norm_action

    def _normalize_action(self, action):
        a_mins5d = self.hardware_config["a_mins5d"]
        a_maxs5d = self.hardware_config["a_maxs5d"]
        norm_action = (action - a_mins5d) / (a_maxs5d - a_mins5d)
        return norm_action

    def _rotate_pcl(self, state, center, rot):
        """
        Rotation augmentation for pointcloud

        :param[in] state(o3d.geometry.PointCloud): Input pointcloud state to be augmented
        :param[in] ctr(numpy.ndarray): (3,) Center location of input pointcloud [x,y,z]
        :param[in] rot(float): Desired rotation of input pointcloud (in degrees)
        :param[out] pcl_aug(numpy.ndarray): (n,3) Augmented pointcloud in array form
        """
        state = state.points - center
        R = Rotation.from_euler("xyz", np.array([0, 0, rot]), degrees=True).as_matrix()
        state = R @ state.T
        pcl_aug = state.T + center
        return pcl_aug

    def _rotate_action(self, action, center, rot):
        """
        Rotation augmentation for action

        :param[in] action(numpy.ndarray): (5,) Action to be augmented [x,y,z,z_rotation,gripper_distance]
        :param[in] center(numpy.ndarray): (3,) Center location of state pointcloud [x,y,z]
        :param[in] rot(float): Desired rotation of input pointcloud (in degrees)
        :param[out] action(numpy.ndarray): (5,) Rotation augmented action [x,y,z,z_rotation,gripper_distance]
        """
        unit_circle_og_grasp = (action[0] - center[0], action[1] - center[1])
        rot_original = math.degrees(
            math.atan2(unit_circle_og_grasp[1], unit_circle_og_grasp[0])
        )
        unit_circle_radius = math.sqrt(
            unit_circle_og_grasp[0] ** 2 + unit_circle_og_grasp[1] ** 2
        )
        rot_new = rot_original + rot

        new_unit_circle_grasp = (
            unit_circle_radius * math.cos(math.radians(rot_new)),
            unit_circle_radius * math.sin(math.radians(rot_new)),
        )

        new_global_grasp = (
            center[0] + new_unit_circle_grasp[0],
            center[1] + new_unit_circle_grasp[1],
        )
        x = new_global_grasp[0]
        y = new_global_grasp[1]
        rz = action[3] + rot
        rz = wrap_rz(rz)
        action_aug = np.array([x, y, action[2], rz, action[4]])

        return action_aug

    def __len__(self):
        """
        Return the number of episodes in the dataset (i.e. the number of actions in the trajectory folder)
        """
        return len(self.episode_idxs)

    def __getitem__(self, index):
        # built in ACT functionality to determine idx randomness
        idx = self.episode_idxs[index]

        # determine which raw_trajectory to index
        raw_traj_idx = idx // self.n_datapoints_per_trajectory

        # determine the rotation augmentation to apply
        aug_rot = (idx % self.n_datapoints_per_trajectory) * self.aug_step

        traj_path = self.dataset_dir + "/Trajectory" + str(raw_traj_idx)

        states = []
        actions = []
        j = 0
        # iterate loading in the actions as long as the next state point cloud exists
        while os.path.exists(traj_path + "/state" + str(j) + ".npy"):
            ctr = np.load(traj_path + "/pcl_center" + str(j) + ".npy")

            state_path = traj_path + "/Raw_State" + str(j) + "/"
            pc2 = o3d.io.read_point_cloud(state_path + "pc_cam2.ply")
            pc3 = o3d.io.read_point_cloud(state_path + "pc_cam3.ply")
            pc4 = o3d.io.read_point_cloud(state_path + "pc_cam4.ply")
            pc5 = o3d.io.read_point_cloud(state_path + "pc_cam5.ply")
            s = stitch_state_pcls(pc2, pc3, pc4, pc5)
            states.append(s)  # append to state list

            if j != 0:
                # apply action augmentation
                a = np.load(traj_path + "/action" + str(j - 1) + ".npy")
                a_rot = self._rotate_action(a, ctr, aug_rot)  # apply action rotation
                a_scaled = self._normalize_action(a_rot)
                actions.append(a_scaled)
            j += 1

        episode_len = len(actions)
        start_ts = np.random.choice(episode_len)  # random starting point in trajectory
        state = states[start_ts]

        # apply state augmentation
        s_rot = self._rotate_pcl(state, ctr, aug_rot)  # apply state rotation
        s_rot_scaled = self._center_pcl(s_rot, ctr)  # center and scale state

        img_arr = convert_state_to_image(s_rot_scaled, np.asarray(state.colors))
        all_cam_images = np.stack([img_arr], axis=0)

        # # load uncentered goal
        # g = np.load(traj_path + "/goal.npy")
        # # apply goal rotation
        # g_rot = self._rotate_pcl(g, ctr, aug_rot)
        # # center and scale goal
        # goal = self._center_pcl(g_rot, ctr)

        action = actions[start_ts:]
        action = np.stack(action, axis=0)

        if start_ts > 0:
            prev_action = actions[start_ts - 1]
        else:
            prev_action = self.hardware_config["initial_action"]

        action_len = episode_len - start_ts

        padded_action = np.zeros(self.action_shape, dtype=np.float32)

        padded_action[:action_len] = action
        is_pad = np.zeros(self.max_len)
        is_pad[action_len:] = 1

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        # goal_data = torch.from_numpy(goal).float()
        prev_action_data = torch.from_numpy(prev_action).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        # action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
        #     "action_std"
        # ]

        return image_data, prev_action_data, action_data, is_pad


def stitch_state_pcls(pc2, pc3, pc4, pc5):
    """
    Single stitched pointcloud of the state

    :param[in] state_path(str): Absolute path to raw point cloud states
    :param[out] pointcloud(o3d.geometry.PointCloud): Stitched pointcloud
    """

    # combine the point clouds
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = pc5.points
    pointcloud.colors = pc5.colors
    pointcloud.points.extend(pc2.points)
    pointcloud.colors.extend(pc2.colors)
    pointcloud.points.extend(pc3.points)
    pointcloud.colors.extend(pc3.colors)
    pointcloud.points.extend(pc4.points)
    pointcloud.colors.extend(pc4.colors)

    # remove grippers
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    hardware_config = HARDWARE_CONFIGS
    table_z = hardware_config["table_center_top"][2]
    gripper_z = hardware_config["gripper_z"]
    ind_z = np.where((points[:, 2] > table_z) & (points[:, 2] < gripper_z))
    pointcloud.points = o3d.utility.Vector3dVector(points[ind_z])
    pointcloud.colors = o3d.utility.Vector3dVector(colors[ind_z])

    # remove background
    radius = 0.12
    center = hardware_config["table_center_top"]
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    distances = np.linalg.norm(points - center, axis=1)
    indices = np.where(distances <= radius)

    pointcloud.points = o3d.utility.Vector3dVector(points[indices])
    pointcloud.colors = o3d.utility.Vector3dVector(colors[indices])

    return pointcloud


def convert_state_to_image(points, colors):
    """
    Converts input state into a top view image

    :param[in] points(numpy.ndarray): (n,3) shape array of pointcloud points
    :param[in] colors(numpy.ndarray): (n,3) shape array of pointcloud colors
    :param[out] (numpy.ndarray): top view RGB image of state
    """

    colors = colors.reshape(-1, 3)
    res = 256
    pts_norm = (
        (points - np.min(points)) * (res / (np.max(points) - np.min(points)))
    ).astype(int)
    img_arr = np.ones((res, res, 3), dtype=np.float32)

    x_coords, y_coords = pts_norm[:, 0], pts_norm[:, 1]

    valid_indices = np.where(
        (0 <= x_coords) & (x_coords < res) & (0 <= y_coords) & (y_coords < res)
    )

    img_arr[y_coords[valid_indices], x_coords[valid_indices]] = np.hstack(
        [
            colors[valid_indices, 2].reshape(len(valid_indices[0]), 1),
            colors[valid_indices, 1].reshape(len(valid_indices[0]), 1),
            colors[valid_indices, 0].reshape(len(valid_indices[0]), 1),
        ]
    )  # this line also converts RGB <-> BGR

    return img_arr


def wrap_rz(original_rz):
    """
    We want rz to be between -90 to 90, so wrap around if outside these bounds due to symmetrical gripper.
    """
    wrapped_rz = (original_rz + 90) % 180 - 90
    return wrapped_rz


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            qvel = root["/observations/qvel"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    return stats


def load_data(
    dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats
    )
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def load_clay_data(dataset_dir, num_episodes, batch_size_train, batch_size_val):
    print(f"\nData from: {dataset_dir}\n")

    train_dir = dataset_dir + "/Train"
    val_dir = dataset_dir + "/Test"

    # number of trajectories in each directory
    num_train_epidodes = len(next(os.walk(train_dir))[1])
    num_val_epidodes = len(next(os.walk(val_dir))[1])
    train_ratio = num_train_epidodes / (num_train_epidodes + num_val_epidodes)

    # obtain train test split
    # shuffled_indices = np.random.permutation(num_episodes)
    # train_ratio = 0.8
    # train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    # val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # define number of ideal final trajectories through augmentation
    num_total_aug_trajectories = 800
    train_total_aug_trajectories = int(train_ratio * num_total_aug_trajectories)
    val_total_aug_trajectories = int((1 - train_ratio) * num_total_aug_trajectories)

    train_indices = np.arange(0, num_train_epidodes, step=1)
    val_indices = np.arange(0, num_val_epidodes, step=1)

    # construct dataset and dataloader
    train_dataset = ClayDataset(
        train_indices,
        train_dir,
        train_total_aug_trajectories,
        num_episodes,
    )
    val_dataset = ClayDataset(
        val_indices, val_dir, val_total_aug_trajectories, num_episodes
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader


### pcl utils


def chamfer(x, y, pkg="numpy"):
    """ """
    if pkg == "numpy":
        # numpy implementation
        x = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        dis = np.linalg.norm(x - y, 2, axis=2)
        dis_xy = np.mean(np.min(dis, axis=1))  # dis_xy: mean over N
        dis_yx = np.mean(np.min(dis, axis=0))  # dis_yx: mean over M
    else:
        # torch implementation
        x = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

    return dis_xy + dis_yx


def emd(x, y, pkg="numpy"):
    if pkg == "numpy":
        # numpy implementation
        x_ = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y_ = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        cost_matrix = np.linalg.norm(x_ - y_, 2, axis=2)
        try:
            ind1, ind2 = linear_sum_assignment(cost_matrix, maximize=False)
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")
        emd = np.mean(np.linalg.norm(x[ind1] - y[ind2], 2, axis=1))
    else:
        # torch implementation
        x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
        cost_matrix = dis.detach().cpu().numpy()
        try:
            ind1, ind2 = linear_sum_assignment(cost_matrix, maximize=False)
        except:
            # pdb.set_trace()
            print("Error in linear sum assignment!")

        emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))

    return emd


def hausdorff(x, y, pkg="numpy"):
    if pkg == "numpy":
        x = np.repeat(np.expand_dims(x, axis=1), y.shape[0], axis=1)  # x: [N, M, D]
        y = np.repeat(np.expand_dims(y, axis=0), x.shape[0], axis=0)  # y: [N, M, D]
        dis = np.linalg.norm(x - y, 2, axis=2)
        dis_xy = np.max(np.min(dis, axis=1))  # dis_xy: mean over N
        dis_yx = np.max(np.min(dis, axis=0))  # dis_yx: mean over M
    else:
        # torch implementation
        x = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
        y = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.max(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.max(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

    return dis_xy + dis_yx


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
