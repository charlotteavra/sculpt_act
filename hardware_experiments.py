import os
import cv2
import time
import json
import torch
import queue
import argparse
import threading
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import robomail.vision as vis
from frankapy import FrankaArm
from constants import DATA_DIR
from constants import TASK_CONFIGS
from constants import HARDWARE_CONFIGS
from imitate_episodes import make_policy
from utils import chamfer, emd, hausdorff, stitch_state_pcls, convert_state_to_image
from robot_utils import get_real_action_from_normalized, goto_grasp

"""
This is the generic script for the clay hardware experiments. It will save all the necessary information to
document each experiment. This includes the following:
    - RGB image from each camera
    - goal point cloud
    - state point clouds
    - number of actions to completion
    - real-world time to completion
    - chamfer distance between final state and goal
    - earth mover's distance between final state and goal
    - video from camera 6 recording the entire experimental run
"""


def main(args):
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    task_name = args["task_name"]
    temporal_agg = args["temporal_agg"]

    # task parameters
    task_config = TASK_CONFIGS[task_name]
    episode_len = task_config["episode_len"]

    # hardware parameters
    hardware_config = HARDWARE_CONFIGS

    # fixed parameters
    state_dim = 5
    lr_backbone = 1e-5
    backbone = "resnet18"

    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "state_dim": state_dim,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
        }
    else:
        raise NotImplementedError

    experiment_config = {
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "temporal_agg": temporal_agg,
        "hardware_config": hardware_config,
    }

    exp_num = 1
    exp_save = "experiments/exp" + str(exp_num)

    # check to make sure the experiment number is not already in use, if it is, increment the number to ensure no save overwrites
    while os.path.exists(exp_save):
        exp_num += 1
        exp_save = "experiments/exp" + str(exp_num)
    os.mkdir(exp_save)

    # initialize the robot and reset joints
    robot = FrankaArm()
    robot.reset_joints()

    # initialize the cameras
    cameras = {
        "2": vis.CameraClass(2),
        "3": vis.CameraClass(3),
        "4": vis.CameraClass(4),
        "5": vis.CameraClass(5),
    }

    # initialize camera 6 pipeline
    W = 1280
    H = 800
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("152522250441")
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(config)

    # load in the goal and save to the experiment folder
    goal_shape = "X"  # 'cone' or 'line' or 'X' or 'Y' or 'cylinder
    goal_shape_path = goal_shape + ".npy"
    goal = np.load(os.path.join(DATA_DIR, "clay_sculpting/goals/", goal_shape_path))
    np.save(exp_save + "/goal.npy", goal)

    # initialize the threads
    done_queue = queue.Queue()

    main_thread = threading.Thread(
        target=experiment_loop,
        args=(
            robot,
            cameras,
            experiment_config,
            exp_save,
            goal,
            done_queue,
        ),
    )
    video_thread = threading.Thread(
        target=video_loop, args=(pipeline, exp_save, done_queue)
    )

    main_thread.start()
    video_thread.start()


def experiment_loop(robot, cameras, experiment_config, save_path, goal, done_queue):
    """ """
    hardware_config = experiment_config["hardware_config"]
    observation_pose = hardware_config["observation_pose"]
    a_mins5d = hardware_config["a_mins5d"]
    a_maxs5d = hardware_config["a_maxs5d"]
    policy_config = experiment_config["policy_config"]
    state_dim = policy_config["state_dim"]

    # go to observation pose
    pose = robot.get_pose()
    pose.translation = observation_pose
    robot.goto_pose(pose)

    # initialize the n_actions counter
    n_action = 0

    # get the starting time
    start_time = time.time()

    # save initial observation state
    rgb2, _, pc2, _ = cameras["2"]._get_next_frame()
    rgb3, _, pc3, _ = cameras["3"]._get_next_frame()
    rgb4, _, pc4, _ = cameras["4"]._get_next_frame()
    rgb5, _, pc5, _ = cameras["5"]._get_next_frame()

    pc2.transform(cameras["2"].get_cam_extrinsics())  # transform to robot frame
    pc3.transform(cameras["3"].get_cam_extrinsics())
    pc4.transform(cameras["4"].get_cam_extrinsics())
    pc5.transform(cameras["5"].get_cam_extrinsics())

    o3d.io.write_point_cloud(save_path + "/cam2_pcl0.ply", pc2)
    o3d.io.write_point_cloud(save_path + "/cam3_pcl0.ply", pc3)
    o3d.io.write_point_cloud(save_path + "/cam4_pcl0.ply", pc4)
    o3d.io.write_point_cloud(save_path + "/cam5_pcl0.ply", pc5)

    cv2.imwrite(save_path + "/rgb2_state0.jpg", rgb2)
    cv2.imwrite(save_path + "/rgb3_state0.jpg", rgb3)
    cv2.imwrite(save_path + "/rgb4_state0.jpg", rgb4)
    cv2.imwrite(save_path + "/rgb5_state0.jpg", rgb5)

    pointcloud = stitch_state_pcls(pc2, pc3, pc4, pc5)
    img_arr = convert_state_to_image(
        np.asarray(pointcloud.points), np.asarray(pointcloud.colors)
    )
    np.save(save_path + "/pcl0.npy", pointcloud)
    np.save(save_path + "/img0.npy", img_arr)

    cd = chamfer(pointcloud, goal)
    earthmovers = emd(pointcloud, goal)
    hausdorff_dist = hausdorff(pointcloud, goal)
    print("\nChamfer Distance: ", cd)
    print("Earth Mover's Distance: ", earthmovers)
    print("Hausdorff Distance: ", hausdorff_dist)

    # load policy and stats
    policy_class = experiment_config["policy_class"]
    policy_config = experiment_config["policy_config"]
    ckpt_dir = experiment_config["ckpt_dir"]
    temporal_agg = experiment_config["temoporal_agg"]

    ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)

    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + num_queries, state_dim]
        ).cuda()

    max_timesteps = experiment_config[
        "episode_len"
    ]  # may increase for real-world tasks
    max_timesteps = int(max_timesteps * 1)

    with torch.inference_mode():
        for t in range(max_timesteps):
            # process image
            image_data = torch.from_numpy(np.stack([img_arr], axis=0))
            image_data = torch.einsum("k h w c -> k c h w", image_data)
            curr_image = image_data / 255.0

            if t == 0:
                robot_action = hardware_config["initial_action"]

            # query policy
            if experiment_config["policy_class"] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(robot_action, curr_image)
                if temporal_agg:
                    all_time_actions[[t], t : t + num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(
                        dim=0, keepdim=True
                    )
                else:
                    raw_action = all_actions[:, t % query_frequency]
            elif policy_config["policy_class"] == "CNNMLP":
                raw_action = policy(robot_action, curr_image)
            else:
                raise NotImplementedError

            # convert action into robot space and go to pose
            raw_action = raw_action.squeeze(0).cpu().numpy()
            robot_action = get_real_action_from_normalized(raw_action)
            goto_grasp(
                robot,
                robot_action[0],
                robot_action[1],
                robot_action[2],
                0,
                0,
                robot_action[3],
                robot_action[4],
            )
            print("\nGRASP ACTION: ", robot_action)

            # pred_action_sequence = None  # TODO: fill this in with respective action sequence prediction model
            # pred_action = pred_action_sequence[
            #     0
            # ]  # TODO: include a parameter to execute N steps before replanning
            # unnorm_a = (
            #     pred_action + 1
            # ) / 2.0  # NOTE: this step is for the model to output actions in the range [-1, 1], if the model outputs actions in the range [0, 1], this step is not necessary
            # unnorm_a = unnorm_a * (a_maxs5d - a_mins5d) + a_mins5d

            # execute the unnormalized action
            # goto_grasp(
            #     robot,
            #     unnorm_a[0],
            #     unnorm_a[1],
            #     unnorm_a[2],
            #     0,
            #     0,
            #     unnorm_a[3],
            #     unnorm_a[4],
            # )
            n_action += 1

            # wait here
            time.sleep(3)

            # open the gripper
            robot.open_gripper(block=True)
            # time.sleep(2)

            # move to observation pose
            pose.translation = observation_pose
            robot.goto_pose(pose)

            # save observation state
            rgb2, _, pc2, _ = cameras["2"]._get_next_frame()
            rgb3, _, pc3, _ = cameras["3"]._get_next_frame()
            rgb4, _, pc4, _ = cameras["4"]._get_next_frame()
            rgb5, _, pc5, _ = cameras["5"]._get_next_frame()

            pc2.transform(cameras["2"].get_cam_extrinsics())  # transform to robot frame
            pc3.transform(cameras["3"].get_cam_extrinsics())
            pc4.transform(cameras["4"].get_cam_extrinsics())
            pc5.transform(cameras["5"].get_cam_extrinsics())

            o3d.io.write_point_cloud(save_path + "/cam2_pcl" + str(t + 1) + ".ply", pc2)
            o3d.io.write_point_cloud(save_path + "/cam3_pcl" + str(t + 1) + ".ply", pc3)
            o3d.io.write_point_cloud(save_path + "/cam4_pcl" + str(t + 1) + ".ply", pc4)
            o3d.io.write_point_cloud(save_path + "/cam5_pcl" + str(t + 1) + ".ply", pc5)

            # save observation
            np.save(save_path + "/pcl" + str(t + 1) + ".npy", pointcloud)
            cv2.imwrite(save_path + "/rgb2_state" + str(t + 1) + ".jpg", rgb2)
            cv2.imwrite(save_path + "/rgb3_state" + str(t + 1) + ".jpg", rgb3)
            cv2.imwrite(save_path + "/rgb4_state" + str(t + 1) + ".jpg", rgb4)
            cv2.imwrite(save_path + "/rgb5_state" + str(t + 1) + ".jpg", rgb5)

            pointcloud = stitch_state_pcls(pc2, pc3, pc4, pc5)
            img_arr = convert_state_to_image(
                np.asarray(pointcloud.points), np.asarray(pointcloud.colors)
            )
            np.save(save_path + "/pcl0.npy", pointcloud)
            np.save(save_path + "/pcl" + str(t + 1) + ".npy", pointcloud)
            np.save(save_path + "/img" + str(t + 1) + ".npy", img_arr)

            # get the distance metrics between the point cloud and goal
            cd = chamfer(pointcloud, goal)
            earthmovers = emd(pointcloud, goal)
            hausdorff_dist = hausdorff(pointcloud, goal)
            print("\nChamfer Distance: ", cd)
            print("Earth Mover's Distance: ", earthmovers)
            print("Hausdorff Distance: ", hausdorff_dist)

            # exit loop early if the goal is reached
            if earthmovers < 0.01 or cd < 0.01:
                break

            # alternate break scenario --> if the past 3 actions have not resulted in a decent change in the emd or cd, break

    # completed the experiment, send the message to the video recording loop
    done_queue.put("Done!")

    # get the ending time
    end_time = time.time()

    # create and save a dictionary of the experiment results
    results_dict = {
        "n_actions": n_action,
        "time_to_completion": end_time - start_time,
        "chamfer_distance": cd,
        "earth_movers_distance": emd,
    }
    with open(save_path + "/results.txt", "w") as f:
        f.write(str(results_dict))


# VIDEO THREAD
def video_loop(cam_pipeline, save_path, done_queue):
    """ """
    forcc = cv2.VideoWriter_fourcc(*"XVID")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    out = cv2.VideoWriter(save_path + "/video.avi", forcc, 30.0, (848, 480))

    frame_save_counter = 0
    # record until main loop is complete
    while done_queue.empty():
        frames = cam_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # crop and rotate the image to just show elevated stage area
        cropped_image = color_image[320:520, 430:690, :]
        rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # save frame approx. every 10 seconds
        if frame_save_counter % 300 == 0:
            cv2.imwrite(
                save_path + "/external_rgb" + str(frame_save_counter) + ".jpg",
                rotated_image,
            )
        frame_save_counter += 1
        out.write(rotated_image)

    cam_pipeline.stop()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    main(vars(parser.parse_args()))
