# [ADD ISSUE: Replace the Dataset with WebDataset]

import os
import time
import random
import argparse
import itertools
from tqdm import tqdm

from omegaconf import OmegaConf

import torch
import torchaudio
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from soundstream.distributed.launch import launch
from soundstream.models.soundstream import SoundStream
from soundstream.dataset.dataset import SoundDataset
from soundstream.modules.loss import criterion_d
from soundstream.modules.loss import criterion_g
from soundstream.modules.loss import loss_dis
from soundstream.modules.loss import loss_g
from soundstream.modules.discriminators.frequency_discriminator import (
    MultiFrequencyDiscriminator,
)

from soundstream.utils.utils import Logger
from soundstream.utils.utils import seed_everything

NODE_RANK = os.environ["INDEX"] if "INDEX" in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = (
    (os.environ["CHIEF_IP"], 22275)
    if "CHIEF_IP" in os.environ
    else ("127.0.0.1", 29500)
)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = "tcp://%s:%s" % (MASTER_ADDR, MASTER_PORT)
NUM_NODE = os.environ["HOST_NUM"] if "HOST_NUM" in os.environ else 1


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print("Total Size of the model is：{:.3f}MB".format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_node",
        type=int,
        default=NUM_NODE,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--ngpus_per_node", type=int, default=8, help="number of gpu on one node"
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=NODE_RANK,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--dist_url",
        type=str,
        default=DIST_URL,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id to use. If given, only the specific gpu will be"
        " used, and ddp will be disabled",
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    # args for random
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for initializing training. "
    )
    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="use tensorboard for logging"
    )
    # args for training
    parser.add_argument(
        "--LAMBDA_ADV", type=float, default=1, help="hyper-parameter for adver loss"
    )
    parser.add_argument(
        "--LAMBDA_FEAT", type=float, default=1, help="hyper-parameter for feat loss"
    )
    parser.add_argument(
        "--LAMBDA_REC", type=float, default=1, help="hyper-parameter for rec loss"
    )
    parser.add_argument(
        "--LAMBDA_COM", type=float, default=1000, help="hyper-parameter for commit loss"
    )
    parser.add_argument(
        "--N_EPOCHS", type=int, default=100, help="Total training epoch"
    )
    parser.add_argument("--st_epoch", type=int, default=0, help="start training epoch")
    parser.add_argument(
        "--global_step", type=int, default=0, help="record the global step"
    )
    parser.add_argument("--discriminator_iter_start", type=int, default=500)
    parser.add_argument("--BATCH_SIZE", type=int, default=2, help="batch size")
    parser.add_argument(
        "--PATH", type=str, default="model_path/", help="The path to save the model"
    )
    parser.add_argument("--sr", type=int, default=16000, help="sample rate")
    parser.add_argument("--print_freq", type=int, default=10, help="the print number")
    parser.add_argument("--save_dir", type=str, default="log", help="log save path")
    parser.add_argument(
        "--audio_type",
        type=str,
        default="vocals",
        help="possible values [vocals, instrumentals]",
    )
    parser.add_argument(
        "--train_csv", type=str, default="path_to_csv", help="train data"
    )
    parser.add_argument(
        "--valid_csv", type=str, default="path_to_val_csv", help="valid data"
    )
    parser.add_argument(
        "--train_data_path", type=str, default="path_to_wavs", help="train data"
    )
    parser.add_argument(
        "--valid_data_path", type=str, default="path_to_val_wavs", help="valid data"
    )
    parser.add_argument("--resume", action="store_true", help="whether re-train model")
    parser.add_argument(
        "--resume_path", type=str, default="path_to_resume", help="resume_path"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default="name_of_resume_checkpoint",
        help="resume_ckpt",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="whether to freeze the encoder weights",
    )
    parser.add_argument(
        "--ratios",
        type=int,
        nargs="+",
        # probs(ratios) = hop_size
        default=[8, 5, 4, 2],
        help="ratios of SoundStream, shoud be set for different hop_size (32d, 320, 240d, ...)",
    )
    parser.add_argument(
        "--target_bandwidths",
        type=float,
        nargs="+",
        # default for 16k_320d
        default=[0.5, 1, 1.5, 2, 4],
        help="target_bandwidths of net3.py",
    )
    parser.add_argument(
        "--use_custom_lr",
        action="store_true",
        help="use a custom lr, not the one from save checkpoint",
    )
    parser.add_argument(
        "--disc_warmup_ratio",
        type=int,
        default=2,
        help="disc_warmup_ratio",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="path_to_architecture_config",
        help="config_path",
    )
    parser.add_argument(
        "--log_dir", type=str, default="path_to_log_dir", help="log_dir"
    )
    parser.add_argument(
        "--save_results_every", type=int, default=50, help="save_results_every"
    )
    parser.add_argument(
        "--checkpoint_every", type=int, default=5000, help="checkpoint_every"
    )
    parser.add_argument(
        "--save_results_dir",
        type=str,
        default="path_to_results_dir",
        help="save_results_dir",
    )

    args = parser.parse_args()
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    if args.resume:
        args.PATH = args.resume_path  # direcly use the old model path
    else:
        args.PATH = os.path.join(args.PATH, time_str)
    args.save_dir = os.path.join(args.save_dir, time_str)
    os.makedirs(args.PATH, exist_ok=True)
    return args


def get_input(x):
    x = x.to(memory_format=torch.contiguous_format)
    return x.float()


def main():
    args = get_args()
    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)
        print("Seed Set")
    if args.num_node == 1:
        args.dist_url == "auto"
    else:
        assert args.num_node > 1
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.num_node  #
    launch(
        main_worker,
        args.ngpus_per_node,
        args.num_node,
        args.node_rank,
        args.dist_url,
        args=(args,),
    )


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    if not args.distributed:
        print("Distributed Training Off")

    # CUDA_VISIBLE_DEVICES = int(args.local_rank)
    logger = Logger(args)
    # 320x downsampling
    arch_config = OmegaConf.load(args.config_path)

    soundstream = SoundStream(
        n_filters=arch_config["generator"]["config"]["n_filters"],
        D=arch_config["generator"]["config"]["D"],
        bins=arch_config["generator"]["config"]["bins"],
        sample_rate=arch_config["generator"]["config"]["sample_rate"],
        target_bandwidths=arch_config["generator"]["config"]["target_bandwidths"],
        ratios=arch_config["generator"]["config"]["ratios"],
    )
    mfd = MultiFrequencyDiscriminator(config=arch_config["mfd"]["config"])

    getModelSize(soundstream)
    getModelSize(mfd)

    if args.distributed:
        soundstream = torch.nn.SyncBatchNorm.convert_sync_batchnorm(soundstream)
        mfd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mfd)
    # torch.distributed.barrier()
    args.device = torch.device("cuda", args.local_rank)
    soundstream.to(args.device)
    mfd.to(args.device)
    if args.distributed:
        soundstream = DDP(
            soundstream, device_ids=[args.local_rank], find_unused_parameters=True
        )  # device_ids=[args.local_rank], output_device=args.local_rank
        mfd = DDP(mfd, device_ids=[args.local_rank], find_unused_parameters=True)

    train_dataset = SoundDataset(
        audio_type=args.audio_type,
        audio_data=args.train_csv,
        audio_dir=args.train_data_path,
    )
    valid_dataset = SoundDataset(
        audio_type=args.audio_type,
        audio_data=args.valid_csv,
        audio_dir=args.valid_data_path,
    )
    # args.sr = train_dataset.sr

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=True, shuffle=True
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    else:
        train_sampler = None
        valid_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        sampler=train_sampler,
        pin_memory=True,
        prefetch_factor=4,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        sampler=valid_sampler,
        pin_memory=True,
        prefetch_factor=4,
    )
    print("Shuffle Activated")

    optimizer_g = torch.optim.AdamW(
        soundstream.parameters(),
        lr=3e-4,
        eps=1e-6,
        betas=(0.8, 0.99),
        weight_decay=0.01,
    )
    lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999)
    optimizer_d = torch.optim.AdamW(
        mfd.parameters(), lr=3e-4, eps=1e-6, betas=(0.8, 0.99), weight_decay=0.01
    )
    lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999)
    if args.resume:
        latest_info = torch.load(os.path.join(args.resume_path, args.resume_ckpt))
        args.st_epoch = latest_info["epoch"]
        soundstream.load_state_dict(latest_info["codec_model"], strict=False)
        mfd.load_state_dict(latest_info["mfd"])
        # if custom_lr then use the custom set lr for both generaor and discriminator
        if args.use_custom_lr == True:
            print(f"using custom lr: {lr_scheduler_g.get_lr()[0]}")
        # if not then load from state_dict
        else:
            optimizer_g.load_state_dict(latest_info["optimizer_g"])
            lr_scheduler_g.load_state_dict(latest_info["lr_scheduler_g"])
            optimizer_d.load_state_dict(latest_info["optimizer_d"])
            lr_scheduler_d.load_state_dict(latest_info["lr_scheduler_d"])
            print(f"using saved model lr: {lr_scheduler_g.get_lr()[0]}")
    train(
        args,
        soundstream,
        mfd,
        train_loader,
        valid_loader,
        optimizer_g,
        optimizer_d,
        lr_scheduler_g,
        lr_scheduler_d,
        logger,
    )


def train(
    args,
    soundstream,
    mfd,
    train_loader,
    valid_loader,
    optimizer_g,
    optimizer_d,
    lr_scheduler_g,
    lr_scheduler_d,
    logger,
):
    print("value of global_rank:", args.global_rank)
    best_val_loss = float("inf")
    best_val_epoch = -1
    global_step = 0
    if args.resume:
        # get the global_step from resume_ckpt
        global_step = int(args.resume_ckpt.split(".")[0].split("_")[-1]) + 1

    if args.freeze_encoder:
        soundstream.freeze_encoder()

    for epoch in range(args.st_epoch, args.N_EPOCHS + 1):
        soundstream.train()
        mfd.train()
        train_loss_d = 0.0
        train_w_loss_d = 0.0
        train_adv_g_loss = 0.0
        train_feat_loss = 0.0
        train_rec_loss = 0.0
        train_loss_g = 0.0
        train_commit_loss = 0.0
        k_iter = 0

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        for x in tqdm(train_loader):
            x = x.to(args.device)
            for optimizer_idx in [0, 1]:  # we have two optimizer
                x_wav = get_input(x)
                G_x, commit_loss, last_layer = soundstream(x_wav)

                if optimizer_idx == 0:
                    # update generator
                    # MFD
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mfd(
                        x_wav.contiguous(), G_x.contiguous()
                    )

                    total_loss_g, rec_loss, adv_g_loss, feat_loss, d_weight = loss_g(
                        commit_loss,
                        x_wav,
                        G_x,
                        fmap_f_r,
                        fmap_f_g,
                        y_df_hat_r,
                        y_df_hat_g,
                        global_step,
                        last_layer=last_layer,
                        is_training=True,
                        args=args,
                    )
                    train_commit_loss += commit_loss
                    train_loss_g += total_loss_g.item()
                    train_adv_g_loss += adv_g_loss.item()
                    train_feat_loss += feat_loss.item()
                    train_rec_loss += rec_loss.item()

                    # update the generator once and discriminator twice
                    if global_step%args.disc_warmup_ratio==0:
                        optimizer_g.zero_grad()  # Clear gradients for the generator optimizer
                        total_loss_g.backward()  # Backpropagate the loss
                        optimizer_g.step()  # Update generator parameters
                        lr_scheduler_g.step()  # Step the learning rate scheduler
                else:
                    # update discriminator
                    # MFD
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mfd(
                        x.detach(), G_x.detach()
                    )

                    w_loss_d, loss_d = loss_dis(
                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, global_step, args
                    )
                    train_w_loss_d += w_loss_d.item()
                    train_loss_d += loss_d.item()

                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()
                    lr_scheduler_d.step()

            # log at step level
            # at each step we get the average loss of the batch (check line 288)
            message = "<epoch:{:d}, iter:{:d}, step:{:d}, lr_d: {:.6f}, lr_g: {:.6f}, total_loss_g:{:.4f}, adv_g_loss:{:.4f}, feat_loss:{:.4f}, rec_loss:{:.4f}, commit_loss:{:.4f}, w_loss_d:{:.4f}, loss_d:{:.4f}, d_weight: {:.4f}>".format(
                epoch,
                k_iter,
                global_step,
                lr_scheduler_d.get_last_lr()[0],
                lr_scheduler_g.get_last_lr()[0],
                total_loss_g.item(),
                adv_g_loss.item(),
                feat_loss.item(),
                rec_loss.item(),
                commit_loss.item(),
                w_loss_d.item(),
                loss_d.item(),
                d_weight.item(),
            )

            # log in tensorboard at every step
            logger.add_scalar(
                **{
                    "tag": "epoch/steps",
                    "scalar_value": epoch,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "lr_d/steps",
                    "scalar_value": lr_scheduler_d.get_last_lr()[0],
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "lr_g/steps",
                    "scalar_value": lr_scheduler_g.get_last_lr()[0],
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "total_loss_g/step",
                    "scalar_value": total_loss_g,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "adv_g_loss/step",
                    "scalar_value": adv_g_loss,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "feat_loss/step",
                    "scalar_value": feat_loss,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "rec_loss/step",
                    "scalar_value": rec_loss,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "commit_loss/step",
                    "scalar_value": commit_loss,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "w_loss_d/step",
                    "scalar_value": w_loss_d,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "loss_d/step",
                    "scalar_value": loss_d,
                    "global_step": global_step,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "d_weight/step",
                    "scalar_value": d_weight,
                    "global_step": global_step,
                }
            )
            if args.global_rank == 0:
                if global_step % args.checkpoint_every == 0:
                    latest_model_soundstream = soundstream.state_dict().copy()
                    latest_mfd = mfd.state_dict().copy()

                    latest_save = {}
                    latest_save["codec_model"] = latest_model_soundstream
                    latest_save["mfd"] = latest_mfd
                    latest_save["epoch"] = epoch
                    latest_save["steps"] = global_step
                    latest_save["optimizer_g"] = optimizer_g.state_dict()
                    latest_save["optimizer_d"] = optimizer_d.state_dict()
                    latest_save["lr_scheduler_g"] = lr_scheduler_g.state_dict()
                    latest_save["lr_scheduler_d"] = lr_scheduler_d.state_dict()
                    torch.save(latest_save, args.PATH + f"/ckpt_{global_step}.pth")

            if global_step % args.save_results_every == 0:
                # get a sample from current train batch data
                filename = f"{args.save_results_dir}/sample_train_gt_{global_step}.wav"
                train_batch_wav = x.clone()
                train_batch_wav = train_batch_wav.unbind(dim=0)
                random_train_wav = random.choice(train_batch_wav)
                torchaudio.save(
                    filename, random_train_wav.detach().cpu(), sample_rate=16000
                )
                print("Devices on which inference audios are:", random_train_wav.device)

                # get a sample from a valid batch data
                filename = f"{args.save_results_dir}/sample_valid_gt_{global_step}.wav"
                rand = random.randint(1, 20)
                for _ in range(rand):
                    valid_batch_wav = next(iter(valid_loader))
                valid_batch_wav = next(iter(valid_loader)).to(args.device)
                valid_batch_wav = get_input(valid_batch_wav)
                valid_batch_wav = valid_batch_wav.unbind(dim=0)
                random_valid_wav = random.choice(valid_batch_wav)
                torchaudio.save(
                    filename, random_valid_wav.detach().cpu(), sample_rate=16000
                )

                print("Devices on which inference audios are:", random_valid_wav.device)
                # set soundstream to eval
                soundstream.eval()

                with torch.inference_mode():
                    filename = (
                        f"{args.save_results_dir}/sample_train_recon_{global_step}.wav"
                    )
                    random_train_wav = random_train_wav.unsqueeze(0)
                    random_train_wav = get_input(random_train_wav)
                    gen_train_wav, commit_loss, last_layer = soundstream(
                        random_train_wav
                    )
                    gen_train_wav = gen_train_wav.squeeze(0)
                    torchaudio.save(
                        filename, gen_train_wav.detach().cpu(), sample_rate=16000
                    )

                    filename = (
                        f"{args.save_results_dir}/sample_valid_recon_{global_step}.wav"
                    )
                    random_valid_wav = random_valid_wav.unsqueeze(0)
                    random_valid_wav = get_input(random_valid_wav)
                    gen_valid_wav, commit_loss, last_layer = soundstream(
                        random_valid_wav
                    )
                    gen_valid_wav = gen_valid_wav.squeeze(0)
                    torchaudio.save(
                        filename, gen_valid_wav.detach().cpu(), sample_rate=16000
                    )

                # set soundstream back to train
                soundstream.train()

            # for logging at step level
            if global_step % args.print_freq == 0:
                logger.log_info(message)

            # next batch/step in the epoch
            k_iter += 1
            # next batch/step overall
            global_step += 1  # record the global step

        # for logging at epoch level
        message = "<epoch:{:d}, step:{:d}, <total_loss_g_train:{:.4f}, recon_loss_train:{:.4f}, adversarial_loss_train:{:.4f}, feature_loss_train:{:.4f}, commit_loss_train:{:.4f}, w_loss_d_train:{:.4f}, loss_d_train:{:.4f}>".format(
            epoch,
            global_step,
            train_loss_g / len(train_loader),
            train_rec_loss / len(train_loader),
            train_adv_g_loss / len(train_loader),
            train_feat_loss / len(train_loader),
            train_commit_loss / len(train_loader),
            train_w_loss_d / len(train_loader),
            train_loss_d / len(train_loader),
        )

        # made changes here, average out the total losses over all batches before logging
        logger.add_scalar(
            **{
                "tag": "total_loss_g_train/epoch",
                "scalar_value": train_loss_g / len(train_loader),
                "global_step": epoch,
            }
        )
        logger.add_scalar(
            **{
                "tag": "recon_loss_train/epoch",
                "scalar_value": train_rec_loss / len(train_loader),
                "global_step": epoch,
            }
        )
        logger.add_scalar(
            **{
                "tag": "adversarial_loss_train/epoch",
                "scalar_value": train_adv_g_loss / len(train_loader),
                "global_step": epoch,
            }
        )
        logger.add_scalar(
            **{
                "tag": "feature_loss_train/epoch",
                "scalar_value": train_feat_loss / len(train_loader),
                "global_step": epoch,
            }
        )
        logger.add_scalar(
            **{
                "tag": "commit_loss_train/epoch",
                "scalar_value": train_commit_loss / len(train_loader),
                "global_step": epoch,
            }
        )
        logger.add_scalar(
            **{
                "tag": "w_loss_d_train/epoch",
                "scalar_value": train_w_loss_d / len(train_loader),
                "global_step": epoch,
            }
        )
        logger.add_scalar(
            **{
                "tag": "loss_d_train/epoch",
                "scalar_value": train_loss_d / len(train_loader),
                "global_step": epoch,
            }
        )

        logger.log_info(message)

        with torch.no_grad():
            soundstream.eval()
            mfd.eval()
            valid_loss_d = 0.0
            valid_w_loss_d = 0.0
            valid_loss_g = 0.0
            valid_commit_loss = 0.0
            valid_adv_g_loss = 0.0
            valid_feat_loss = 0.0
            valid_rec_loss = 0.0
            if args.distributed:
                valid_loader.sampler.set_epoch(epoch)
            for x in tqdm(valid_loader):
                x = x.to(args.device)
                for optimizer_idx in [0, 1]:
                    x_wav = get_input(x)
                    G_x, commit_loss, _ = soundstream(x_wav)
                    if optimizer_idx == 0:
                        valid_commit_loss += commit_loss

                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mfd(
                            x_wav.contiguous(), G_x.contiguous()
                        )
                        total_loss_g, adv_g_loss, feat_loss, rec_loss = criterion_g(
                            commit_loss,
                            x_wav,
                            G_x,
                            fmap_f_r,
                            fmap_f_g,
                            y_df_hat_r,
                            y_df_hat_g,
                            args=args,
                        )
                        valid_loss_g += total_loss_g.item()
                        valid_adv_g_loss += adv_g_loss.item()
                        valid_feat_loss += feat_loss.item()
                        valid_rec_loss += rec_loss.item()
                    else:

                        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mfd(
                            x_wav.contiguous().detach(), G_x.contiguous().detach()
                        )
                        w_loss_d, loss_d = criterion_d(
                            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g
                        )
                        valid_w_loss_d += w_loss_d.item()
                        valid_loss_d += loss_d.item()

            # if dist.get_rank() == 0:
            val_loss = valid_loss_g / len(valid_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch

                # save the model
                latest_model_soundstream = soundstream.state_dict().copy()
                latest_mfd = mfd.state_dict().copy()

                latest_save = {}
                latest_save["codec_model"] = latest_model_soundstream
                latest_save["mfd"] = latest_mfd
                latest_save["epoch"] = epoch
                latest_save["steps"] = global_step
                latest_save["optimizer_g"] = optimizer_g.state_dict()
                latest_save["optimizer_d"] = optimizer_d.state_dict()
                latest_save["lr_scheduler_g"] = lr_scheduler_g.state_dict()
                latest_save["lr_scheduler_d"] = lr_scheduler_d.state_dict()
                torch.save(
                    latest_save,
                    args.PATH + f"/ckpt_S{global_step}_L{best_val_loss}.pth",
                )

            # made changes here, average out the total losses by dividing with the num_batches
            message = "<epoch:{:d}, total_loss_g_valid:{:.4f}, recon_loss_valid:{:.4f}, adversarial_loss_valid:{:.4f}, feature_loss_valid:{:.4f}, commit_loss_valid:{:.4f}, valid_loss_d:{:.4f}, valid_w_loss_d:{:.4f}, best_epoch:{:d}>".format(
                epoch,
                valid_loss_g / len(valid_loader),
                valid_rec_loss / len(valid_loader),
                valid_adv_g_loss / len(valid_loader),
                valid_feat_loss / len(valid_loader),
                valid_commit_loss / len(valid_loader),
                valid_loss_d / len(valid_loader),
                valid_w_loss_d / len(valid_loader),
                best_val_epoch,
            )

            # made changes here, average out the total losses before logging
            logger.add_scalar(
                **{
                    "tag": "total_loss_g_valid/epoch",
                    "scalar_value": valid_loss_g / len(valid_loader),
                    "global_step": epoch,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "recon_loss_valid/epoch",
                    "scalar_value": valid_rec_loss / len(valid_loader),
                    "global_step": epoch,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "adversarial_loss_valid/epoch",
                    "scalar_value": valid_adv_g_loss / len(valid_loader),
                    "global_step": epoch,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "feature_loss_valid/epoch",
                    "scalar_value": valid_feat_loss / len(valid_loader),
                    "global_step": epoch,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "commit_loss_valid/epoch",
                    "scalar_value": valid_commit_loss / len(valid_loader),
                    "global_step": epoch,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "total_w_loss_d_valid/epoch",
                    "scalar_value": valid_w_loss_d / len(valid_loader),
                    "global_step": epoch,
                }
            )
            logger.add_scalar(
                **{
                    "tag": "total_loss_d_valid/epoch",
                    "scalar_value": valid_loss_d / len(valid_loader),
                    "global_step": epoch,
                }
            )

            logger.log_info(message)

    logger.close()


if __name__ == "__main__":
    main()
