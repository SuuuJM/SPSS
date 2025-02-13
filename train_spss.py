import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from networks.vnet_projection import VNet
from networks.unet import UNet
from networks.spatial import SpatialTransformer
from utils import ramps, losses
from utils.util_pln import *
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from dataloaders.lits import LiTS
from dataloaders.kits import KiTS

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/LA/processed_h5',
                    help='data load root path')
parser.add_argument('--exp', type=str, default='spss', help='name of experiment')
parser.add_argument('--dataset', type=str, default='la', help='dataset to use')
parser.add_argument('--label_num', type=int, default=16, help='number of labeled samples')

parser.add_argument('--pretrain_reg_epoch', type=int, default=200, help='pretrain epoch number for reg module')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled samples batch size')
parser.add_argument('--reg_iter', type=int, default=10, help='number of registration training per iter')

parser.add_argument('--seg_lr', type=float, default=0.01, help='seg learning rate')
parser.add_argument("--reg_lr", type=float, default=0.0001, help="reg learning rate")
parser.add_argument("--sim_loss", type=str, default='mse', help="similarity loss: mse or ncc")
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema decay')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency rampup')

parser.add_argument('--w_p_max', type=float, default=0.9, help='max weight of seg pred')
parser.add_argument('--alpha_p', type=float, default=0.2, help='final pseudo-label cal: seg net prediction power')
parser.add_argument('--alpha_d', type=float, default=0.8, help='slice-wise weight')
parser.add_argument('--save_img', type=int, default=6000, help='img saving iterations')
parser.add_argument('--w_dice', type=float, default=0.3, help='dice loss')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

parser.add_argument('--temperature', type=float, help='temperature coefficient', default=0.8)
parser.add_argument('--consist_lambda', type=float, help='consist nums', default=0.1)
parser.add_argument('--consist_lambda_rate', type=float, help='lambda update rate', default=1.01)

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "./model_" + args.dataset + "/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.seg_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def cutmix(imageA1, imageA2, labelA1, labelA2):
    image1, image2, label1, label2 = imageA1.clone(), imageA2.clone(), labelA1.clone(), labelA2.clone()
    batch_size, num_channels, height, width, depth = image1.shape

    cut_height = int(np.random.uniform(height * 0.1, height * 0.9))
    cut_width = int(np.random.uniform(width * 0.1, width * 0.9))
    y1 = np.random.randint(0, height - cut_height)
    x1 = np.random.randint(0, width - cut_width)
    image1[:, :, y1:y1 + cut_height, x1:x1 + cut_width, :] = image2[:, :, y1:y1 + cut_height, x1:x1 + cut_width, :]
    label1[:, :, y1:y1 + cut_height, x1:x1 + cut_width, :] = label2[:, :, y1:y1 + cut_height, x1:x1 + cut_width, :]

    return image1, label1


def check_input(input_logits, target_logits):
    input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
    target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
    return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs


class getTopK(nn.Module):
    def __init__(self, k: int, dim: int = -1, gumble: bool = False):
        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):
        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            return torch.topk(logits, self.k, dim=self.dim)


def contrastiveLoss(input, positive, negative, input_logits, negative_logits, bidirectional=True):
    tau = 0.07
    sample_num = 50
    topk = getTopK(k=sample_num, dim=0)
    B, C, *spatial_size = input.shape
    spatial_dims = len(spatial_size)

    norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C),
                             dim=-1)
    norm_positive = F.normalize(positive.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C),
                                dim=-1)
    norm_negative = F.normalize(negative.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C),
                                dim=-1)
    seg_input, seg_negative, _, negative_prob = check_input(input_logits, negative_logits)

    seg_input = seg_input.flatten()
    seg_negative = seg_negative.flatten()
    negative_prob = negative_prob.flatten()

    positive = F.cosine_similarity(norm_input, norm_positive, dim=-1)

    diff_cls_matrix = (seg_input.unsqueeze(0) != seg_negative.unsqueeze(1)).to(torch.float32)
    prob_matrix = negative_prob.unsqueeze(1).expand_as(diff_cls_matrix)
    masked_target_prob_matrix = diff_cls_matrix * prob_matrix

    sampled_negative_indices = topk(masked_target_prob_matrix).indices
    sampled_negative = norm_negative[sampled_negative_indices]
    negative_sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0).expand_as(sampled_negative),
                                              sampled_negative, dim=-1)

    nominator = torch.exp(positive / tau)
    denominator = torch.exp(negative_sim_matrix / tau).sum(dim=0) + nominator
    loss = -torch.log(nominator / (denominator + 1e-8)).mean()
    if bidirectional:
        alter_negative_sim_matrix = F.cosine_similarity(norm_positive.unsqueeze(0).expand_as(sampled_negative),
                                                        sampled_negative, dim=-1)
        alter_denominator = torch.exp(alter_negative_sim_matrix / tau).sum(dim=0) + nominator
        alter_loss = -torch.log(nominator / (alter_denominator + 1e-8)).mean()
        loss = loss + alter_loss
    return loss


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    criterion_ce = nn.CrossEntropyLoss(reduction='none')
    criterion_dsc = losses.DiceLoss(num_classes=2, is_3d=True)

    if args.dataset == 'la':
        num_classes = 2
        patch_size = (112, 112, 80)
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 80))
    elif args.dataset == 'lits':
        num_classes = 2
        patch_size = (176, 176, 64)
        db_train = LiTS(base_dir=train_data_path,
                        split='train',
                        transform=transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                        ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 100))
    elif args.dataset == 'kits':
        num_classes = 2
        patch_size = (176, 176, 64)
        db_train = KiTS(base_dir=train_data_path,
                        split='train',
                        transform=transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                        ]))
        labeled_idxs = list(range(args.label_num))
        unlabeled_idxs = list(range(args.label_num, 190))

    reg_size = (patch_size[0], patch_size[1])
    num_slice = patch_size[2]  # 80
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def create_regnet(vol_size):
        nf_enc = [16, 32, 32, 32]
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
        net = UNet(len(vol_size), nf_enc, nf_dec).cuda()
        return net


    def create_model(ema=False):
        model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    reg_unet = create_regnet(reg_size)
    reg_stn = SpatialTransformer(reg_size).cuda()
    reg_stn_label = SpatialTransformer(reg_size, mode="nearest").cuda()

    model = create_model()
    ema_model = create_model(ema=True)

    trainLoader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    reg_unet.train()
    reg_stn.train()
    reg_stn_label.eval()
    model.train()
    ema_model.train()

    reg_optimizer = optim.Adam(reg_unet.parameters(), lr=args.reg_lr)
    seg_optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    sim_loss_fn = losses.ncc_loss if args.sim_loss == "ncc" else losses.mse_loss
    grad_loss_fn = losses.gradient_loss
    consistency_criterion = losses.softmax_mse_loss

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainLoader)))

    iter_num = 0
    max_epoch = args.pretrain_reg_epoch + max_iterations // len(trainLoader) + 1
    lr_ = base_lr

    loss_ce_dice_nomask = -1

    ld = args.consist_lambda
    tau = args.temperature

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainLoader):
            sample, imageA1, imageA2 = sampled_batch
            imageA1, imageA2 = imageA1.cuda(), imageA2.cuda()
            volume_batch, label_batch, label_full_batch = sample['image'], sample['label'], sample['label_full']

            slice_volume_batch = volume_batch.cpu().detach().numpy()
            slice_volume_batch = np.squeeze(slice_volume_batch, 1)
            slice_volume_batch = np.transpose(slice_volume_batch, (0, 3, 1, 2))
            slice_volume_batch = slice_volume_batch.reshape(
                (-1, slice_volume_batch.shape[2], slice_volume_batch.shape[3]))

            slice_label_batch = label_batch.cpu().detach().numpy()
            slice_label_batch = slice_label_batch[:labeled_bs]
            lbl_idx = label_index(slice_label_batch, labeled_bs, num_slice)
            slice_label_batch = np.transpose(slice_label_batch, (0, 3, 1, 2))
            slice_label_batch = slice_label_batch.reshape(
                (-1, slice_label_batch.shape[2], slice_label_batch.shape[3]))

            train_generator = vxm_data_generator(num_slice, x_data=slice_volume_batch, x_label=None, batch_size=32)

            for i in range(1, args.reg_iter + 1):
                input_sample = next(train_generator)
                input_moving, input_fixed = torch.from_numpy(input_sample[0]).cuda().float(), torch.from_numpy(
                    input_sample[1]).cuda().float()

                flow_m2f = reg_unet(input_moving, input_fixed)
                m2f = reg_stn(input_moving, flow_m2f)

                reg_loss = sim_loss_fn(m2f, input_fixed)

                reg_optimizer.zero_grad()
                reg_loss.backward()
                reg_optimizer.step()

            if epoch_num >= args.pretrain_reg_epoch:
                reg_unet.eval()
                reg_pred = regnet_test(slice_volume_batch, slice_label_batch, reg_unet, reg_stn_label,
                                       label_batch[:labeled_bs].shape, num_slice, lbl_idx)
                reg_unet.train()

                volume_batch = volume_batch.cuda()
                noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
                ema_inputs = volume_batch + noise

                outputs, _, _ = model(volume_batch)
                outputs_soft = F.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_outputs, _, _ = ema_model(ema_inputs)
                    ema_outputs_soft = F.softmax(ema_outputs, dim=1)

                consistency_weight = get_current_consistency_weight(iter_num // 150)
                b, h, w, d = label_batch.size()
                if loss_ce_dice_nomask >= ld or loss_ce_dice_nomask == -1:
                    consist_rate = 0.1 * min(consistency_weight * tau, 1)
                    num_consist = int(consist_rate * (h * w * d))
                else:
                    consist_rate = (1 - loss_ce_dice_nomask / ld) * min(consistency_weight * tau, 1)
                    num_consist = int(consist_rate * (h * w * d))

                T = 8
                volume_batch_r = volume_batch.repeat(2, 1, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
                for i in range(T // 2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2 * stride * i:2 * stride * (i + 1)], _, _ = ema_model(ema_inputs)
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, 2, 112, 112, 80)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                               keepdim=True)
                b, c, h, w, d = uncertainty.shape
                uncertainty = uncertainty.permute([1, 0, 2, 3, 4]).reshape([c, b * h * w * d]).squeeze(dim=0)

                if num_consist == 0:
                    length = uncertainty.shape[0]
                    mask = torch.zeros(length)
                    mask = mask.reshape(b, h, w, d).cuda()
                else:
                    idxcert_sorted = torch.argsort(uncertainty)[:num_consist]

                    length = uncertainty.shape[0]
                    mask = torch.zeros(length)
                    mask[idxcert_sorted] = 1

                    mask = mask.reshape(b, h, w, d).cuda()

                with torch.no_grad():
                    label1, _, _ = ema_model(imageA1)
                    label2, _, _ = ema_model(imageA2)
                image12_strong, label12 = cutmix(imageA1, imageA2, label1, label2)

                label12_strong, project_negative, project_map_negative = model(image12_strong)

                label12_soft = torch.softmax(label12, dim=1)
                label12_pseudo = torch.argmax(label12_soft, dim=1)
                label12_strong_soft = torch.softmax(label12_strong, dim=1)
                label12_strong_pseudo = torch.argmax(label12_strong_soft, dim=1)

                stu_output_A1, project1, project_map1 = model(imageA1)
                stu_output_A2, project2, project_map2 = model(imageA2)

                w_p = args.w_p_max * pow(iter_num / max_iterations, args.alpha_p)
                seg_pred, prediction = get_prediction(reg_pred, w_p, ema_outputs_soft, lbl_idx, labeled_bs)

                if iter_num % args.save_img == 0:
                    save_images(volume_batch, label_batch, label_full_batch, reg_pred, seg_pred, prediction,
                                snapshot_path, iter_num)

                label_batch[:labeled_bs] = torch.from_numpy(prediction).long()
                label_batch = label_batch.cuda()
                weight = get_ce_weight(label_batch, reg_pred, seg_pred, labeled_bs, iter_num, lbl_idx, num_slice,
                                       args.alpha_d)
                loss_sup_ce = losses.pixel_weighted_ce_loss(outputs[:labeled_bs], weight, label_batch[:labeled_bs],
                                                            labeled_bs)
                loss_sup_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                loss_sup = 0.5 * (loss_sup_ce + loss_sup_dice * args.w_dice)

                mask_one = torch.ones(length)
                mask_one = mask_one.reshape(b, h, w, d).cuda()

                consistency_weight = get_current_consistency_weight(iter_num // 150)
                loss_ce = criterion_ce(label12_strong.softmax(dim=1), label12_pseudo.detach())
                loss_ce = ((mask * (loss_ce)).sum()) / (mask.sum() + 1e-16)
                with torch.no_grad():
                    loss_ce_nomask = (
                            ((mask_one * (loss_ce.detach())).sum()) / (mask_one.sum() + 1e-16)).detach().item()
                loss_dice = criterion_dsc(label12_strong.softmax(dim=1), label12_pseudo.detach(),
                                          ignore=((mask == 0)).float())
                loss_u = consistency_weight * (loss_ce + loss_dice)
                with torch.no_grad():
                    loss_dice_nomask = criterion_dsc(label12_strong.softmax(dim=1),
                                                     label12_pseudo.detach()).float().detach().item()
                    loss_ce_dice_nomask = loss_ce_nomask + loss_dice_nomask

                loss_contrastive = consistency_weight * torch.utils.checkpoint.checkpoint(contrastiveLoss, project1,
                                                                                          project2, project_negative,
                                                                                          project_map1,
                                                                                          project_map_negative)

                loss = loss_sup + loss_u + loss_contrastive

                seg_optimizer.zero_grad()
                loss.backward()
                seg_optimizer.step()

                update_ema_variables(model, ema_model, args.ema_decay, iter_num)

                slice_label_batch = np.transpose(prediction, (0, 3, 1, 2))
                slice_label_batch = slice_label_batch.reshape(
                    (-1, slice_label_batch.shape[2], slice_label_batch.shape[3]))

                train_generator = vxm_data_generator(num_slice, x_data=slice_volume_batch[:slice_label_batch.shape[0]],
                                                     x_label=slice_label_batch, batch_size=32)
                for i in range(1, args.reg_iter + 1):
                    input_sample, input_label = next(train_generator)
                    input_moving, input_fixed = torch.from_numpy(input_sample[0]).cuda().float(), torch.from_numpy(
                        input_sample[1]).cuda().float()
                    input_moving_label, input_fixed_label = torch.from_numpy(
                        input_label[0]).cuda().float(), torch.from_numpy(input_label[1]).cuda().float()

                    flow_m2f = reg_unet(input_moving, input_fixed)
                    m2f = reg_stn(input_moving, flow_m2f)
                    m2f_label = reg_stn(input_moving_label, flow_m2f)

                    sim_loss = sim_loss_fn(m2f, input_fixed)
                    dice_loss = losses.dice_loss(m2f_label, input_fixed_label)
                    reg_loss = sim_loss + dice_loss

                    reg_optimizer.zero_grad()
                    reg_loss.backward()
                    reg_optimizer.step()

                iter_num = iter_num + 1
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_sup', loss_sup, iter_num)
                writer.add_scalar('train/loss_u', loss_u, iter_num)
                writer.add_scalar('loss/loss_contrastive', loss_contrastive, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/num_consist', num_consist, iter_num)
                writer.add_scalar('train/consist_rate', consist_rate, iter_num)

                logging.info(
                    'iteration %d : loss : %f, loss_sup: %f, loss_u : %f, loss_ce : %f, loss_dice : %f, loss_contrastive : %f, loss_weight: %f, num_consist: %d' %
                    (iter_num, loss.item(), loss_sup.item(), loss_u.item(), loss_ce.item(), loss_dice.item(),
                     loss_contrastive.item(), consistency_weight, num_consist))

                # change lr
                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in seg_optimizer.param_groups:
                        param_group['lr'] = lr_
                if iter_num % 500 == 0:
                    tau = tau * 1.4
                if iter_num % 2000 == 0:
                    save_mode_path_reg = os.path.join(snapshot_path, 'reg_iter_' + str(iter_num) + '.pth')
                    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                    keys_to_ignore = ['projector.0.weight', 'projector.0.bias', 'projector.1.weight',
                                      'projector.1.bias', 'projector.2.weight', 'projector.2.bias']

                    reg = reg_unet.state_dict()
                    reg_dict = {k: v for k, v in reg.items() if k not in keys_to_ignore}
                    stu = model.state_dict()
                    stu_dict = {k: v for k, v in stu.items() if k not in keys_to_ignore}

                    torch.save(reg_dict, save_mode_path_reg)
                    torch.save(stu_dict, save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

                if iter_num >= max_iterations:
                    break
        if iter_num >= max_iterations:
            break
        if epoch_num > 200:
            ld = ld * args.consist_lambda_rate
    keys_to_ignore = ['projector.0.weight', 'projector.0.bias', 'projector.1.weight', 'projector.1.bias',
                      'projector.2.weight', 'projector.2.bias']

    reg = reg_unet.state_dict()
    reg_dict = {k: v for k, v in reg.items() if k not in keys_to_ignore}
    stu = model.state_dict()
    stu_dict = {k: v for k, v in stu.items() if k not in keys_to_ignore}
    save_mode_path_reg = os.path.join(snapshot_path, 'reg_iter_' + str(iter_num) + '.pth')
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(reg_dict, save_mode_path_reg)
    torch.save(stu_dict, save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
