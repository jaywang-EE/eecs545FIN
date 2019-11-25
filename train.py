#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, load_checkpoints, save_checkpoints, Discriminator_G, Discriminator_L

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = datetime.now().strftime("%m%d_%H%M"))
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-d', '--debug', type=str, default="debug")
    parser.add_argument('-w', '--winsize', type=int, default=48)

    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument('-s', "--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def train_gmm(opt, train_loader, model, board):
    # change
    #print(len(train_loader.data_loader))
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    #change
    loss_sum = 0
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        loss = criterionL1(warped_cloth, im_c)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()   
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            
        #change
        if (step+1) % len(train_loader.data_loader):
            t = time.time() - iter_start_time
            loss_avr = loss_sum/(step+1) 
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    (step+1, t, loss_avr, loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)
            loss_sum = 0

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, d_g, d_l, board):
    model.cuda()
    model.train()
    d_g.cuda()
    d_g.train()
    d_l.cuda()
    d_l.train()

    dis_label = Variable(torch.FloatTensor(opt.batch_size)).cuda()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    criterionGAN = nn.BCELoss()#MSE
    
    # optimizer
    optimizerG = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerDG = torch.optim.Adam(d_g.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerDL = torch.optim.Adam(d_l.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    schedulerDG = torch.optim.lr_scheduler.LambdaLR(optimizerDG, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    schedulerDL = torch.optim.lr_scheduler.LambdaLR(optimizerDL, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        #prep
        inputs = train_loader.next_batch()
            
        im = inputs['image'].cuda()#sz=b*3*256*192
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        batch_size = im.size(0)

        optimizerDG.zero_grad()
        optimizerDL.zero_grad()
        #D_real
        dis_label.data.resize_(batch_size).fill_(1)
        dis_g_output = d_g(im)
        
        errDg_real = criterionGAN(dis_g_output, dis_label)
        errDg_real.backward()

        #tom_gen
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)
        
        real_crop, fake_crop = random_crop(im, p_tryon, opt.winsize)
        dis_l_output = d_l(real_crop)
        errDl_real = criterionGAN(dis_l_output, dis_label)
        errDl_real.backward()

        #D_fake
        dis_label.data.fill_(0)

        dis_g_output = d_g(p_tryon.detach())
        errDg_fake = criterionGAN(dis_g_output, dis_label)
        errDg_fake.backward()
        optimizerDG.step()

        dis_l_output = d_l(fake_crop.detach())
        errDl_fake = criterionGAN(dis_l_output, dis_label)
        errDl_fake.backward()
        optimizerDL.step()

        #tom_train
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask

        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()
        
        #tensorboradX
        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        if (step+1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            
            loss_dict = {"metric":loss.item(), "L1":loss_l1.item(), "VGG":loss_vgg.item(), 
                         "Mask":loss_mask.item(), "DG":((errDg_fake+errDg_real)/2).item(), 
                         "DL":((errDl_fake+errDl_real)/2).item()}
            print('step: %8d, time: %.3f'%(step+1, t, loss.item()), end="")
            
            board_add_images(board, 'combine', visuals, step+1)
            for k, v in loss_dict.getitems():
                print('%s: %.4f'%(k, v), end="")
                board.add_scalar(k, v, step+1)
            print()
            
        if (step+1) % opt.save_count == 0:
            sm_image(combine_images(im, p_tryon, real_crop, fake_crop), "combined%d.jpg"%step, opt.debug)
            save_checkpoints(model, d_g, d_l, 
                os.path.join(opt.checkpoint_dir, opt.stage +'_'+ opt.name, "step%06d"%step, '%s.pth'))



def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        d_g= Discriminator_G(opt)
        d_l= Discriminator_L(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            if not os.path.isdir(opt.checkpoint):
                raise NotImplementedError('checkpoint should be dir, not file: %s' % opt.checkpoint)
            load_checkpoints(model, d_g, d_l, os.path.join(opt.checkpoint, "%s.pth"))
        train_tom(opt, train_loader, model, d_g, d_l, board)

        save_checkpoints(model, d_g, d_l, 
            os.path.join(opt.checkpoint_dir, opt.stage +'_'+ opt.name+"_final", '%s.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
