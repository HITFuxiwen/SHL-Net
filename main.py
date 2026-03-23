import os.path as osp
from eval import Eval_thread
from dataloader import EvalDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def evaluate():
    pred_dir = './test/'
    output_dir = ''
    gt_dir = './Data/'

    # pred_dir= osp.join(pred_dir, 'ISTD/RGB_VST_wo/')
    # pred_dir= osp.join(pred_dir, 'ResNet50/')

    gt_dir = osp.join(osp.join(gt_dir, 't_gt/'))

    loader = EvalDataset(pred_dir, gt_dir)
    thread = Eval_thread(loader, output_dir, cuda=True)
    print(thread.run())

    pred_dir = './res/'
    output_dir = ''
    gt_dir = './Data/'


    gt_dir = osp.join(osp.join(gt_dir, 't_gt/'))

    loader = EvalDataset(pred_dir, gt_dir)
    thread = Eval_thread(loader, output_dir, cuda=True)
    print('Gele')
    print(thread.run())
# evaluate()