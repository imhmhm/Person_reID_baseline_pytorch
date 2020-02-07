import argparse

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/hmhm/reid/Market/pytorch', type=str,
                    help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')

parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')

parser.add_argument('--use_alex', action='store_true', help='use alex')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NASnet')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--PCB_parts', default=6, type=int, help='number of parts in PCB')

parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--warmup', action='store_true', help='use warmup lr_scheduler')
parser.add_argument('--lr', default=0.00035, type=float, help='learning rate')
parser.add_argument('--epoch', default=120, type=int, help='epoch number')
parser.add_argument('--droprate', default=0.0, type=float, help='drop rate')

parser.add_argument('--mixup', action='store_true', help='use mixup')

parser.add_argument('--lsr', action='store_true', help='use label smoothing')

parser.add_argument('--triplet', action='store_true', help='use triplet loss')
parser.add_argument('--margin', default=0.3, type=float, help='metric loss margin')

parser.add_argument('--wt_xent', default=1.0, type=float, help='weight of xent loss')
parser.add_argument('--wt_tri', default=1.0, type=float, help='weight of triplet loss')


parser.add_argument('--use_sampler', action='store_true', help='use batch sampler')
parser.add_argument('--num_per_id', default=4, type=int, help='number of images per id in a batch')
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )

opt = parser.parse_args()
