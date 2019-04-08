from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings

import numpy as np
import copy
from PIL import Image
import torch

def read_image(path):

    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img

class Dataset(object):

    def __init__(self, query, gallery, amount_gen, transform=None, mode='query', **kwargs):
        # self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.amount_gen = amount_gen
        self.mode = mode

        if self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def parse_data(self, data):
        pids = set()
        cams = set()
        imids = set()
        for _, _, pid, camid, imid in data:
            pids.add(pid)
            cams.add(camid)
            imids.add(imid)
        print(len(imids))
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def check_before_run(self, required_files):
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
              num_query_pids, len(self.query), num_query_cams,
              num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg


class ImageDataset(Dataset):

    def __init__(self, query, gallery, amount_gen, **kwargs):
        super(ImageDataset, self).__init__(query, gallery, amount_gen, **kwargs)

    def __getitem__(self, index):
        img_path, gen_path_list, pid, camid, imid = self.data[index]
        # img = read_image(img_path)
        # if self.transform is not None:
        #     img = self.transform(img)
        gen_img_list = []
        for i in range(self.amount_gen):
            gen_img = read_image(gen_path_list[i])
            if self.transform is not None:
                gen_img = self.transform(gen_img)
            gen_img_list.append(gen_img)
        return gen_img_list[0], gen_img_list[1], gen_img_list[2], gen_img_list[3], pid, camid, imid


class Market1501(ImageDataset):

    def __init__(self, root='/home/tianlab/hengheng/reid', amount_gen=4, **kwargs):
        self.root = '/home/tianlab/hengheng/reid'
        self.dataset_name = 'Market'
        # osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_name)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        # data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        # if osp.isdir(data_dir):
        #     self.data_dir = data_dir
        # else:
        #     warnings.warn('The current data structure is deprecated. Please '
        #                   'put data folders such as "bounding_box_train" under '
        #                   '"Market-1501-v15.09.15".')

        # self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gen_query_dir = osp.join(self.data_dir, 'gen_query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.gen_gallery_dir = osp.join(self.data_dir, 'gen_gallery')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        # self.market1501_500k = market1501_500k

        required_files = [
            self.query_dir,
            self.gen_query_dir,
            self.gallery_dir,
            self.gen_gallery_dir
        ]

        self.check_before_run(required_files)

        self.amount_gen = amount_gen

        # train = self.process_dir(self.train_dir, relabel=True)
        gallery = self.process_dir(self.gallery_dir, self.gen_gallery_dir, self.amount_gen, relabel=False)
        query = self.process_dir(self.query_dir, self.gen_query_dir, self.amount_gen, relabel=False)

        super(Market1501, self).__init__(query, gallery, amount_gen, **kwargs)

    def process_dir(self, dir_path, gen_dir_path, amount_gen, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            # if pid == -1:
            #     continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        miss_count = 0
        imid = 0
        for img_path in img_paths:
            img_name, img_ext = osp.splitext(osp.basename(img_path))

            gen_path_list = []
            if osp.isfile(osp.join(gen_dir_path, img_name + '_gen_{}.png'.format(0))):
                for i in range(amount_gen):
                    gen_path = osp.join(gen_dir_path, img_name + '_gen_{}.png'.format(i))
                    gen_path_list.append(gen_path)
                    pid, camid = map(int, pattern.search(img_path).groups())
                    # if pid == -1:
                    #     continue # junk images are just ignored
                    assert -1 <= pid <= 1501  # pid == 0 means background
                    assert 1 <= camid <= 6
                    # camid -= 1 # index starts from 0
                    if relabel:
                        pid = pid2label[pid]
                data.append((img_path, gen_path_list, pid, camid, imid))
            else:
                miss_count += 1
                    # gen_path_list.append(img_path)
            imid += 1
        # print(data)

        # print(miss_count)
        # sys.exit()

        return data
