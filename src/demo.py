from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import shutil
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import cv2
import numpy as np


logger.setLevel(logging.INFO)


def combine_frames(frame_1, frame_2):
    """Combine two frames side by side."""
    height = max(frame_1.shape[0], frame_2.shape[0])
    width = frame_1.shape[1] + frame_2.shape[1]
    combined_frame = np.zeros((height, width, 3), dtype=np.uint8)

    combined_frame[:frame_1.shape[0], :frame_1.shape[1], :] = frame_1
    combined_frame[:frame_2.shape[0], frame_1.shape[1]:frame_1.shape[1] + frame_2.shape[1], :] = frame_2

    return combined_frame


class CombinedVideoLoader:
    def __init__(self, video1_loader, video2_loader, img_size):
        self.video1_loader = video1_loader
        self.video2_loader = video2_loader
        self.length = min(len(video1_loader), len(video2_loader))
        self.img_size = img_size

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.length:
            raise StopIteration
        _, img1, frame_1 = next(self.video1_loader)
        _, img2, frame_2 = next(self.video2_loader)
        combined_frame = combine_frames(frame_1, frame_2)

        # Resize the combined frame to the desired size
        combined_frame_resized = cv2.resize(combined_frame, self.img_size)

        # Normalize RGB
        img = combined_frame_resized[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        self.idx += 1
        return self.idx, img, combined_frame_resized  # Return resized combined frame for both img and img0

    def __len__(self):
        return self.length


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    # Setting file names.
    slash_1 = opt.input_video1.rindex('/') + 1
    ext_1 = opt.input_video1.rindex('.')
    res_vid = opt.input_video1[slash_1:ext_1] + '_combined.mp4'

    logger.info('Starting tracking...')

    # Load videos
    dataloader_1 = datasets.LoadVideo(opt.input_video1, opt.img_size)
    dataloader_2 = datasets.LoadVideo(opt.input_video2, opt.img_size)

    res_txt = opt.input_video1[slash_1:ext_1] + '_combined.txt'
    result_filename = os.path.join(result_root, res_txt)
    frame_rate = min(dataloader_1.frame_rate,
                     dataloader_2.frame_rate)  # Assumindo que os dois vídeos têm frame rates compatíveis

    # Purging and recreating the frame directory.
    op_dir = osp.join(result_root, 'frame')
    if osp.exists(op_dir) and osp.isdir(op_dir):
        shutil.rmtree(op_dir)
    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    mkdir_if_missing(frame_dir)

    # Cria o CombinedVideoLoader
    combined_loader = CombinedVideoLoader(dataloader_1, dataloader_2, opt.img_size)

    # Realiza o rastreamento
    eval_seq(opt, combined_loader, 'mot', result_filename, save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus != [-1])

    # Combine the frames into the video.
    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, res_vid)
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'),
                                                                                  output_video_path)
        os.system(cmd_str)