from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import csv
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
from google.colab.patches import cv2_imshow

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(base_filename, results, data_type, max_lines=1000):
    file_count = 0
    current_lines = 0
    complete_filename = f"{base_filename.replace('.csv', '')}_part{file_count}.csv"

    def write_header(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            if data_type == 'mot':
                writer.writerow(['frame', 'id', 'x1', 'y1', 'w', 'h', 'confidence', 'class', 'visibility', 'truncated'])
            elif data_type == 'kitti':
                writer.writerow(['frame', 'id', 'class', 'truncated', 'occluded', 'alpha', 'x1', 'y1', 'x2', 'y2', 'height', 'width','length', 'location', 'rotation_y', 'score'])
            else:
                raise ValueError(data_type)

    write_header(complete_filename)

    with open(complete_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                if data_type == 'mot':
                    writer.writerow([frame_id, track_id, x1, y1, w, h, 1, -1, -1, -1])
                elif data_type == 'kitti':
                    writer.writerow([frame_id, track_id, 'pedestrian', 0, 0, -10, x1, y1, x2, y2, -10, -10, -10, -1000, -1000, -1000, -10])
                current_lines += 1
                if current_lines >= max_lines:
                    file_count += 1
                    complete_filename = f"{base_filename.replace('.csv', '')}_part{file_count}.csv"
                    write_header(complete_filename)
                    current_lines = 0

    logger.info(f'Saved complete results to {complete_filename}')


def write_results_incremental(base_filename, frame_id, tlwhs, track_ids, data_type, max_lines=1000):
    file_count = 0
    current_lines = 0
    incremental_filename = f"{base_filename.replace('.csv', '')}_part{file_count}.csv"

    def write_header(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            if data_type == 'mot':
                writer.writerow(['frame', 'id', 'x1', 'y1', 'w', 'h', 'confidence', 'class', 'visibility', 'truncated'])
            elif data_type == 'kitti':
                writer.writerow(
                    ['frame', 'id', 'class', 'truncated', 'occluded', 'alpha', 'x1', 'y1', 'x2', 'y2', 'height',
                     'width', 'length', 'location', 'rotation_y', 'score'])
            else:
                raise ValueError(data_type)

    # Check for existing files and count lines
    while os.path.isfile(incremental_filename):
        with open(incremental_filename, 'r') as f:
            current_lines = sum(1 for line in f) - 1  # Subtract 1 for the header line
        if current_lines < max_lines:
            break
        file_count += 1
        incremental_filename = f"{base_filename.replace('.csv', '')}_part{file_count}.csv"
        current_lines = 0

    if not os.path.isfile(incremental_filename):
        write_header(incremental_filename)

    with open(incremental_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if data_type == 'kitti':
            frame_id -= 1
        for tlwh, track_id in zip(tlwhs, track_ids):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            if data_type == 'mot':
                writer.writerow([frame_id, track_id, x1, y1, w, h, 1, -1, -1, -1])
            elif data_type == 'kitti':
                writer.writerow(
                    [frame_id, track_id, 'pedestrian', 0, 0, -10, x1, y1, x2, y2, -10, -10, -10, -1000, -1000, -1000,
                     -10])
            current_lines += 1
            if current_lines >= max_lines:
                file_count += 1
                incremental_filename = f"{base_filename.replace('.csv', '')}_part{file_count}.csv"
                write_header(incremental_filename)
                current_lines = 0

    logger.info(f'Appended incremental results to {incremental_filename}')

def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def save_image_with_boxes(save_dir, img, online_targets, frame_idx, time_data, waiting_area, service_area):
    # Desenhar as áreas de espera e de atendimento
    cv2.rectangle(img, (waiting_area[0], waiting_area[1]), (waiting_area[2], waiting_area[3]), (255, 0, 0), 2)
    cv2.putText(img, 'Waiting Area', (waiting_area[0], waiting_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)

    cv2.rectangle(img, (service_area[0], service_area[1]), (service_area[2], service_area[3]), (0, 0, 255), 2)
    cv2.putText(img, 'Service Area', (service_area[0], service_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        # Desenhar a bounding box ao redor do objeto rastreado
        cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), (0, 255, 0),
                      2)

        # Adicionar o ID do objeto
        cv2.putText(img, f'ID: {tid}', (int(tlwh[0]), int(tlwh[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Adicionar tempos
        total_time = time_data[tid]['total']
        waiting_time = time_data[tid]['waiting']
        service_time = time_data[tid]['service']

        # Adicionar texto com tempo total
        cv2.putText(img, f'Total: {total_time:.2f}s', (int(tlwh[0]), int(tlwh[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # Adicionar texto com tempo de espera
        cv2.putText(img, f'Waiting: {waiting_time:.2f}s', (int(tlwh[0]), int(tlwh[1] - 50)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

        # Adicionar texto com tempo de atendimento
        cv2.putText(img, f'Service: {service_time:.2f}s', (int(tlwh[0]), int(tlwh[1] - 70)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # Caminho de saída para a imagem processada
    output_path = os.path.join(save_dir, f'{frame_idx:05d}.jpg')
    cv2.imwrite(output_path, img)


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    tracking_data = {}  # Para armazenar os dados de rastreamento

    # Definir áreas de interesse (substituir pelos valores reais)
    waiting_area = (100, 200, 400, 600)
    service_area = (500, 200, 800, 600)

    # Inicializar dicionário para tempos
    time_data = {}

    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

        results.append((frame_id + 1, online_tlwhs, online_ids))

        for tlwh, track_id in zip(online_tlwhs, online_ids):
            if track_id not in time_data:
                time_data[track_id] = {'wait_start': None, 'service_start': None, 'wait_time': 0, 'service_time': 0}

            # Calcular o centro do bounding box
            x1, y1, w, h = tlwh
            center = (x1 + w / 2, y1 + h / 2)

            if is_in_area(center, waiting_area):
                if time_data[track_id]['wait_start'] is None:
                    time_data[track_id]['wait_start'] = frame_id
                if time_data[track_id]['service_start'] is not None:
                    time_data[track_id]['service_time'] += frame_id - time_data[track_id]['service_start']
                    time_data[track_id]['service_start'] = None

            elif is_in_area(center, service_area):
                if time_data[track_id]['wait_start'] is not None:
                    time_data[track_id]['wait_time'] += frame_id - time_data[track_id]['wait_start']
                    time_data[track_id]['wait_start'] = None
                if time_data[track_id]['service_start'] is None:
                    time_data[track_id]['service_start'] = frame_id

            else:
                if time_data[track_id]['wait_start'] is not None:
                    time_data[track_id]['wait_time'] += frame_id - time_data[track_id]['wait_start']
                    time_data[track_id]['wait_start'] = None
                if time_data[track_id]['service_start'] is not None:
                    time_data[track_id]['service_time'] += frame_id - time_data[track_id]['service_start']
                    time_data[track_id]['service_start'] = None

        # Incrementally write results
        write_results_incremental(result_filename, frame_id + 1, online_tlwhs, online_ids, data_type)

        timer.toc()
        frame_id += 1

    write_results(result_filename, results, data_type)
    logger.info('Time elapsed: {:.2f}s'.format(timer.total_time))
    logger.info(f'Results saved to {result_filename}')
    return frame_id, timer.average_time, time_data


def is_in_area(center, area):
    x, y = center
    x1, y1, x2, y2 = area
    return x1 <= x <= x2 and y1 <= y <= y2


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=True,
         save_images=False,
         save_videos=False)