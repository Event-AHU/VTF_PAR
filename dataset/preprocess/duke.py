import os
import numpy as np
import random
import pickle
import csv
from easydict import EasyDict
from scipy.io import loadmat
import glob
from sentence_transformers import SentenceTransformer

np.random.seed(0)
random.seed(0)

attr_words = [
    'carring backpack',#0
    'carring shoulder bag',#1
    'carring hand bag',#2
    'shoes boots',#3
    'female',#4
    'heaed hat',#5
    'shoes',#6
    'top long',#7
    'pose frontal','pose lateral-frontal','pose lateral','pose lateral-back','pose back','pose varies',#pose [8:14]
    'motion walking','motion running','motion riding','motion staying','motion varies',#motion[14:19]
    'top color black','top color white','top color red','top color purple','top color gray','top color blue','top color green','top color brown','top color complex',#[19:28]
    'bottom color black','bottom color white','bottom color red','bottom color grey','bottom color blue','bottom color green','bottom color brown','bottom color complex'#[28:36]
]


def track_imgpath(track_path):  # 返回为每个tracklet里照片路径的列表
    images_path = glob.glob(os.path.join(track_path, "*.jpg"))
    for count, i in enumerate(images_path):
        images_path[count] = r'/'.join(i.split('\\'))
    return images_path


def generate_imgs(path, track_name):  # 形成一个字典，key=track_name，value=imgs_path
    imgs_path = {}
    for i in track_name:
        tracklet_path = path + '/' + str(i) + '/'
        result = track_imgpath(tracklet_path)
        imgs_path[i] = result
    return imgs_path


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def get_label_embeds(labels):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(labels)
    return embeddings


# 生成一个标签的字典，其中键代表tracklets_id,值为一个list（为样本的标签值）
def generate_label(filename):
    with open(filename, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # 获取CSV文件的表头
        result_dict = {}
        for row in csv_reader:
            row_buf = [int(i) for i in row[1:]]
            result_dict[str(row[0])] = np.array(row_buf)
    return result_dict


def generate_data_description(save_dir):
    dataset = EasyDict()
    dataset.description = 'duke'
    dataset.root = os.path.join(save_dir, 'pad_dataset')
    dataset.attr_name = attr_words
    dataset.words = np.array(attr_words)
    dataset.attr_vectors = get_label_embeds(attr_words)
    result_dict = generate_label("../DUKE_MCMT/duke_annotation/new_encoded.csv")
    trainval_name = []
    test_name = []
    trainval_gt_list = []
    test_gt_list = []
    track_name = []
    track_gt_list = []

    track_name_file = open("../DUKE_MCMT/duke_annotation/duke_track_name.txt", 'r', encoding='utf8').readlines()
    for name in track_name_file:
        curLine = name.strip('\n')
        track_name.append(curLine)

    trainval_name_file = open("../DUKE_MCMT/duke_annotation/trainval_name.txt", 'r', encoding='utf8').readlines()
    for name in trainval_name_file:
        curLine = name.strip('\n')
        trainval_name.append(curLine)
    test_name_file = open("../DUKE_MCMT/duke_annotation/test_name.txt", 'r', encoding='utf8').readlines()
    for name in test_name_file:
        curLine = name.strip('\n')
        test_name.append(curLine)

    for gt in track_name:
        curLine = name.strip('\n')
        track_gt_list.append(result_dict[curLine])

    for gt in trainval_name:
        curLine = name.strip('\n')
        trainval_gt_list.append(result_dict[curLine])

    for gt in test_name_file:
        curLine = name.strip('\n')
        test_gt_list.append(result_dict[curLine])

    # dataset.track_name=track_name
    dataset.test_name = test_name  # 4908
    dataset.trainval_name = trainval_name  # 11452
    dataset.track_name = dataset.trainval_name + dataset.test_name
    dataset.trainval_gt_list = trainval_gt_list
    dataset.test_gt_list = test_gt_list
    dataset.track_gt_list = track_gt_list
    dataset.result_dict = result_dict
    dataset.attr_vectors = get_label_embeds(attr_words)

    dataset.label = np.concatenate((np.array(trainval_gt_list), np.array(test_gt_list)), axis=0)
    assert dataset.label.shape == (2636+2196, 36), \
            f'dataset.label.shape {dataset.label.shape}'

    dataset.partition = EasyDict()
    dataset.attr_name = attr_words
    dataset.partition.test = np.arange(2196, 2196 + 2636)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 2196)  # np.array(range(90000))
    # dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)
    # 包含每个tracklet中图片的地址
    path1 = "/amax/DATA/jinjiandong/VTB-main/dataset/DUKE_MCMT/pad_dataset"
    dataset.track_imgs_path = generate_imgs(path1, track_name)

    with open(os.path.join(save_dir, 'pad_duke.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/amax/DATA/jinjiandong/VTB-main/dataset/DUKE_MCMT/'
    generate_data_description(save_dir)
