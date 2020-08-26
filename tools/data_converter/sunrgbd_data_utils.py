import mmcv
import numpy as np
from concurrent import futures as futures
from os import path as osp
from scipy import io as sio


def random_sampling(points, num_points, replace=None, return_choices=False):
    """Random sampling.
    随机均匀下采样，将点云采样成一定数量的点
    Sampling point cloud to a certain number of points.

    Args:
        points (ndarray): Point cloud.采样的点云
        num_points (int): The number of samples.要采样的点的数量，指定
        replace (bool): Whether the sample is with or without replacement.
        return_choices (bool): Whether to return choices.

    Returns:
        points (ndarray): Point cloud after sampling.
    """

    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace) #允许重复采样
    if return_choices:
        return points[choices], choices # 返回采样得到的点和序号
    else:
        return points[choices]


class SUNRGBDInstance(object):

    def __init__(self, line):
        data = line.split(' ') # classname[1]+gtBb2D(2D bbox)[4]+centroid[3]+coeffs[3]+orientation[2]
        data[1:] = [float(x) for x in data[1:]]
        self.classname = data[0]
        self.xmin = data[1]
        self.ymin = data[2]
        self.xmax = data[1] + data[3]
        self.ymax = data[2] + data[4]
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])# 注意在原数据集中使用mincoor+width+length的形式的
        self.centroid = np.array([data[5], data[6], data[7]]) 
        self.w = data[8] # 原来三个coeff的绝对值分别是bbox3D的宽长高
        self.l = data[9]  # noqa: E741
        self.h = data[10]
        self.orientation = np.zeros((3, )) # 最后一个总是0，代表绕着x和y轴的方向总是0
        self.orientation[0] = data[11]
        self.orientation[1] = data[12]
        self.heading_angle = -1 * np.arctan2(self.orientation[1],
                                             self.orientation[0]) #由于这里的坐标系是反的，所以要乘以-1
        self.box3d = np.concatenate([
            self.centroid,
            np.array([self.l * 2, self.w * 2, self.h * 2, self.heading_angle])# x,y,z,w,l,h,/theta
        ])


class SUNRGBDData(object):
    """SUNRGBD data.

    Generate scannet infos for sunrgbd_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str): Set split type of the data. Default: 'train'.
        use_v1 (bool): Whether to use v1. Default: False.
    """

    def __init__(self, root_path, split='train', use_v1=False):
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path, 'sunrgbd_trainval')
        self.classes = [
            'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
            'night_stand', 'bookshelf', 'bathtub'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes} #字典的格式，如{'bed':0}这种
        self.label2cat = { #字典的格式，如{0:'bed'}这种
            label: self.classes[label] 
            for label in range(len(self.classes))
        }
        assert split in ['train', 'val', 'test'] # split只能是这三个中的一个
        split_file = osp.join(self.split_dir, f'{split}_data_idx.txt') #数据的索引
        mmcv.check_file_exist(split_file)
        self.sample_id_list = map(int, mmcv.list_from_file(split_file))
        self.image_dir = osp.join(self.split_dir, 'image') #存着2D图片
        self.calib_dir = osp.join(self.split_dir, 'calib') #相机和坐标转换的信息
        self.depth_dir = osp.join(self.split_dir, 'depth') #深度信息，txt
        if use_v1:
            self.label_dir = osp.join(self.split_dir, 'label_v1')
        else:
            self.label_dir = osp.join(self.split_dir, 'label')

    def __len__(self):
        return len(self.sample_id_list)

    def get_image(self, idx):
        img_filename = osp.join(self.image_dir, f'{idx:06d}.jpg')
        return mmcv.imread(img_filename)

    def get_image_shape(self, idx):
        image = self.get_image(idx)
        return np.array(image.shape[:2], dtype=np.int32) # 图像的尺寸，不包含通道数

    def get_depth(self, idx):
        depth_filename = osp.join(self.depth_dir, f'{idx:06d}.mat')
        depth = sio.loadmat(depth_filename)['instance']
        return depth

    def get_calibration(self, idx):
        calib_filepath = osp.join(self.calib_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rt = np.array([float(x) for x in lines[0].split(' ')])
        Rt = np.reshape(Rt, (3, 3), order='F') # 3x3的旋转矩阵
        K = np.array([float(x) for x in lines[1].split(' ')]) # 相机内参矩阵
        return K, Rt

    def get_label_objects(self, idx):
        label_filename = osp.join(self.label_dir, f'{idx:06d}.txt')
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [SUNRGBDInstance(line) for line in lines] #处理label
        return objects

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int): Number of threads to be used. Default: 4. # 默认4线程
            has_label (bool): Whether the data has label. Default: True.
            sample_id_list (list[int]): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            # convert depth to points
            SAMPLE_NUM = 50000
            # TODO: Check whether can move the point
            #  sampling process during training.
            pc_upright_depth = self.get_depth(sample_idx)
            pc_upright_depth_subsampled = random_sampling(
                pc_upright_depth, SAMPLE_NUM)

            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            pc_upright_depth_subsampled.tofile(
                osp.join(self.root_dir, 'points', f'{sample_idx:06d}.bin'))

            info['pts_path'] = osp.join('points', f'{sample_idx:06d}.bin')
            img_name = osp.join(self.image_dir, f'{sample_idx:06d}')
            img_path = osp.join(self.image_dir, img_name)
            image_info = {
                'image_idx': sample_idx,
                'image_shape': self.get_image_shape(sample_idx),
                'image_path': img_path
            }
            info['image'] = image_info

            K, Rt = self.get_calibration(sample_idx)
            calib_info = {'K': K, 'Rt': Rt}
            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label_objects(sample_idx)
                annotations = {}
                annotations['gt_num'] = len([
                    obj.classname for obj in obj_list
                    if obj.classname in self.cat2label.keys()
                ])
                if annotations['gt_num'] != 0:
                    annotations['name'] = np.array([
                        obj.classname for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['bbox'] = np.concatenate([
                        obj.box2d.reshape(1, 4) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],
                                                         axis=0)
                    annotations['location'] = np.concatenate([
                        obj.centroid.reshape(1, 3) for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ],
                                                             axis=0)
                    annotations['dimensions'] = 2 * np.array([
                        [obj.l, obj.h, obj.w] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])  # lhw(depth) format
                    annotations['rotation_y'] = np.array([
                        obj.heading_angle for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['index'] = np.arange(
                        len(obj_list), dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat2label[obj.classname] for obj in obj_list
                        if obj.classname in self.cat2label.keys()
                    ])
                    annotations['gt_boxes_upright_depth'] = np.stack(
                        [
                            obj.box3d for obj in obj_list
                            if obj.classname in self.cat2label.keys()
                        ],
                        axis=0)  # (K,8)
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if \
            sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
