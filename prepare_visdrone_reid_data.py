import argparse
import copy
import os
import sys
from os import path
import cv2
import numpy as np

from loguru import logger
from matplotlib.pyplot import pause

from global_id_tracker import GlobalIDTracker

# Global variables
ID_COUNTER = 0  # IDs of objects
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


# 10 valid VisDrone categories
class_indices = (1, 2, 3, 4 ,5, 6, 7, 8 ,9, 10)
class_names = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
CATEGORIES = dict(zip(class_indices, class_names))

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def get_image_list(seq_path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(seq_path):
        for filename in file_name_list:
            apath = maindir +'/' + filename
            ext = path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def main():
    parser = argparse.ArgumentParser(description='Prepare VisDrone-ReID data from VisDrone-MOT train and val sequences.')

    parser.add_argument("--input_dir", default='../datasets/VisDrone2019', type=str,
                        help="path to dataset under evaluation, currently only support VisDrone2019.")
    parser.add_argument('--output_dir', type=str, default='./VisDroneReID_data',
                        help='Path to output folder to be saved to.')
    parser.add_argument('--display', default=True, action="store_true",
                        help='Display sequences with ground-truth tracks.')

    args = parser.parse_args()


    # Input sequences
    input_dir_train = args.input_dir + '/VisDrone2019-MOT-train'
    train_dir_sequences = input_dir_train + '/sequences'
    train_dir_annotations = input_dir_train + '/annotations'

    input_dir_val = args.input_dir + '/VisDrone2019-MOT-val'
    val_dir_sequences = input_dir_val + '/sequences'
    val_dir_annotations = input_dir_val + '/annotations'

    input_dir_test = args.input_dir + '/VisDrone2019-MOT-test-dev'
    test_dir_sequences = input_dir_test + '/sequences'
    test_dir_annotations = input_dir_test + '/annotations'

    # Output ReID dataset
    output_dir = args.output_dir
    output_dir_train = output_dir + '/train'
    output_dir_test = output_dir + '/test'

    # Create output folders
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_dir_train):
        os.mkdir(output_dir_train)
    if not os.path.isdir(output_dir_test):
        os.mkdir(output_dir_test)


    # SEQUENCES: train and val sequences create the ReID train set, whereas test sequences create the ReID test set.
    # Every 10th frame of each sequence is used to crop the objects’ image patches. When evaluating the ReID performance,
    # each image crop of the test set is taken as query image once and is compared against all other test images that
    # serve as gallery.
    train_seqs = ['uav0000013_01073_v', 'uav0000013_01392_v', 'uav0000020_00406_v', 'uav0000071_03240_v', 'uav0000072_04488_v',
                  'uav0000072_05448_v', 'uav0000072_06432_v', 'uav0000076_00720_v', 'uav0000079_00480_v', 'uav0000084_00000_v',
                  'uav0000099_02109_v', 'uav0000124_00944_v', 'uav0000126_00001_v', 'uav0000138_00000_v', 'uav0000140_01590_v',
                  'uav0000143_02250_v', 'uav0000145_00000_v', 'uav0000150_02310_v', 'uav0000218_00001_v', 'uav0000222_03150_v',
                  'uav0000239_03720_v', 'uav0000239_12336_v', 'uav0000243_00001_v', 'uav0000244_01440_v', 'uav0000248_00001_v',
                  'uav0000263_03289_v', 'uav0000264_02760_v', 'uav0000266_03598_v', 'uav0000266_04830_v', 'uav0000270_00001_v',
                  'uav0000273_00001_v', 'uav0000278_00001_v', 'uav0000279_00001_v', 'uav0000281_00460_v', 'uav0000288_00001_v',
                  'uav0000289_00001_v', 'uav0000289_06922_v', 'uav0000295_02300_v', 'uav0000300_00000_v', 'uav0000307_00000_v',
                  'uav0000308_00000_v', 'uav0000308_01380_v', 'uav0000309_00000_v', 'uav0000315_00000_v', 'uav0000316_01288_v',
                  'uav0000323_01173_v', 'uav0000326_01035_v', 'uav0000329_04715_v', 'uav0000342_04692_v', 'uav0000352_05980_v',
                  'uav0000357_00920_v', 'uav0000360_00001_v', 'uav0000361_02323_v', 'uav0000363_00001_v', 'uav0000366_00001_v',
                  'uav0000013_00000_v']  # 56 sequences
    val_seqs = ['uav0000086_00000_v', 'uav0000117_02622_v', 'uav0000137_00458_v', 'uav0000182_00000_v', 'uav0000268_05773_v',
                'uav0000305_00000_v', 'uav0000339_00001_v']  # 7 sequences
    test_seqs = ['uav0000009_03358_v', 'uav0000073_00600_v', 'uav0000073_04464_v', 'uav0000077_00720_v', 'uav0000088_00290_v',
                 'uav0000119_02301_v', 'uav0000120_04775_v', 'uav0000161_00000_v', 'uav0000188_00000_v', 'uav0000201_00000_v',
                 'uav0000249_00001_v', 'uav0000249_02688_v', 'uav0000297_00000_v', 'uav0000297_02761_v', 'uav0000306_00230_v',
                 'uav0000355_00001_v', 'uav0000370_00001_v']  # 17 sequences

    # For Train sequences
    for seq in train_seqs:  # test_seqs
        seq_path = train_dir_sequences + '/' + seq
        # seq_path = test_dir_sequences + '/' + seq

        if path.isdir(seq_path):
            files = get_image_list(seq_path)
        else:
            files = [seq_path]
        files.sort()

        num_frames = len(files)

        print('===== Sequence ======== : ', seq)

        for frame, img_path in enumerate(files, 1):
            img_orig = cv2.imread(img_path)
            img = copy.deepcopy(img_orig)
            annotations = np.loadtxt(train_dir_annotations + '/' + seq + '.txt', delimiter=',')
            # annotations = np.loadtxt(test_dir_annotations + '/' + seq + '.txt', delimiter=',')
            annotations_frame = annotations[np.where(annotations[:, 0] == frame)]
            annotations_frame[:, 4:6] += annotations_frame[:, 2:4]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]

            text_scale = 2
            text_thickness = 2
            line_thickness = 3

            for i in range(annotations_frame.shape[0]):
                obj_id = int(annotations_frame[i, 1])
                #print('ID: ', obj_id)
                class_idx = int(annotations_frame[i, 7])
                if class_idx not in CATEGORIES.keys():
                    continue
                class_name = CATEGORIES[class_idx]
                bbox = annotations_frame[i, 2:6]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                crop = img_orig[y1:y2, x1:x2, :]  # Not to include the plotted lines to the cropped images

                ID_write = format(ID_COUNTER, '06d') + '_' + class_name
                color = get_color(obj_id)
                # id_text = '{}'.format(int(obj_id)) + ', ' + class_name
                # id_text = class_name
                id_text = '{}'.format(int(obj_id))
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
                cv2.putText(img, id_text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                            thickness=text_thickness)


            # Display tracks
            if args.display:
                cv2.imshow('Tracking', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            if frame % 20 == 0:
                logger.info('Processing frame {}/{})'.format(frame, num_frames))



# Execute from the interpreter
if __name__ == "__main__":
    main()

