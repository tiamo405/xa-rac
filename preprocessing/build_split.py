import argparse
import os


def build_split_list(raw_path, mode):
    raw_path = os.path.join(raw_path, '{}list01.txt'.format(mode))
    print('{} analysis begin'.format(raw_path))
    with open(raw_path, 'r') as fin:
        lines = fin.readlines()
    fin.close()

    with open('{}list.txt'.format(mode), 'w') as fout:
        for i, line in enumerate(lines):
            line = line.strip()  # 'class_name/video_name'
            label_dir = 'labels' + '/' + line  # 'data/ucf24/labels/class_name/video_name'
            if not os.path.isdir(label_dir):
                continue
            txt_list = os.listdir(label_dir)
            txt_list.sort()
            for txt_item in txt_list:
                filename = label_dir + '/' + txt_item
                fout.write(filename + '\n')
            if i % 200 == 0:
                print('{} videos parsed'.format(i))
    fout.close()
    print('{} analysis done'.format(raw_path))


def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('--raw_path', type=str, default='./splitfiles')
    parser.add_argument('--out_list_path', type=str, default='./')
    parser.add_argument('--shuffle', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    build_split_list(args.raw_path, 'train')
    build_split_list(args.raw_path, 'test')