import pandas as pd
import os
import copy

METADATA_DIR = 'metadata'

class DataSet(object):

    def __init__(self, database_dir: str, file_ext: str = '.jpg'):
        '''
        :param database_dir: Directory of dataset
        :param file_ext: File extension of pictures in the dataset

        Create a .csv file and load in DataFrame
        .dataset_dir - Directory of dataset
        .data - DataFrame with paths and classes
        .labels - set with classes
        '''
        self.dataset_dir = database_dir
        self.file_ext = file_ext
        self.file_name_csv = os.path.join(METADATA_DIR, os.path.basename(database_dir) + '.csv')
        self.create_csv()
        self.data = pd.read_csv(self.file_name_csv)
        self.labels = set(self.data["class_image"])
        os.remove(self.file_name_csv)

    def create_csv(self):
        '''
        Creating a .csv file to load in pandas DataFrame
        '''
        if not os.path.exists(METADATA_DIR):
            os.mkdir(METADATA_DIR)
        if os.path.exists(self.file_name_csv):
            return
        with open(self.file_name_csv, 'w', encoding='UTF-8') as file_temp:
            file_temp.write("image_path,class_image\n")
            for root, _, files in os.walk(self.dataset_dir, topdown=False):
                class_image = root.split(os.path.sep)[-1]
                for name in files:
                    if not name.endswith(self.file_ext):
                        continue
                    image_path = os.path.join(root, name)
                    file_temp.write("{},{}\n".format(image_path, class_image))

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data

    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def split(self, ratio: float, pos: float):
        another = copy.deepcopy(self)
        yet_another = copy.deepcopy(self)

        split_point_1 = int(ratio * pos * len(self.data))
        split_point_2 = int(ratio * (pos + 1) * len(self.data))

        another.data = self.data[0:split_point_1].append(self.data[split_point_2:]).reset_index()
        yet_another.data = self.data[split_point_1:split_point_2].reset_index()

        return another, yet_another
