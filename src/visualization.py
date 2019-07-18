import matplotlib.pyplot as plt
import pandas as pd
import json
from src.convenion import get_and_preprocess_data


def visualize_history(history_path='history/hiddenlayer 0-vector-baomoi.json'):
    with open(history_path, 'r') as file:
        history = json.load(file)
        print(history.keys())
        # summarize history for accuracy
        plt.plot(history['train_map'])
        plt.plot(history['val_map'])
        plt.plot(history['test_map'])

        plt.title('hiddenlayer 0-base mAP')
        plt.ylabel('mAP')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('hiddenlayer 0-base loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

def visualize_feature():
    data_df = pd.read_csv('data/similar1/raw/ez-moreinfo-strict.json', sep='\t', header=None, names=['org_q', 'related_q',
                                                                                             'label', 'elastic_similar',
                                                                                             'vector_similar'])
    data_df['color'] = 'red'
    data_df['color'][data_df['label'] == 0] = 'blue'

    x_red = data_df['elastic_similar'][data_df['label'] == 1].values
    y_red = data_df['vector_similar'][data_df['label'] == 1].values

    x_blue = data_df['elastic_similar'][data_df['label'] == 0].values
    y_blue = data_df['vector_similar'][data_df['label'] == 0].values
    # x = data_df['elastic_similar'].values
    # y = data_df['vector_similar'].values
    # color = data_df['color'].values

    plt.xlabel('elastic_similar')
    plt.ylabel('vector_similar')

    relevan = plt.scatter(x_red, y_red, color='red', s=0.5, alpha=0.5)
    inrelevan = plt.scatter(x_blue, y_blue, color='blue', s=0.5, alpha=0.5)

    # plt.legend((relevan, inrelevan),
    #            ('Relevan', 'Inrelevan'),
    #            scatterpoints=1,
    #            loc='upper left',
    #            ncol=1,
    #            bbox_to_anchor=(0.5,1.1))
    # plt.scatter(x, y, color=color)
    plt.show()


def count_positive():
    PATH_DATA_TRAIN = 'data/pool1/raw/train.txt'
    PATH_DATA_DEV = 'data/pool1/raw/dev.txt'
    PATH_DATA_TEST = 'data/test_data/raw/test.txt'
    train_data_df = get_and_preprocess_data(PATH_DATA_TRAIN, separator='\t')
    dev_data_df = get_and_preprocess_data(PATH_DATA_DEV, separator='\t')
    test_data_df = get_and_preprocess_data(PATH_DATA_TEST, separator='\t')

    train_data_df['len_org_q'] = [len(row['org_q'].split()) for index, row in train_data_df.iterrows()]
    train_data_df['len_related_q'] = [len(row['related_q'].split()) for index, row in train_data_df.iterrows()]

    dev_data_df['len_org_q'] = [len(row['org_q'].split()) for index, row in dev_data_df.iterrows()]
    dev_data_df['len_related_q'] = [len(row['related_q'].split()) for index, row in dev_data_df.iterrows()]

    test_data_df['len_org_q'] = [len(row['org_q'].split()) for index, row in test_data_df.iterrows()]
    test_data_df['len_related_q'] = [len(row['related_q'].split()) for index, row in test_data_df.iterrows()]

    test_data_df['len_related_q'].mean()


    len_positive_train = len(train_data_df[train_data_df['label'] == 1])
    len_positive_dev = len(dev_data_df[dev_data_df['label'] == 1])
    len_positive_test = len(test_data_df[test_data_df['label'] == 1])


visualize_history('concat-bilstm-0516-pool1-256units-64nodes.json')

