import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def visualize_history(history_path='history/bi-lstm-vector-baomoi.json'):
    with open(history_path, 'r') as file:
        history = json.load(file)
        print(history.keys())
        # summarize history for accuracy
        plt.plot(history['train_map'])
        plt.plot(history['val_map'])
        plt.plot(history['test_map'])

        plt.title('bi-lstm-base mAP')
        plt.ylabel('mAP')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('bi-lstm-base loss')
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


visualize_history('siamese-lstm-pool1-06032357.json')