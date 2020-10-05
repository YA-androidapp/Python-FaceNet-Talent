# Copyright (c) 2020 YA-androidapp(https://github.com/YA-androidapp) All rights reserved.
# pip install matplotlib numpy Pillow sklearn tensorflow

from facenet.src import facenet
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# facenet/src/facenet.py のimport文も併せて修正
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class FaceEmbedding(object):

    def __init__(self, model_path):
        facenet.load_model(model_path)

        self.input_image_size = 160
        self.sess = tf.Session()
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph(
        ).get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def __del__(self):
        self.sess.close()

    def load_image(self, image_path, width, height, mode):
        image = Image.open(image_path).resize([width, height], Image.BILINEAR)
        return np.array(image.convert(mode))

    def face_embeddings(self, image_path):
        image = self.load_image(
            image_path, self.input_image_size, self.input_image_size, 'RGB')
        prewhiten = facenet.prewhiten(image)
        feed_dict = {self.images_placeholder: prewhiten.reshape(
            -1, prewhiten.shape[0], prewhiten.shape[1], prewhiten.shape[2]),
            self.phase_train_placeholder: False}
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embeddings


if __name__ == "__main__":
    FACE_MEDEL_PATH = './20180402-114759/20180402-114759.pb'
    face_embedding = FaceEmbedding(FACE_MEDEL_PATH)

    fp = FontProperties(fname='C:\\WINDOWS\\Fonts\\YuGothM.ttc', size=10)

    types = ['jpg', 'png', 'gif']
    faces_image_paths = []
    for ext in types:
        paths = os.path.join(
            './images', '*_*_*-*-*-*.{}'.format(ext)
        )
        faces_image_paths.extend(glob.glob(paths))

    features = np.array(
        [face_embedding.face_embeddings(f)[0]
            for f in faces_image_paths]
    )

    basenames = [
        ((os.path.splitext(os.path.basename(f))[0]).split('-'))[0]
        for f in faces_image_paths]
    print(len(basenames))

    print(features.shape)
    print(features.reshape(*features.shape))

    # (9704, 512)
    # [[ 0.04433454 -0.01964697 -0.00387371 ... -0.0371541   0.0486134   0.01375637]
    #  [ 0.0650957   0.00260274  0.04348386 ...  0.03959348  0.06707621  0.05216877]
    #  [ 0.00080608 -0.0144004  -0.08873469 ... -0.03010601  0.0475108  -0.02686549]
    #  ...
    #  [-0.00625158  0.02463781 -0.08126526 ...  0.03991456  0.043775    0.05725215]
    #  [ 0.02276824 -0.01531809 -0.07569425 ... -0.03630852  0.03150427  0.04206658]
    #  [ 0.02412252  0.01870904 -0.02422359 ... -0.01704034 -0.01292099  0.01838743]]

    # ##########

    pca = PCA(n_components=2)
    pca.fit(features)
    reduced = pca.fit_transform(features)
    print(reduced.shape)
    print(reduced.reshape(*reduced.shape))

    # (9704, 2)
    # [[ 0.00566315  0.08859307]
    #  [ 0.02169254 -0.12193641]
    #  [-0.36523288  0.01194304]
    #  ...
    #  [ 0.26168096 -0.21410468]
    #  [-0.07790159  0.19103085]
    #  [ 0.3117506  -0.22082329]]

    # ##########

    K = 4
    kmeans = KMeans(n_clusters=K).fit(reduced)
    pred_label = kmeans.predict(reduced)
    print(len(pred_label))
    print(pred_label)

    x = reduced[:, 0]
    y = reduced[:, 1]
    plt.scatter(x, y, c=pred_label)

    for (i, j, k) in zip(x, y, basenames):
        plt.annotate(k, xy=(i, j), fontproperties=fp)
    plt.title("散布図", fontproperties=fp)
    plt.colorbar()
    plt.show()

    # #####

    plt.rcParams["font.family"] = "Yu Gothic"
    result = linkage(
        pd.DataFrame(features.reshape(*features.shape)),
        # metric='braycurtis',
        # metric = 'canberra',
        # metric = 'chebyshev',
        # metric = 'cityblock',
        # metric = 'correlation',
        # metric = 'cosine',
        metric='euclidean',
        # metric = 'hamming',
        # metric = 'jaccard',
        # method= 'single'
        method='average'
        # method= 'complete'
        # method='weighted'
    )
    print(result)
    dendrogram(result, labels=basenames, leaf_rotation=30)
    plt.title("デンドログラム")
    plt.ylabel("閾値")
    plt.subplots_adjust(bottom=0.4)
    plt.show()

    for cluster_count in range(2, K+1):
        clusters = fcluster(result, t=cluster_count, criterion='maxclust')
        for i, c in enumerate(clusters):
            print('\t{}\t{}\t{}'.format(cluster_count, i, c))
