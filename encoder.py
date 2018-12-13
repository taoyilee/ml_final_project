import os
from keras.models import model_from_json
import configparser as cp
from preprocessing.dataset import SVHNDataset, ColorConverter
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    exp_dir = "experiments/12_12_142836_grayscale_cnn"
    with open(os.path.join(exp_dir, f"encoder.json"), "r") as f:
        encoder = model_from_json(f.read())  # type: Model
    encoder.load_weights(os.path.join(exp_dir, f"encoder_30.h5"))

    encoder.summary()

    config = cp.ConfigParser()
    config.read("config.ini")

    batch_size = config["general"].getint("batch_size")
    ae_model = config["general"].get("ae_model")
    color_mode = config["general"].get("color_mode")
    noise_ratio = config["general"].getfloat("noise_ratio")
    train_set = SVHNDataset.from_npy(config["general"].get("training_set"))
    converter = ColorConverter("grayscale")
    train_set_gray = converter.transform(train_set)
    input_image = train_set_gray.images[0][np.newaxis, :] / 255.0
    print(input_image.shape)

    all_images = train_set_gray.images / 255.0

    #features_in_32 = encoder.predict(input_image)
    #print(features_in_32.shape)
    #print(features_in_32)

    all_features_in_64 = encoder.predict(all_images, batch_size=512)

    print(all_features_in_64.shape)
    print(all_features_in_64[0])

    all_features_in_64 = all_features_in_64.reshape(71791, 64)

    print(all_features_in_64.shape)
    print(all_features_in_64[0])


    np.save("all_features_64", all_features_in_64)

    print(all_features_in_64[1])

    print("SHAPE",train_set.labels.shape[0])

    colors = ['b','g','r','c','m','y','k','w', 'aqua', 'olivedrab']
    feature_dict = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}}

    for f in range(all_features_in_64.shape[1]):
        for i in range(train_set.labels.shape[0]):
            feature = round(all_features_in_64[:, f][i],1)

            if feature in feature_dict[train_set.labels[i]]:
                feature_dict[train_set.labels[i]][feature] += 1
            else:
                feature_dict[train_set.labels[i]][feature] = 1

        ys = []

        for i in range(1,11):
            for key in feature_dict[i]:
                ys.append(feature_dict[i][key])

            plt.bar(feature_dict[i].keys(), ys, color=colors[i - 1])
            #plt.plot(feature_dict[i].keys(), ys, color=colors[i - 1], marker='o', linestyle=''))
            print(ys)
            ys = []

        labels = set(train_set.labels)

        plt.xlabel("Values of the feature")
        plt.ylabel("Frequency")

        plt.savefig(f"hists/one_feature_64_color_coded_{f}.png")

    plt.figure(2)
    plt.hist(all_features_in_64[:, 0])
    plt.savefig(f"hists/one_feature_64_hist.png")

    # Make a histogram plot for each feature with each histogram
    # having a bar by the label of the data point, color coded
    # each i is referring to an image
    # each j is referring to a feature
    """for i in range(all_features_in_64.shape[0]):
        for j in range(all_features_in_64.shape[1]):"""

    """print(train_set.labels.shape)
    print(all_features_in_64[:, 0].shape)

    features_labels = np.stack((train_set.labels, all_features_in_64[:, 0]), axis=1)

    print("LABELS AND FEATURES ", features_labels.shape)

    df = pd.DataFrame(features_labels, columns=['', 'values'])
    print(df)"""



    """sns_plot = sns.barplot(x="labels", y="values", data=df)
    fig = sns_plot.get_figure()
    fig.savefig(f"hists/one_feature_64_hist.png")"""

    print("ONE FEATURE?", all_features_in_64[:, 0])

    #plt.hist(train_set.labels)
    #plt.savefig(f"hists/all_labels_64_hist.png")

    """pca = PCA(n_components=5)
    image_pc = pca.fit_transform(all_features_in_64)
    print(image_pc.shape)

    df = pd.DataFrame(columns=["pc0", "pc1", "label"])
    df["label"] = train_set.labels.flatten()
    df["pc0"] = image_pc[:, 0]
    df["pc1"] = image_pc[:, 1]

    for i in range(4):
        for j in range(i + 1, 5):
            plt.figure(figsize=(8, 8))
            for k in np.unique(train_set.labels):
                plt.scatter(image_pc[train_set.labels.flatten() == k, i], image_pc[train_set.labels.flatten() == k, j],
                            cmap='jet', s=1, label=f"{k}")
            plt.xlabel(f"Principal Component {i}")
            plt.ylabel(f"Principal Component {j}")
            plt.grid()
            plt.legend()
            plt.savefig(f"images/pca_{train_set.name}_{i}_{j}.png")
            plt.close()"""


