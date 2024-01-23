# Program to test some specific things about the code
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# LABELS_MAT_PATH_test = "/home/santiagojn/disentangling-vae-master/data/custom/trial3/test-material-labels-lightness.npy"
# TEST_MAT = "/home/santiagojn/disentangling-vae-master/data/custom/trial3/test-material_packed.npy"
# NAMES_MAT = "/home/santiagojn/disentangling-vae-master/data/custom/trial3/test-material-names.npy"


# images = np.load(TEST_MAT)
# labels = np.load(LABELS_MAT_PATH_test).astype(np.float)
# names = np.load(NAMES_MAT)

# print(f'Images: {images.shape}')
# print(f'Labels: {labels.shape}')
# print(f'Names: {names.shape}')

# print(f'Name: {names[1000]}')
# print(f'Labels: {labels[1000]}')
# plt.imshow(images[1000])
# plt.show()


# TEST_ANALYTIC = "/home/santiagojn/disentangling-vae-master/data/custom/trial3/test_analytic_packed.npy"
# TEST_ANALYTIC_path = "/home/santiagojn/disentangling-vae-master/data/custom/trial3/test_analytic"

# images = np.load(TEST_ANALYTIC)
# names = sorted(os.listdir(TEST_ANALYTIC_path))

# print(f'Name: {names[291]}')
# plt.imshow(images[291])
# plt.show()

MASKED_TEST_PATH = "/home/santiagojn/lib_disentanglement/disentanglement_lib/evaluation/data/masked-serrano-test_packed.npy"
MASKED_TRAIN_PATH = "/home/santiagojn/lib_disentanglement/disentanglement_lib/evaluation/data/masked-serrano-train_packed.npy"
LABELS_GLOSS_TRAIN = "/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano-train-labels-glossiness.npy"
NAMES_PATH = "/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano-train"

MASKED_SERRANO = "/home/santiagojn/lib_disentanglement/disentanglement_lib/evaluation/data/masked-serrano_packed.npy"
# TEST_TRAIN_GLOSS_LABELS = "/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano-labels-lightness.npy"
# TEST_TRAIN_GLOSS_path = "/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano"

# images = np.load(MASKED_TRAIN_PATH)
# labels = np.load(LABELS_GLOSS_TRAIN)
# names = sorted(os.listdir(NAMES_PATH))

# print(f'images: {images.shape}')
# print(f'labels: {labels.shape}')
# print(f'names: {len(names)}')

# for idx in range(50):
#        print(f'{idx} - {names[idx]} - {labels[idx]}')

# idx = random.randint(0, 7000) 
# print(f'{idx} - Name: {names[idx]} --> labels: {labels[idx]}')
# plt.imshow(images[idx])
# plt.show()


TRAIN_PATH = "/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano-train"
indices = []
NAMES_PATH = "/home/santiagojn/disentangling-vae-master/data/serrano/samples_names.npy"
names = np.load(NAMES_PATH)
for train_name in sorted(os.listdir(TRAIN_PATH)):
       indices.append(np.where(names==train_name))

indices = (np.array(indices)).flatten()
print(indices[:10])
np.save("/home/santiagojn/disentangling-vae-master/data/serrano/train_indices", indices)

LABELS_GLOSS_TEST = "/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano-test-labels-glossiness.npy"
labels_test = np.load(LABELS_GLOSS_TEST)
# print(labels_test[:20])
for geometry in [1,5]:
       for illum in range(1,8):
              for gloss in range(1,8):
                     f = [illum, geometry, gloss]
                     possibles = np.where((labels_test==f).all(axis=1))
                     possibles = possibles[0].tolist()
                     if illum == 5 and geometry == 1:
                            print(f'Gloss {gloss} has {len(possibles)} -> {possibles}')
                     if len(possibles) == 0:
                            print(f'Factors {f} have no samples :(')


test_names =  sorted(os.listdir("/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano-test"))
whole_names = sorted(os.listdir("/home/santiagojn/disentangling-vae-master/data/serrano/masked-serrano"))
print(test_names[1042])
print("----------")

all_labels = np.load("/home/santiagojn/disentangling-vae-master/data/serrano/samples_labels.npy")
idcs = np.where((all_labels==[5,1,5]).all(axis=1))
idcs = idcs[0].tolist()
print(all_labels[4808,2])
all_labels[4808,2] = 4.0
print(all_labels[4808,2])
np.save("/home/santiagojn/disentangling-vae-master/data/serrano/samples_labels.npy", all_labels)
# print(whole_names[4709])
# idx = np.where((whole_names == test_names[1042])(axis=1))
# print(idx)