from lib.mnist_aug.mnist_augmenter import DataManager, MNISTAug


aug = MNISTAug()
dm = DataManager()
dm.load_test()

x, y = aug.get_augmented(dm.x_test, dm.y_test, 2)

print(x.shape, y)

# [DataManager.plot_num(xi) for xi in x]


print(dm.x_test.shape)
print(dm.y_test.shape)

