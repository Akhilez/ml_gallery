from lib.mnist_aug.mnist_augmenter import DataManager, MNISTAug

aug = MNISTAug()
dm = DataManager()
dm.load_test()


x, y = aug.get_augmented(dm.x_test, dm.y_test, 1)

DataManager.plot_num(x[0], y[0])
