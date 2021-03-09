#%%

from lib.mnist_aug.mnist_augmenter import DataManager, MNISTAug


aug = MNISTAug()
dm = DataManager()
dm.load_test()

x, y = aug.get_augmented(dm.x_test, dm.y_test, 2)

print(x.shape, y)

[DataManager.plot_num(x[i], y[i]) for i in range(len(x))]


print(dm.x_test.shape)
print(dm.y_test.shape)

# --- aug

#%%

from lib.mnist_aug.mnist_augmenter import DataManager, MNISTAug

#%%

aug = MNISTAug()
dm = DataManager()
dm.load_test()

#%%

aug.max_objects = 5
aug.min_objects = 3

x, y = aug.get_augmented(dm.x_test, dm.y_test, 1, get_captions=True)

[print(yi["class"]) for yi in y[0]]

for i in range(len(x)):
    DataManager.plot_num(x[i], y[i])
    DataManager.plot_num(x[i])

#%%
