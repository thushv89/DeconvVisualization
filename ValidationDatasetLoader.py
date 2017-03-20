

def load_cifar_dataset(dataset_info):
    global valid_dataset

    dataset_size = dataset_info['dataset_size']

    valid_dataset_filename = '..' + os.sep + 'data' + os.sep + 'cifar-10.pickle'

    (train_dataset, train_labels), \
    (valid_dataset, valid_labels), \
    (test_dataset, test_labels) = load_data.reformat_data_cifar10(valid_dataset_filename)

    del train_dataset,train_labels,test_dataset,test_labels
    del v_labels

    valid_dataset = v_dataset


def load_imagenet_dataset(dataset_info):

    valid_dataset_fname = 'imagenet_small' + os.sep + 'imagenet_small_valid_dataset'
    valid_label_fname = 'imagenet_small' + os.sep + 'imagenet_small_valid_labels'

    fp1 = np.memmap(valid_dataset_fname, dtype=np.float32, mode='r',
                    offset=np.dtype('float32').itemsize * 0,
                    shape=(valid_size, image_size, image_size, num_channels))
    fp2 = np.memmap(valid_label_fname, dtype=np.int32, mode='r',
                    offset=np.dtype('int32').itemsize * 0, shape=(valid_size, 1))
    v_dataset = fp1[:, :, :, :]
    v_labels = fp2[:]

    v_dataset, v_labels = load_data.reformat_data_imagenet_with_memmap_array(
        v_dataset, v_labels, silent=True
    )

    del v_labels
    valid_dataset = v_dataset