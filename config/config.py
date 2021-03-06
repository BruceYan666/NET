PARA = dict(
    train=dict(
        EPOCH=135,
        BATCH_SIZE=128,
        LR=0.1,
        momentum=0.9,
        wd=5e-4,
        num_workers=2,
        device_ids=[0]
    ),
    test=dict(
        BATCH_SIZE=100,
        NUM_CLASSES = 10
    ),
    data=dict(
        validation_rate=0.05,
        original_trainset_path='../../DATASET/cifar-10/cifar-10-batches-py/train_batch_path',
        original_testset_path='../../DATASET/cifar-10/cifar-10-batches-py/test_batch_path/test_batch',
        after_trainset_path='../../DATASET/cifar-10/trainset/',
        after_valset_path='../../DATASET/cifar-10/valset/',
        after_testset_path='../../DATASET/cifar-10/testset/',
        train_data_txt='../DATASET/cifar-10/train.txt',
        val_data_txt='../DATASET/cifar-10/val.txt',
        test_data_txt='../DATASET/cifar-10/test.txt'
    )
)