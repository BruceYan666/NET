from matplotlib import pyplot as plt

Epoch, ACC_TRAIN, ACC_VAL, Loss = [], [], [], []
filename1 = './train.txt'
filename2 = './val.txt'
with open(filename1, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        try:
            temp = line.split(',')
            epoch = temp[0].split('=')[1]
            acc = temp[1].split('=')[1]
            loss = temp[2].split('=')[1]
            Epoch.append(epoch)
            ACC_TRAIN.append(acc)
            Loss.append(loss)
        except:
            continue
with open(filename2, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        try:
            temp = line.split(',')
            acc = temp[1].split('=')[1]
            ACC_VAL.append(acc)
        except:
            continue

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
ax1.plot(Epoch, Loss, 'red', label='loss')
ax1.legend()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train-loss')

ax2 = fig.add_subplot(212)
ax2.plot(Epoch, ACC_TRAIN, 'red', label='acc_train')
ax2.plot(Epoch, ACC_VAL, 'blue', label='acc_val')
ax2.legend()
ax2.set_xlabel('epoch')
ax2.set_ylabel('acc')
plt.show()