import tensorflow as tf
import numpy as np
import os
import time

from keras.datasets.cifar10 import load_data

from ares import BIM, CrossEntropyLoss
from ares.model.loader import load_model_from_path
import matplotlib.pyplot as plt

def plot_image(image,path): 
    plt.figure(figsize=(40, 40))

    for i in range(100) :
        plt.subplot(10,10,i+1)
        plt.imshow(image[i])
        plt.axis('off')
        plt.tight_layout()

    #plt.show()
    
    plt.savefig(path, bbox_inches='tight') 

batch_size = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/cifar10/resnet56.py')
rs_model = load_model_from_path(model_path)
model = rs_model.load(session)

_, (xs_test, ys_test) = load_data()
xs_test = (xs_test / 255.0) * (model.x_max - model.x_min) + model.x_min
ys_test = ys_test.reshape(len(ys_test))

xs_ph = tf.placeholder(model.x_dtype, shape=(batch_size, *model.x_shape))
lgs, lbs = model.logits_and_labels(xs_ph)


def iteration_callback(xs, xs_adv):
    delta = tf.abs(xs_adv - xs)
    return tf.reduce_max(tf.reshape(delta, (xs.shape[0], -1)), axis=1)


loss = CrossEntropyLoss(model)
attack = BIM(
    model=model,
    batch_size=batch_size,
    loss=loss,
    goal='ut',
    distance_metric='l_inf',
    session=session,
    iteration_callback=iteration_callback,
)
attack.config(
    iteration=10,
    magnitude=8.0 / 255.0,
    alpha=1.0 / 255.0,
)


i = 0

start = time.time()
for lo in range(batch_size, 5*batch_size, batch_size):
    xs = xs_test[lo - batch_size:lo]
    ys = ys_test[lo - batch_size:lo]

    #path = "BIM/attack_1/original_"+str(i)+".png"
    #plot_image(xs,"BIM/attack_1/original_"+str(i)+".png")
    count = 1
    try:
        g = attack.batch_attack(xs, ys=ys)
        while True:
            if count < 11:
                #print("iter ", count)
                count += 1
            #print(next(g))
            next(g)
    except StopIteration as e:
        xs_adv = e.value

    #plot_image(xs_adv,"BIM/attack_1/adv"+str(i)+".png")
    #i+=1

    lbs_pred = session.run(lbs, feed_dict={xs_ph: xs})
    lbs_adv = session.run(lbs, feed_dict={xs_ph: xs_adv})

    print(
        np.equal(ys, lbs_pred).astype(np.float).mean(),
        np.equal(ys, lbs_adv).astype(np.float).mean()
    )
end = time.time()
print("time:", end - start,"s")
print("\nchanging attack config\n")

eps = np.concatenate((np.ones(50) * 4.0 / 255.0, np.ones(50) * 8.0 / 255.0))
attack.config(
    iteration=10,
    magnitude=eps,
    alpha=eps / 8,
)
start = time.time()

i = 0
for lo in range(batch_size, 5*batch_size, batch_size):
    xs = xs_test[lo - batch_size:lo]
    ys = ys_test[lo - batch_size:lo]
    
    #plot_image(xs,"BIM/attack_2/original_"+str(i)+".png")
    count = 1
    try:
        g = attack.batch_attack(xs, ys=ys)
        while True:
            if count < 11:
                #print("iter ", count)
                count += 1
            #print(next(g))
            next(g)
    except StopIteration as exp:
        xs_adv = exp.value

    #plot_image(xs_adv,"BIM/attack_2/adv_concat(4)_"+str(i)+".png")
    i+=1

    lbs_pred = session.run(lbs, feed_dict={xs_ph: xs})
    lbs_adv = session.run(lbs, feed_dict={xs_ph: xs_adv})

    print(
        np.equal(ys, lbs_pred).astype(np.float).mean(),
        np.equal(ys, lbs_adv).astype(np.float).mean()
    )
end = time.time()
print("time:", end - start,"s")
