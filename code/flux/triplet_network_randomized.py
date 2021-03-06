import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def triplets(n, d, train, test):
    '''
    Generate a random set of triplets. Returns the embedding X, a train set, a test set
    '''
    X = np.random.randn(n,d)
    triplets = []
    labels = []
    for t in range(train+test):
        q = random.sample(range(n), 3)
        if np.linalg.norm(X[q[0],:] - X[q[1],:]) < np.linalg.norm(X[q[0],:] - X[q[2],:]):
            labels.append([0,1])
        else:
            labels.append([1,0])
        triplets.append(q)
    return X, {'samples':triplets[:train], 'labels':labels[:train]}, {'samples':triplets[train:], 'labels':labels[train:]}

def get_distances(X,n):
    D = np.zeros(int(n*(n-1)/2))
    print('size of D', len(D))

    for i in range(n):
        for j in range(i+1,n):
            idx = fold_idx(i,j,n)
            print('idx', idx)
            D[int(idx)] = np.linalg.norm(X[i,:] - X[j,:])
            print(i,j)
    return D

def fold_idx(i,j,n):
    if i>j:
        s=j
        b=i
    else:
        s=i
        b=j
    return n*(n-1)/2 - (n-s)*(n-s+1)/2 + b-s

def all_triplets(n, d):
    '''
    Generate all triplets
    '''
    X = np.random.randn(n,d)
    triplets = []
    labels = []
    labels_new=[]
    for i in range(n):
        for j in range(n):
            for k in range(j):
                if i!=j and i!=k and j!=k:
                    q = [i , j, k]
                    if np.linalg.norm(X[q[0],:] - X[q[1],:]) < np.linalg.norm(X[q[0],:] - X[q[2],:]):
                        labels.append([0,1])
                    else:
                        labels.append([1,0])
                    triplets.append(q)
    return X, {'samples':triplets, 'labels':labels}, {'samples':triplets, 'labels':labels}

def procrustes(X, Y, scaling=True, reflection='best'):
    n = X.shape[0]; m = X.shape[1]
    ny = Y.shape[0]; my = Y.shape[1]
    muX = X.mean(0); muY = Y.mean(0)
    X0 = X - muX; Y0 = Y - muY
    ssX = (X0**2.).sum(); ssY = (Y0**2.).sum()
    normX = np.sqrt(ssX); normY = np.sqrt(ssY)
    X0 /= normX; Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    if reflection is not 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

def find_embedding(exp_num, n,decay_rate):
    exp = str(exp_num)
    decay = str(decay_rate)
    items = str(n)

    fname = items  + '/' + 'triplet_result_' + items + '_' + decay + '_' + exp  + '.txt'

    with open(fname, 'w') as file_handler:
        data = []
        d=2
        # Build the triples
        #X, train, test = triplets(n,d, train_size, test_size)
        X, train, test = all_triplets(n,d)

        print('length train', len(train['samples']))
        train_size = len(train['samples'])
        #data.append(['exp num', exp_num, 'num points', n, 'decay rate', decay_rate, 'dim', d, 'std_dev', 1000, 'learning_rate', .1])
        file_handler.write(str(exp_num) + ',' + str(n) + ',' + str(decay_rate) + ',' + str(d) + ',1000, .1 \n')

        # Build the Triplet Net
        # W - rows of W will be our learned embedding
        #W = tf.Variable(tf.random_normal([n, 2], 0., 1), name="weights")
        W = tf.Variable(tf.random_normal([n, d], 0., 1000), name="weights")
        head = tf.placeholder(tf.float32, [None, n])
        left = tf.placeholder(tf.float32, [None, n])
        right = tf.placeholder(tf.float32, [None, n])
        y = tf.placeholder(tf.float32, [None, 2])

        # Computes |W*e_i - W*e_j|^2
        dleft = tf.pow(tf.norm(tf.subtract(tf.matmul(head,W), tf.matmul(left, W)), axis=1), 2.)
        # Computes |W*e_i - W*e_k|^2
        dright = tf.pow(tf.norm(tf.subtract(tf.matmul(head,W), tf.matmul(right, W)), axis=1), 2.)

        # Computes the logisitic probabilities and the associated loss
        #p = tf.nn.softmax(tf.stack([dleft, dright],axis = 1))
        #loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(p), reduction_indices=[1]))

        # Hinge loss variant
        p = tf.stack([dleft-dright, dright-dleft], axis=1)

        loss = tf.reduce_mean(tf.maximum(0., 1.-tf.reduce_sum(y*p, reduction_indices=[1])))#tf.losses.hinge_loss(y, p)

        I = np.eye(n,n)
        head_data = [I[q[0], :] for q in train['samples']]
        left_data = [I[q[1], :] for q in train['samples']]
        right_data = [I[q[2], :] for q in train['samples']]

        test_head_data = [I[q[0], :] for q in test['samples']]
        test_left_data = [I[q[1], :] for q in test['samples']]
        test_right_data = [I[q[2], :] for q in test['samples']]

        #shuffle the data
        zipped = list(zip(head_data, left_data, right_data, test_head_data, test_left_data, test_right_data, test['labels'], train['labels']))
        random.shuffle(zipped)
        head_data, left_data, right_data, test_head_data, test_left_data, test_right_data, test['labels'], train['labels'] = zip(*zipped)

        # For debugging purposes
        #from tensorflow.python import debug as tf_debug
        sess = tf.InteractiveSession()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Setup the optimizer and pass in the loss function/all the data batched in sets of 100 for SGD
        global_step = tf.Variable(1.0, trainable=False)
        learning_rate = tf.train.exponential_decay(.1, global_step, len(head_data), decay_rate)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        var_grad = tf.gradients(loss, W)[0]

        # Compute training/test loss
        correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.global_variables_initializer().run()

        initials = sess.run([W, loss, y*p, 1.-tf.reduce_sum(y*p, reduction_indices=[1])],feed_dict={head: head_data,
                                                                  left: left_data,
                                                                  right: right_data,
                                                          y: train['labels']})
        Winitial = initials[0]
        delta = 100
        start = time.time()
        iteration = 0
        old_loss = 0

        for _ in range(2000):
            print('iter {}'.format(_))
            x = sess.run([accuracy, loss, var_grad, learning_rate],
                                  feed_dict={head: test_head_data,
                                             left: test_left_data,
                                             right: test_right_data,
                                             y: test['labels']})

            print('acuracy', x[0], 'loss', x[1], 'gradient', np.linalg.norm(x[2],ord='fro'), 'learning_rate', x[3])
            #data.append(['iteration number', iteration, 'acuracy', x[0],
            #'num constraints violated', (1-x[0])*train_size, 'loss', x[1], 'gradient',
            #np.linalg.norm(x[2],ord='fro'), 'learning_rate', x[3]])

            file_handler.write(str(iteration) + ',' + str(x[0]) + ',' + str((1-x[0])*train_size)
            + ',' + str(x[1]) + ',' + str(np.linalg.norm(x[2],ord='fro')) + ',' + str(x[3]) + '\n')

            iteration +=1

            if x[0] ==1 or old_loss == x[1] :
                break
            old_loss = x[1]

            for i in range(1, int(np.round(train_size/delta))):
                if len(head_data[delta*(i-1): delta*i]) > 0:
                    #np.random.rand(len(head_data[delta*(i-1): delta*i]), 1)
                    sess.run([train_step],
                              feed_dict={head: head_data[delta*(i-1): delta*i],
                                         left: left_data[delta*(i-1): delta*i],
                                         right: right_data[delta*(i-1): delta*i],
                                         y: train['labels'][delta*(i-1): delta*i]})
                else:
                    sess.run([train_step],
                              feed_dict={head: head_data[delta*(i-1): end],
                                         left: left_data[delta*(i-1): end],
                                         right: right_data[delta*(i-1): end],
                                         y: train['labels'][delta*(i-1): end]})

            #print learning_rate.eval()

        result_test = sess.run([accuracy, W, loss,p], feed_dict={head: test_head_data,
                                                         left: test_left_data,
                                                         right: test_right_data,
                                                         y: test['labels']})
        end = time.time()
        total_time=(end - start)

        print('Final loss', result_test[2])
        print('Accuracy on test set:', result_test[0], result_test[1])
        result_train = sess.run([accuracy, W], feed_dict={head: head_data,
                                                          left: left_data,
                                                          right: right_data,
                                                          y: train['labels']})
        print('Accuracy on train set:', result_train[0]) # result_train[1])
        min_gap = np.min(np.abs(result_test[3]))
        print('min gap', min_gap)

        #Procrustes and plot
        if d==2:
            W = result_train[1]
            _, Wpro, _ = procrustes(X, W)
            frob_error = np.linalg.norm(X-Wpro,'fro')

        #data.append('final_accuracy, ' + str(result_test[0]) + ',final_loss,' + str(result_test[2]))
        #data.append('total_time', total_time, 'min gap' , min_gap, 'frob_error', frob_error,'rel_frob_error',frob_error/np.linalg.norm(X,'fro'))

        file_handler.write(str(result_test[0]) + ',' + str(result_test[2]) + ',' + str(total_time)
         + ',' + str(min_gap)  + ',' + str(frob_error)  + ',' + str(frob_error/np.linalg.norm(X,'fro')) + '\n')



        # with open(fname, 'w') as file_handler:
        #     for item in data:
        #         file_handler.write("{}\n".format(item))
