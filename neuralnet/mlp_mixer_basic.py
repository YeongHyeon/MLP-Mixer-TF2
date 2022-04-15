import os
import numpy as np
import tensorflow as tf
import source.utils as utils
import whiteboxlayer.layers as wbl
import whiteboxlayer.extensions.convolution as wblc

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.patch_size = kwargs['patch_size']
        self.d_emb = kwargs['d_emb']
        self.d_mix_t = kwargs['d_mix_t']
        self.d_mix_c = kwargs['d_mix_c']
        self.depth = kwargs['depth']

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.variables = {}

        self.__model = Neuralnet(dim_h=self.dim_h, dim_w=self.dim_w, dim_c=self.dim_c, num_class=self.num_class, \
            patch_size=self.patch_size, d_emb=self.d_emb, d_mix_t=self.d_mix_t, d_mix_c=self.d_mix_c, depth=self.depth)
        self.__model.forward(x=tf.zeros((1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32), verbose=True)

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

    @tf.autograph.experimental.do_not_convert
    def step(self, minibatch, iteration=0, training=False):

        x, y = minibatch['x'], minibatch['y']

        with tf.GradientTape() as tape:
            logits = self.__model.forward(x=x, verbose=False)
            smce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.math.reduce_mean(smce)

        score = tf.nn.softmax(logits)
        pred = tf.argmax(score, 1)
        correct_pred = tf.equal(pred, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if(training):
            gradients = tape.gradient(loss, self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/loss_mean' %(self.__model.who_am_i), loss, step=iteration)

        return loss, accuracy, score


    def save_params(self, model='base'):

        vars_to_save = self.__model.layer.parameters.copy()
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
        ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = "Mixer"

        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.patch_size = kwargs['patch_size']
        self.d_emb = kwargs['d_emb']
        self.d_mix_t = kwargs['d_mix_t']
        self.d_mix_c = kwargs['d_mix_c']
        self.depth = kwargs['depth']

        self.layer = wbl.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, verbose=False):

        emb = self.__embedding(x=x, name='embedding', verbose=verbose)
        mix = self.__mixer(x=emb, name='mixer', verbose=verbose)
        y_hat = self.__classifier(x=mix, name='clssifier', verbose=verbose)
        y_hat = tf.math.add(y_hat, 0, name='y_hat')

        return y_hat

    def __embedding(self, x, name='embedding', verbose=False):

        x = wblc.conv2d(layer=self.layer, x=x, stride=self.patch_size, \
            filter_size=[self.patch_size, self.patch_size, self.dim_c, self.d_emb], \
            activation=None, name='embedding', verbose=verbose)
        [n, h, w, c] = x.shape
        x = tf.reshape(x, [n, h*w, c])

        return x

    def __mixer(self, x, name='mixer', verbose=False):

        for depth in range(self.depth):
            x = self.__mixer_block(x=x, name='mixer_%d' %(depth), verbose=verbose)

        return x

    def __classifier(self, x, name='clssifier', verbose=False):

        gap = tf.math.reduce_mean(x, axis=-1)
        x = self.layer.fully_connected(x=gap, c_out=self.num_class, \
            batch_norm=False, activation=None, name="%s" %(name), verbose=verbose)

        return x

    def __mixer_block(self, x, name='mixer', verbose=False):

        # token mixing
        x_laynorm_token = self.layer.layer_normalization(x, trainable=True, \
            name='%s_laynorm_0' %(name), verbose=verbose)
        x_laynorm_t = tf.transpose(x_laynorm_token, (0, 2, 1))
        x_mixed_token = self.__mlp_block(x=x_laynorm_t, c_out=self.d_mix_t, \
            name='%s_mlp_token' %(name), verbose=verbose)
        x_mixed_token = tf.transpose(x_mixed_token, (0, 2, 1))
        x = x + x_mixed_token # skip connection

        # channel mixing
        x_laynorm_chennel = self.layer.layer_normalization(x, trainable=True, \
            name='%s_laynorm_1' %(name), verbose=verbose)
        x_mixed_channel = self.__mlp_block(x=x_laynorm_chennel, c_out=self.d_mix_c, \
            name='%s_mlp_channel' %(name), verbose=verbose)

        return x + x_mixed_channel # skip connection

    def __mlp_block(self, x, c_out, name='mixer', verbose=False):

        [n, h, w] = x.shape
        x = self.layer.fully_connected(x=x, c_out=c_out, \
            batch_norm=False, activation=None, name="%s_0" %(name), verbose=verbose)
        x = tf.nn.gelu(x)
        x = self.layer.fully_connected(x=x, c_out=w, \
            batch_norm=False, activation=None, name="%s_1" %(name), verbose=verbose)

        return x
