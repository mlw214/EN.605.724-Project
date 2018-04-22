from __future__ import division, print_function

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

class SGANBuilder(object):
    def __init__(self, generator, discriminator, input_shape, noise_shape):
        self.generator = generator
        self.discriminator = discriminator
        self.gd_stack = None
        self.input_shape = input_shape
        self.noise_shape = noise_shape
        self.discriminator_optimizer = Adam()
        self.generator_optimizer = Adam()
        self.discriminator_loss = ['binary_crossentropy', 'categorical_crossentropy']
        self.discriminator_loss_weights = [0.5, 0.5]
        self.generator_loss = 'binary_crossentropy'
        self.generator_loss_weights = None
        self.discriminator_metrics = ['accuracy']
        self.generator_metrics = None

    def build(self):
        self.discriminator.compile(
            loss=self.discriminator_loss,
            loss_weights=self.discriminator_loss_weights,
            optimizer=self.discriminator_optimizer,
            metrics=self.discriminator_metrics,
        )
        self.generator.compile(
            loss=self.generator_loss,
            loss_weights=self.generator_loss_weights,
            optimizer=self.generator_optimizer,
            metrics=self.generator_metrics,
        )

        noise = Input(shape=self.noise_shape)
        generated = self.generator(noise)

        self.discriminator.trainable = False

        valid, _ = self.discriminator([generated, generated])

        self.gd_stack = Model(noise, valid)
        self.gd_stack.compile(
            loss=self.generator_loss,
            loss_weights=self.generator_loss_weights,
            optimizer=self.generator_optimizer,
            metrics=self.generator_metrics,
        )

        return self.discriminator, self.generator, self.gd_stack

    def build_using_hook(self, build_hook):
        return build_hook(self)
    
    def set_discriminator_optimizer(self, opt):
        self.discriminator_optimizer = opt
        return self

    def set_generator_optimizer(self, opt):
        self.generator_optimizer = opt
        return self
    
    def set_discriminator_loss(self, loss):
        self.discriminator_loss = loss
        return self

    def set_generator_loss(self, loss):
        self.generator_loss = loss
        return self

    def set_discriminator_loss_weights(self, weights):
        self.discriminator_loss_weights = weights
        return self

    def set_generator_loss_weights(self, weights):
        self.generator_loss_weights = weights
        return self

    def set_discriminator_metrics(self, metrics):
        self.discriminator_metrics = metrics
        return self

    def set_generator_metrics(self, metrics):
        self.generator_metrics = metrics
        return self
