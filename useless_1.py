from ops import *
from utils import *
from glob import glob
import time
import tflib.data_cat2lion
import tflib as lib
from tensorflow.contrib.data import batch_and_drop_remainder
from scipy.misc import imsave as IMSAVE
import pdb
import os

class MUNIT(object) :
    def __init__(self, sess, config):
        self.model_name = 'MUNIT'
        self.sess = sess
        self.phase = config["phase"]
        self.checkpoint_dir = config['checkpoint_dir']
        self.result_dir = config['result_dir']
        self.log_dir = config['log_dir']
        self.sample_dir = config['sample_dir']
        self.dataset_name = config['dataset']
        self.prefix = config[ 'prefix' ]
        self.augment_flag = config['augment_flag']

        self.epoch = config['epoch']
        self.iteration = config['iteration']

        self.gan_type = config['gan_type']

        self.batch_size = config['batch_size']
        self.print_freq = config['print_freq']
        self.save_freq = config['save_freq']
        self.img_freq = config['img_freq']
        self.num_style = config['num_style'] # for test
        self.guide_img = config['guide_img']
        self.direction = config['direction']

        self.img_h = config['img_h']
        self.img_w = config['img_w']
        self.img_ch = config['img_ch']
        self.anno_cl = config['anno_cl']
        self.num_workers = config["num_workers"]

        self.init_lr = config['lr']
        self.ch = config['ch']

        """ Weight """
        self.gan_w = config['gan_w']
        self.recon_x_w = config['recon_x_w']
        self.recon_s_w = config['recon_s_w']
        self.recon_c_w = config['recon_c_w']
        self.recon_k_w = config['recon_k_w']
        self.recon_x_cyc_w = config['recon_x_cyc_w']
        self.recon_x_twin_w = config['recon_x_twin_w']
        self.vgg_w = config['vgg_w']
        self.vgg_layer_names = config["vgg_layer_names"].split(",")
        self.vgg_weight = config['vgg_weight_file']

        """ Generator """
        self.n_res = config['n_res']
        self.mlp_dim = config['mlp_dim']

        self.n_downsample = config['n_sample']
        self.n_upsample = config['n_sample']
        self.style_dim = config['style_dim']


        """ Discriminator """
        self.n_dis = config['n_dis']
        self.n_scale = config['n_scale']


        self.data_path = config['data_path']
        self.sample_dir = os.path.join(config['sample_dir'], self.model_dir)
        check_folder(self.sample_dir)

        #self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        #self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        #self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# model directory : ", self.model_dir)
        #print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)
        print("# style in test phase : ", self.num_style)
        print("# VGG16 layer names : ", self.vgg_layer_names)
        print("# VGG16 weight file : ", self.vgg_weight)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# Style dimension : ", self.style_dim)
        print("# MLP dimension : ", self.mlp_dim)
        print("# Down sample : ", self.n_downsample)
        print("# Up sample : ", self.n_upsample)

        print()

        print("##### Discriminator #####")
        print("# Discriminator layer : ", self.n_dis)
        print("# Multi-scale Dis : ", self.n_scale)

    ##################################################################################
    # Encoder and Decoders
    ##################################################################################

    def Style_Encoder(self, x, reuse=False, scope='style_encoder'):
        # IN removes the original feature mean and variance that represent important style information
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = relu(x)

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i+1))
                x = relu(x)

                channel = channel * 2

            for i in range(self.n_downsample - 2) :
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='down_conv_'+str(i))
                x = relu(x)

            x = adaptive_avg_pooling(x) # global average pooling
            x = conv(x, self.style_dim, kernel=1, stride=1, scope='SE_logit')

            return x

    def Content_Encoder(self, x, key_point_channel=256, reuse=False, scope='content_encoder'):
        channel = self.ch

        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_0')
            x = relu(x)

            for i in range(self.n_downsample) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i+1))
                x = instance_norm(x, scope='ins_'+str(i+1))
                x = relu(x)
                factor = 2**(i + 1) 
                channel = channel * 2

            # adapt channel to one required in resblock block 
            channel_add_key = channel + key_point_channel
            x = conv(x, channel_add_key, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv_'+str(self.n_downsample + 1))
            x = instance_norm(x, scope='ins_'+str(self.n_downsample + 1))
            x = relu(x)

            x_atten = x
            for i in range(self.n_res) :
                x_atten = resblock(x_atten, channel_add_key, scope='atten_resblock_'+str(i))
            x_atten = relu(x_atten)
            x_atten = conv(x_atten, 1, kernel=7, stride=1, pad=3, pad_type='reflect', scope='atten_conv_'+str(self.n_res))
            x_atten = tf.sigmoid(x_atten)

            for i in range(self.n_res) :
                x = resblock(x, channel_add_key, scope='resblock_'+str(i))

            return x[:, :, :, :channel], x[:, :, :, channel:], x_atten 

    def generator(self, contents, style, label, attention, reuse=False, scope="decoder"):
        channel = 2*self.mlp_dim
        with tf.variable_scope(scope, reuse=reuse) :
            mu, sigma = self.MLP(style, reuse)
            x = contents
            x = tf.concat([x,label],3)
            x = tf.multiply(x, attention)
            for i in range(self.n_res) :
                x = adaptive_resblock(x, channel, mu, sigma, scope='adaptive_resblock'+str(i))

            #  inverse to encoder  

            for i in range(self.n_upsample) :
                # # IN removes the original feature mean and variance that represent important style information
                factor = 2**(self.n_upsample -i - 1)
                x = up_sample(x, scale_factor=2)
                #x = tf.concat([x,tf.image.resize_bilinear(label,(self.img_h/factor,self.img_w/factor), align_corners=False)],3)
                x = conv(x, channel//2, kernel=5, stride=1, pad=2, pad_type='reflect', scope='conv_'+str(i))
                x = layer_norm(x, scope='layer_norm_'+str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x

    def MLP(self, style, reuse=False, scope='MLP'):
        channel = 2*self.mlp_dim
        with tf.variable_scope(scope, reuse=reuse) :
            x = linear(style, channel, scope='linear_0')
            x = relu(x)

            x = linear(x, channel, scope='linear_1')
            x = relu(x)

            mu = linear(x, channel, scope='mu')
            sigma = linear(x, channel, scope='sigma')

            mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
            sigma = tf.reshape(sigma, shape=[-1, 1, 1, channel])

            return mu, sigma

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse) :
            for scale in range(self.n_scale) :
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.2)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', scope='ms_' + str(scale) +'conv_' + str(i))
                    x = lrelu(x, 0.2)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, scope='ms_' + str(scale) + 'D_logit')
                D_logit.append(x)

                x_init = down_sample(x_init)

            return D_logit

    ##################################################################################
    # Model
    ##################################################################################

    def Encoder_A(self, x_A, reuse=False):
        style_A = self.Style_Encoder(x_A, reuse=reuse, scope='style_encoder_A')
        content_A, keypoint_A, attention_A = self.Content_Encoder(x_A, reuse=reuse, scope='content_encoder_A')

        return content_A, style_A, keypoint_A, attention_A 

    def Encoder_B(self, x_B, reuse=False):
        style_B = self.Style_Encoder(x_B, reuse=reuse, scope='style_encoder_B')
        content_B, keypoint_B, attention_B = self.Content_Encoder(x_B, reuse=reuse, scope='content_encoder_B')

        return content_B, style_B, keypoint_B, attention_B

    def Decoder_A(self, content_B, style_A, label, attention, reuse=False):
        x_ba = self.generator(contents=content_B, style=style_A, label = label, attention = attention, reuse=reuse, scope='decoder_A')

        return x_ba

    def Decoder_B(self, content_A, style_B, label, attention, reuse=False):
        x_ab = self.generator(contents=content_A, style=style_B, label = label, attention = attention, reuse=reuse, scope='decoder_B')

        return x_ab

    def discriminate_real(self, x_A, x_B):
        real_A_logit = self.discriminator(x_A, scope="discriminator_A")
        real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
        fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        lr_scal = tf.summary.scalar( "Learning_Rate", self.lr )
        self.lr_sum = tf.summary.merge( [ lr_scal ] )

        """ Input Image"""
       # Image_Data_Class = ImageData(self.img_h, self.img_w, self.img_ch, self.augment_flag)

       # trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
       # trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)

       # trainA = trainA.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=self.num_workers).apply(batch_and_drop_remainder(self.batch_size)).repeat()
       # trainB = trainB.prefetch(self.batch_size).shuffle(self.dataset_num).map(Image_Data_Class.image_processing, num_parallel_calls=self.num_workers).apply(batch_and_drop_remainder(self.batch_size)).repeat()

       # trainA_iterator = trainA.make_one_shot_iterator()
       # trainB_iterator = trainB.make_one_shot_iterator()
       # 
        """ Get image batch"""
        self.domain_A = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_h, self.img_w, self.img_ch], name='domain_a')
        self.domain_B = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_h, self.img_w, self.img_ch], name='domain_b')
        self.domain_A_twin = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_h, self.img_w, self.img_ch], name='domain_a_twin')
        self.domain_B_twin = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_h, self.img_w, self.img_ch], name='domain_b_twin')

        self.domain_A_key_point = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_h, self.img_w], name='key_point_a') 
        self.domain_B_key_point = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_h, self.img_w], name='key_point_b') 
        self.domain_A_key_point_one_hot = preprocess_annotation(self.domain_A_key_point, self.anno_cl)
        self.domain_B_key_point_one_hot = preprocess_annotation(self.domain_B_key_point, self.anno_cl)

        self.domain_A_twin_key_point = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_h, self.img_w], name='key_point_a_twin') 
        self.domain_B_twin_key_point = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_h, self.img_w], name='key_point_b_twin') 
        self.domain_A_twin_key_point_one_hot = preprocess_annotation(self.domain_A_twin_key_point, self.anno_cl)
        self.domain_B_twin_key_point_one_hot = preprocess_annotation(self.domain_B_twin_key_point, self.anno_cl)

        """ Define Encoder, Generator, Discriminator """
        self.style_a = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_a')
        self.style_b = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 1, self.style_dim], name='style_b')

        # encode
        content_a, style_a_prime, keypoint_a, attention_a = self.Encoder_A(self.domain_A)
        content_b, style_b_prime, keypoint_b, attention_b = self.Encoder_B(self.domain_B)
        # another images fo doamian A and domain B
        content_a_twin, style_a_twin_prime, keypoint_a_twin, attention_a_twin = self.Encoder_A(self.domain_A_twin, reuse=True)
        content_b_twin, style_b_twin_prime, keypoint_b_twin, attention_b_twin = self.Encoder_B(self.domain_B_twin, reuse=True)

        # decode (within domain)
        x_aa = self.Decoder_A(content_B=content_a, style_A=style_a_prime, label= keypoint_a, attention = attention_a)
        x_bb = self.Decoder_B(content_A=content_b, style_B=style_b_prime, label= keypoint_b, attention = attention_b)

        x_a_twin_2_a_twin = self.Decoder_A(content_B=content_a_twin, style_A=style_a_twin_prime, label= keypoint_a_twin, attention = attention_a_twin, reuse=True)
        x_b_twin_2_b_twin = self.Decoder_B(content_A=content_b_twin, style_B=style_b_twin_prime, label= keypoint_b_twin, attention = attention_b_twin, reuse=True)

        #encoder and decoder within domain guilded by pose
        #content_a_with_pose_from_a_twin, _ = self.Encoder_A(self.domain_A, self.domain_A_twin_key_point_one_hot,reuse = True)
        #content_b_with_pose_from_b_twin, _ = self.Encoder_B(self.domain_B, self.domain_B_twin_key_point_one_hot,reuse = True) 

        #content_a_twin_with_pose_from_a, _ = self.Encoder_A(self.domain_A_twin, self.domain_A_key_point_one_hot, reuse = True )
        #content_b_twin_with_pose_from_b, _ = self.Encoder_B(self.domain_B_twin, self.domain_B_key_point_one_hot, reuse = True )

        x_a2a_twin = self.Decoder_A(content_B=content_a, style_A=style_a_twin_prime, label=keypoint_a_twin, attention = attention_a_twin, reuse=True) 
        x_b2b_twin = self.Decoder_B(content_A=content_b, style_B=style_b_twin_prime, label=keypoint_b_twin, attention = attention_b_twin, reuse=True) 

        x_a_twin2a = self.Decoder_A(content_B=content_a_twin, style_A=style_a_prime, label=keypoint_a, attention = attention_a, reuse = True)
        x_b_twin2b = self.Decoder_B(content_A=content_b_twin, style_B=style_b_prime, label=keypoint_b, attention = attention_b, reuse = True)

        # decode (cross domain)
        x_ba = self.Decoder_A(content_B=content_b, style_A=self.style_a, label=keypoint_b_twin, attention = attention_b_twin, reuse=True)
        x_ab = self.Decoder_B(content_A=content_a, style_B=self.style_b, label=keypoint_a_twin, attention = attention_a_twin, reuse=True)

        x_b_with_pose_from_b_twin_a = self.Decoder_A(content_B=content_b, style_A=self.style_a, label=keypoint_b, attention = attention_b, reuse=True)
        x_a_with_pose_from_a_twin_b = self.Decoder_B(content_A=content_a, style_B=self.style_b, label=keypoint_a, attention = attention_a, reuse=True)


        # encode again
        content_b_, style_a_, keypoint_b_, attention_b_ = self.Encoder_A(x_ba, reuse=True)
        content_a_, style_b_, keypoint_a_, attention_a_  = self.Encoder_B(x_ab, reuse=True)

        self.A_image = tf.summary.image( 'A', self.domain_A )
        self.B_image = tf.summary.image( 'B', self.domain_B )
        self.AA_image = tf.summary.image( 'AA', x_aa )
        self.BB_image = tf.summary.image( 'BB', x_bb )
        self.A_attention_image = tf.summary.image( 'A_attention', attention_a*2 -1)
        self.B_attention_image = tf.summary.image( 'B_attention', attention_b*2 -1)
        self.A2A_twin_image = tf.summary.image('A2A_twin', x_a2a_twin)
        self.B2B_twin_image = tf.summary.image('B2B_twin', x_b2b_twin)
        self.A_twin2A_image = tf.summary.image('A_twin2A', x_a_twin2a)
        self.B_twin2B_image = tf.summary.image('B_twin2B', x_b_twin2b)
        self.A2B_image = tf.summary.image('A2B', x_ab)
        self.B2A_image = tf.summary.image('A2B', x_ba)

         

        # decode again (if needed)
        if self.recon_x_cyc_w > 0 :
            x_aba = self.Decoder_A(content_B=content_a_, style_A=style_a_prime, reuse=True)
            x_bab = self.Decoder_B(content_A=content_b_, style_B=style_b_prime, reuse=True)

            self.ABA_image = tf.summary.image( 'ABA', x_aba )
            self.BAB_image = tf.summary.image( 'BAB', x_bab )

            self.Image_Sum = tf.summary.merge( [ self.A_image, self.B_image, self.AA_image, self.BB_image, self.ABA_image, self.BAB_image ] )

            cyc_recon_A = L1_loss(x_aba, self.domain_A)
            cyc_recon_B = L1_loss(x_bab, self.domain_B)

        else :
            cyc_recon_A = 0.0
            cyc_recon_B = 0.0

            self.Image_Sum = tf.summary.merge( [ self.A_image, self.B_image, self.AA_image, self.BB_image, self.A_attention_image, self.B_attention_image, self.A2A_twin_image, self.B2B_twin_image, self.A_twin2A_image, self.B_twin2B_image, self.A2B_image, self.B2A_image] )
            

        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

        """ Define Loss """
        G_ad_loss_a = generator_loss(self.gan_type, fake_A_logit)
        G_ad_loss_b = generator_loss(self.gan_type, fake_B_logit)

        D_ad_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        D_ad_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)

        recon_A = L1_loss(x_aa, self.domain_A) # reconstruction
        recon_B = L1_loss(x_bb, self.domain_B) # reconstruction

        recon_a_twin_from_a_twin = L1_loss(self.domain_A_twin,x_a_twin_2_a_twin)# reconstruction
        recon_b_twin_from_b_twin = L1_loss(self.domain_B_twin,x_b_twin_2_b_twin)# reconstruction

        # The style reconstruction loss encourages
        # diverse outputs given different style codes
        recon_style_A = L1_loss(style_a_, self.style_a)
        recon_style_B = L1_loss(style_b_, self.style_b)

        # The content reconstruction loss encourages
        # the translated image to preserve semantic content of the input image
        recon_content_A = L1_loss(content_a_, content_a_twin)
        recon_content_B = L1_loss(content_b_, content_b_twin)

        # The keypoint reconstruction loss encourages
        # the translated image to preserve keypoint of the input image
        recon_keypoint_A = L1_loss(keypoint_a_, keypoint_a_twin)
        recon_keypoint_B = L1_loss(keypoint_b_, keypoint_b_twin)

        """Load perceptual loss (VGG16) network"""
        if self.recon_x_twin_w > 0:
            _,_,recon_a_twin_from_a, recon_b_twin_from_b= compute_vgg_loss( [(x_a2a_twin + 1.)*127.5, (self.domain_A_twin+ 1.)*127.5, (x_b2b_twin+ 1.)*127.5, (self.domain_B_twin+ 1.)*127.5], self.vgg_layer_names, self.batch_size, self.img_ch, self.vgg_weight, output_without_instance= True )

            _,_,recon_a_from_a_twin, recon_b_from_b_twin= compute_vgg_loss( [( x_a_twin2a + 1.)*127.5, (self.domain_A+ 1.)*127.5, (x_b_twin2b+ 1.)*127.5, (self.domain_B + 1.)*127.5], self.vgg_layer_names, self.batch_size, self.img_ch, self.vgg_weight, output_without_instance= True )

        else:

            recon_a_twin_from_a = 0 
            recon_b_twin_from_b = 0 
            recon_a_from_a_twin = 0
            recon_b_from_b_twin = 0
         
        Generator_A_loss = self.gan_w * G_ad_loss_a + \
                           self.recon_x_w * recon_A + \
                           self.recon_x_w * recon_a_twin_from_a_twin +\
                           self.recon_x_twin_w * (recon_a_twin_from_a + recon_a_from_a_twin) +\
                           self.recon_s_w * recon_style_A + \
                           self.recon_c_w * recon_content_A + \
                           self.recon_k_w * recon_keypoint_A + \
                           self.recon_x_cyc_w * cyc_recon_A 

        Generator_B_loss = self.gan_w * G_ad_loss_b + \
                           self.recon_x_w * recon_B + \
                           self.recon_x_w * recon_b_twin_from_b_twin + \
                           self.recon_x_twin_w * (recon_b_twin_from_b + recon_b_from_b_twin) + \
                           self.recon_s_w * recon_style_B + \
                           self.recon_c_w * recon_content_B + \
                           self.recon_k_w * recon_keypoint_B + \
                           self.recon_x_cyc_w * cyc_recon_B 

        Discriminator_A_loss = self.gan_w * D_ad_loss_a
        Discriminator_B_loss = self.gan_w * D_ad_loss_b

        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]


        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)
        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)
        self.recon_A = tf.summary.scalar("recon_A", self.recon_x_w * recon_A)
        self.recon_style_A = tf.summary.scalar("recon_style_A", self.recon_s_w * recon_style_A)
        self.recon_content_A = tf.summary.scalar("recon_content_A", self.recon_c_w * recon_content_A )
        self.recon_keypoint_A = tf.summary.scalar("recon_keypoint_A", self.recon_k_w * recon_keypoint_A )
        self.cyc_recon_A = tf.summary.scalar("cyc_recon_A", self.recon_x_cyc_w * cyc_recon_A )
        self.recon_a_twin_from_a = tf.summary.scalar("recon_a_twin_from_a", self.recon_x_twin_w *recon_a_twin_from_a)
        self.recon_b_twin_from_b = tf.summary.scalar("recon_b_twin_from_b", self.recon_x_twin_w *recon_b_twin_from_b)


        
        if self.vgg_w > 0:
            self.G_A_vgg_loss = tf.summary.scalar( "G_A_vgg_loss", G_vgg_loss_a )
            self.G_B_vgg_loss = tf.summary.scalar( "G_B_vgg_loss", G_vgg_loss_b )
    
            self.G_loss = tf.summary.merge([self.G_A_loss, self.G_B_loss, self.all_G_loss, self.G_A_vgg_loss, self.G_B_vgg_loss])
            self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])
        else:
            self.G_loss = tf.summary.merge( [ self.G_A_loss, self.G_B_loss, self.all_G_loss, self.recon_A,self.recon_style_A, self.recon_content_A,self.recon_keypoint_A,self.cyc_recon_A, self.recon_a_twin_from_a,self.recon_b_twin_from_b] )
            self.D_loss = tf.summary.merge( [ self.D_A_loss, self.D_B_loss, self.all_D_loss ] )

        """ Image """
        self.fake_A = x_ba
        self.fake_B = x_ab

        self.fake_A_with_guild_pose=x_b_with_pose_from_b_twin_a
        self.fake_B_with_guild_pose=x_a_with_pose_from_a_twin_b 

        self.real_A = self.domain_A
        self.real_B = self.domain_B

        if self.phase == "test":
            """ Test """
            self.test_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='test_image')
            self.test_style = tf.placeholder(tf.float32, [1, 1, 1, self.style_dim], name='test_style')
    
            test_content_a, _ = self.Encoder_A(self.test_image, reuse=True)
            test_content_b, _ = self.Encoder_B(self.test_image, reuse=True)
    
            self.test_fake_A = self.Decoder_A(content_B=test_content_b, style_A=self.test_style, reuse=True)
            self.test_fake_B = self.Decoder_B(content_A=test_content_a, style_B=self.test_style, reuse=True)

        if self.phase == "guide":
            """ Guided Image Translation """
            self.content_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='content_image')
            self.style_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='guide_style_image')
    
            if self.direction == 'a2b' :
                guide_content_A, guide_style_A = self.Encoder_A(self.content_image, reuse=True)
                guide_content_B, guide_style_B = self.Encoder_B(self.style_image, reuse=True)
    
            else :
                guide_content_B, guide_style_B = self.Encoder_B(self.content_image, reuse=True)
                guide_content_A, guide_style_A = self.Encoder_A(self.style_image, reuse=True)
    
            self.guide_fake_A = self.Decoder_A(content_B=guide_content_B, style_A=guide_style_A, reuse=True)
            self.guide_fake_B = self.Decoder_B(content_A=guide_content_A, style_B=guide_style_B, reuse=True)

        if self.phase == "style_guide_with_key":
            """ Guided Image Translation """
            self.content_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='content_image')
            self.style_image = tf.placeholder(tf.float32, [1, self.img_h, self.img_w, self.img_ch], name='guide_style_image')
            self.key_point_image = tf.placeholder(tf.int32, shape=[1, self.img_h, self.img_w], name='guide_key_point_image') 
            self.key_point_one_hot_image = preprocess_annotation(self.key_point_image, self.anno_cl)

            if self.direction == 'a2b' :
                guide_content_A, guide_style_A = self.Encoder_A(self.content_image, self.key_point_one_hot_image, reuse=True)
                guide_content_B, guide_style_B = self.Encoder_B(self.style_image, self.key_point_one_hot_image,  reuse=True)
    
            else :
                guide_content_B, guide_style_B = self.Encoder_B(self.content_image,  self.key_point_one_hot_image, reuse=True)
                guide_content_A, guide_style_A = self.Encoder_A(self.style_image,  self.key_point_one_hot_image, reuse=True)
    
            # reconstruct by itself
            self.guide_fake_A = self.Decoder_A(content_B=guide_content_A, style_A=guide_style_A, label= self.key_point_one_hot_image, reuse=True)
            self.guide_fake_B = self.Decoder_B(content_A=guide_content_B, style_B=guide_style_B, label= self.key_point_one_hot_image, reuse=True)


    def train(self):
        #print( tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES ) )
        #print("")
        #print( tf.all_variables() )
        #return
        print("########## Starting training ##########")
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        domainA_train, domainA_test = lib.data_cat2lion.load(self.batch_size, self.dataset_name, image_size = self.img_h, domain = 'domainA')
        domainB_train, domainB_test = lib.data_cat2lion.load(self.batch_size, self.dataset_name, image_size = self.img_h, domain = 'domainB')
        def domainA_train_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainA_train():
                    yield images,labels,images_twin, labels_twin 
        def domainB_train_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainB_train():
                    yield images,labels,images_twin, labels_twin 

        domainA_squeue = domainA_train_export()
        domainB_squeue = domainB_train_export()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        
        # loop for epoch
        sample_num = 5
        start_time = time.time()
        Image_A_twin_set = {i: None for i in xrange(sample_num)}
        Image_B_twin_set = {i: None for i in xrange(sample_num)}
        Noise_A_set = {i: None for i in xrange(sample_num)}
        Noise_B_set = {i: None for i in xrange(sample_num)}
        for i_ in xrange(sample_num):
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                _imagesA,_labelsA,_imagesA_twin, _labelsA_twin = domainA_squeue.next() 
                _imagesB,_labelsB,_imagesB_twin, _labelsB_twin = domainB_squeue.next() 

                Image_A_twin_set[i_] =_imagesA_twin.copy()
                Image_B_twin_set[i_] =_imagesB_twin.copy() 
                Noise_A_set[i_] = style_a.copy()
                Noise_B_set[i_] = style_b.copy()
        for i_ in xrange(sample_num):
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                _imagesA,_labelsA,_imagesA_twin, _labelsA_twin = domainA_squeue.next() 
                _imagesB,_labelsB,_imagesB_twin, _labelsB_twin = domainB_squeue.next() 

                Image_A_twin_set[i_] =_imagesA_twin.copy()
                Image_B_twin_set[i_] =_imagesB_twin.copy() 
                Noise_A_set[i_] = style_a.copy()
                Noise_B_set[i_] = style_b.copy()
        for epoch in range(start_epoch, self.epoch):
            lr = self.init_lr * pow(0.5, epoch)
            for idx in range(start_batch_id, self.iteration):
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                _imagesA,_labelsA,_imagesA_twin, _labelsA_twin = domainA_squeue.next() 
                _imagesB,_labelsB,_imagesB_twin, _labelsB_twin = domainB_squeue.next() 

                Image_A_twin_set[np.mod( idx, sample_num)] =_imagesA_twin.copy() 
                Image_B_twin_set[np.mod( idx, sample_num)] =_imagesB_twin.copy() 
                Noise_A_set[np.mod( idx, sample_num)] = style_a
                Noise_B_set[np.mod( idx, sample_num)] = style_b

                train_feed_dict = {
                    self.style_a : style_a,
                    self.style_b : style_b,
                    self.lr : lr,
                    self.domain_A : _imagesA,
                    self.domain_B : _imagesB,
                    self.domain_A_twin : _imagesA_twin, 
                    self.domain_B_twin : _imagesB_twin,
                    self.domain_A_key_point : _labelsA,  
                    self.domain_B_key_point : _labelsB,
                    self.domain_A_twin_key_point : _labelsA_twin,  
                    self.domain_B_twin_key_point : _labelsB_twin
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_images, batch_B_images, fake_A, fake_A_with_guild_pose, fake_B, fake_B_with_guild_pose, _, g_loss, summary_str, summary_image, summary_lr = self.sess.run([self.real_A, self.real_B, self.fake_A, self.fake_A_with_guild_pose, self.fake_B, self.fake_B_with_guild_pose, self.G_optim, self.Generator_loss, self.G_loss, self.Image_Sum, self.lr_sum], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)
                
                self.writer.add_summary(summary_lr, counter)
                
                if np.mod( idx + 1, self.img_freq ) == 0:
                    self.writer.add_summary( summary_image, counter )

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                if np.mod(idx+1, self.print_freq) == 0 :


                    DomainA2B = np.zeros((self.img_h*sample_num, self.img_w*(sample_num + 2), 3))
                    DomainB2A = np.zeros((self.img_h*sample_num, self.img_w*(sample_num + 2), 3))

                    for _iter_pose in xrange(sample_num):

                        for _iter_style in xrange(sample_num):
                            train_feed_dict = {
                                self.style_a : Noise_A_set[_iter_style],
                                self.style_b : Noise_B_set[_iter_style],
                                self.lr : lr,
                                self.domain_A : _imagesA,
                                self.domain_B : _imagesB,
                                self.domain_A_twin : Image_A_twin_set[_iter_pose], 
                                self.domain_B_twin : Image_B_twin_set[_iter_pose],
                                self.domain_A_key_point : _labelsA,  
                                self.domain_B_key_point : _labelsB,
                                self.domain_A_twin_key_point : _labelsA_twin,  
                                self.domain_B_twin_key_point : _labelsB_twin
                            }


                            batch_A_images,batch_A_twin_images, batch_B_images, batch_B_twin_images, fake_A, fake_B  = self.sess.run([self.real_A, self.domain_A_twin, self.real_B,self.domain_B_twin, self.fake_A, self.fake_B], feed_dict = train_feed_dict)
                            DomainA2B[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), : self.img_w ,:] = batch_A_images[0].copy() 
                            DomainA2B[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w : self.img_w * 2,:] = batch_A_twin_images[0].copy() 
                            DomainA2B[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w * (_iter_style + 2): self.img_w * (_iter_style + 3),:] = fake_B[0].copy() 

                            DomainB2A[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), : self.img_w ,:] = batch_B_images[0].copy() 
                            DomainB2A[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w : self.img_w * 2,:] = batch_B_twin_images[0].copy() 
                            DomainB2A[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w * (_iter_style + 2): self.img_w * (_iter_style + 3),:] = fake_A[0].copy() 
                    DomainA2B = (DomainA2B + 1.)*127.5
                    DomainB2A = (DomainB2A + 1.)*127.5
                    IMSAVE('{}/real_A2B_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1), DomainA2B)    
                    IMSAVE('{}/real_B2A_{:02d}_{:06d}.jpg'.format(self.sample_dir, epoch, idx+1), DomainB2A)    


                if np.mod(idx+1, self.save_freq) == 0 :

                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.prefix is not None:
            return "{}_{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type, self.prefix)
        else:
            return "{}_{}_{}".format( self.model_name, self.dataset_name, self.gan_type )

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    def test_with_key_v2(self):
        tf.global_variables_initializer().run()
        # saver to save model
        self.saver = tf.train.Saver()
        # restore check-point if it exits
        domainA_train, domainA_test = lib.data_cat2lion.load(self.batch_size, self.dataset_name, image_size = self.img_h, domain = 'domainA')
        domainB_train, domainB_test = lib.data_cat2lion.load(self.batch_size, self.dataset_name, image_size = self.img_h, domain = 'domainB')
        def domainA_train_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainA_train():
                    yield images,labels,images_twin, labels_twin 
        def domainB_train_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainB_train():
                    yield images,labels,images_twin, labels_twin 

        domainA_squeue = domainA_train_export()
        domainB_squeue = domainB_train_export()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        # upload the facial_keypoint_detector

        #SAVER = tf.train.Saver(self.facial_detector)
        #SAVER.restore(self.sess, 'facial_keypoint_model/check-25000')

        
        # loop for epoch
        start_time = time.time()
        sample_num = 5
        Image_A_twin_set = {i: None for i in xrange(sample_num)}
        Image_B_twin_set = {i: None for i in xrange(sample_num)}
        Noise_A_set = {i: None for i in xrange(sample_num)}
        Noise_B_set = {i: None for i in xrange(sample_num)}
        for i_ in xrange(sample_num):
            style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
            style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
            _imagesA,_labelsA,_imagesA_twin, _labelsA_twin = domainA_squeue.next() 
            _imagesB,_labelsB,_imagesB_twin, _labelsB_twin = domainB_squeue.next() 

            Image_A_twin_set[i_] =_imagesA_twin.copy()
            Image_B_twin_set[i_] =_imagesB_twin.copy() 
            Noise_A_set[i_] = style_a.copy()
            Noise_B_set[i_] = style_b.copy()
        for epoch in range(start_epoch, self.epoch):
            lr = self.init_lr * pow(0.5, epoch)
            for idx in range(start_batch_id, self.iteration):
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, 1, self.style_dim])
                _imagesA,_labelsA,_imagesA_twin, _labelsA_twin = domainA_squeue.next() 
                _imagesB,_labelsB,_imagesB_twin, _labelsB_twin = domainB_squeue.next() 

                Image_A_twin_set[np.mod( idx, sample_num)] =_imagesA_twin.copy() 
                Image_B_twin_set[np.mod( idx, sample_num)] =_imagesB_twin.copy() 
                Noise_A_set[np.mod( idx, sample_num)] = style_a
                Noise_B_set[np.mod( idx, sample_num)] = style_b

                DomainA2B = np.zeros((self.img_h*sample_num, self.img_w*(sample_num + 2), 3))
                DomainB2A = np.zeros((self.img_h*sample_num, self.img_w*(sample_num + 2), 3))

                if np.mod( idx, sample_num) == 0:
                    for _iter_pose in xrange(sample_num):
                        for _iter_style in xrange(sample_num):
                            train_feed_dict = {
                                    self.style_a : Noise_A_set[_iter_style],
                                    self.style_b : Noise_B_set[_iter_style],
                                    self.lr : lr,
                                    self.domain_A : _imagesA,
                                    self.domain_B : _imagesB,
                                    self.domain_A_twin : Image_A_twin_set[_iter_pose], 
                                    self.domain_B_twin : Image_B_twin_set[_iter_pose],
                                    self.domain_A_key_point : _labelsA,  
                                    self.domain_B_key_point : _labelsB,
                                    self.domain_A_twin_key_point : _labelsA_twin,  
                                    self.domain_B_twin_key_point : _labelsB_twin
                            }

                            batch_A_images,batch_A_twin_images, batch_B_images, batch_B_twin_images, fake_A, fake_A_with_guild_pose, fake_B, fake_B_with_guild_pose, domain_A_twin_key_point, domain_B_twin_key_point = self.sess.run([self.real_A, self.domain_A_twin, self.real_B,self.domain_B_twin, self.fake_A, self.fake_A_with_guild_pose, self.fake_B, self.fake_B_with_guild_pose, self.domain_A_twin_key_point, self.domain_B_twin_key_point], feed_dict = train_feed_dict)
                            DomainA2B[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), : self.img_w ,:] = batch_A_images[0].copy() 
                            DomainA2B[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w : self.img_w * 2,:] = batch_A_twin_images[0].copy() 
                            DomainA2B[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w * (_iter_style + 2): self.img_w * (_iter_style + 3),:] = fake_B_with_guild_pose[0].copy() 

                            DomainB2A[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), : self.img_w ,:] = batch_B_images[0].copy() 
                            DomainB2A[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w : self.img_w * 2,:] = batch_B_twin_images[0].copy() 
                            DomainB2A[self.img_h * _iter_pose : self.img_h * (_iter_pose + 1), self.img_w * (_iter_style + 2): self.img_w * (_iter_style + 3),:] = fake_A_with_guild_pose[0].copy() 
                    DomainA2B = (DomainA2B + 1.)*127.5
                    DomainB2A = (DomainB2A + 1.)*127.5
                    IMSAVE('{}/real_A2B_{:02d}_{:06d}.jpg'.format(self.result_dir, 0, idx+1), DomainA2B)    
                    IMSAVE('{}/real_B2A_{:02d}_{:06d}.jpg'.format(self.result_dir, 0, idx+1), DomainB2A)    
                    print idx 

    def test_with_key(self):
        tf.global_variables_initializer().run()

        batch_test = 5
        self.num_key_point = batch_test
        domainA_train, domainA_test = lib.data.load(batch_test, self.dataset_name, image_size = self.img_h, domain = 'domainA')
        domainB_train, domainB_test = lib.data.load(batch_test, self.dataset_name, image_size = self.img_h, domain = 'domainB')

        def domainA_test_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainA_train():
                    yield images,labels,images_twin, labels_twin 
        def domainB_test_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainB_train():
                    yield images,labels,images_twin, labels_twin 

        domainA_squeue = domainA_test_export()
        domainB_squeue = domainB_test_export()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        self.saver.restore(self.sess, 'result/face2cat_key_point/checkpoint/copy of MUNIT_face2cat_lsgan_2perl/MUNIT.model-97001') 
        # write html for visual comparison

        _idx = 0
        while True : # A -> B
            _imagesA,_labelsA,_imagesA_twin, _labelsA_twin = domainA_squeue.next() 
            _imagesB,_labelsB,_imagesB_twin, _labelsB_twin = domainB_squeue.next() 
            style_a = np.random.normal(loc=0.0, scale=1.0, size=[batch_test, 1, 1, self.style_dim])
            style_b = np.random.normal(loc=0.0, scale=1.0, size=[batch_test, 1, 1, self.style_dim])
            for _imgs in xrange(1):
                fake_imageA = np.zeros((self.img_h * (batch_test + 1), self.img_w* (batch_test + 1), self.img_ch), dtype = np.float32)
                fake_imageA[:self.img_h, :self.img_w, :self.img_ch] = (_imagesA[_imgs] + 1.)*127.5 
                for i in range(self.num_style) :
                    for j in range(self.num_key_point) :

                        train_feed_dict = {
                            self.style_b : np.expand_dims(style_b[i], axis=0),
                            self.domain_A : np.expand_dims(_imagesA[_imgs], axis=0),
                            self.domain_A_twin : np.expand_dims(_imagesA_twin[j], axis=0), 
                            self.domain_A_key_point : np.expand_dims(_labelsA[_imgs], axis=0),  
                            self.domain_A_twin_key_point : np.expand_dims(_labelsA_twin[j], axis=0),  
                        }
                        fake_img_key_itself, fake_img_key_twin = self.sess.run([self.fake_B, self.fake_B_with_guild_pose], feed_dict =train_feed_dict)

                        if i ==0:
                            key_point = _labelsA_twin[j].copy()
                            key_point[key_point!=-1] = 255
                            key_point = np.expand_dims(key_point, axis = 2)
                            key_point = np.concatenate((key_point,key_point,key_point), axis = 2)
                            fake_imageA[(j + 1)*self.img_h:(j + 2)*self.img_h,  : self.img_w, :self.img_ch] = key_point.copy() 
                        if j ==0:
                            fake_imageA[:self.img_h,  (i + 1) * self.img_w:(i + 2) * self.img_w, :self.img_ch] = (fake_img_key_itself[0].copy() + 1.)*127.5 

                        begin = (j + 1)*self.img_h
                        end = (j + 2)*self.img_h
                        begin_ = (i+1) * self.img_w
                        end_ = (i+2) * self.img_w
                        fake_imageA[begin:end,  begin_:end_, :] = (fake_img_key_twin[0].copy() + 1.0)*127.5 
            for _imgs in xrange(1):
                fake_imageB = np.zeros((self.img_h * (batch_test + 1), self.img_w* (batch_test + 1), self.img_ch), dtype = np.float32)
                fake_imageB[:self.img_h, :self.img_w, :self.img_ch] = (_imagesB[_imgs] + 1.)*127.5 
                for i in range(self.num_style) :
                    for j in range(self.num_key_point) :

                        train_feed_dict = {
                            self.style_a : np.expand_dims(style_a[i], axis=0),
                            self.domain_B : np.expand_dims(_imagesB[_imgs], axis=0),
                            self.domain_B_twin : np.expand_dims(_imagesB_twin[j], axis=0), 
                            self.domain_B_key_point : np.expand_dims(_labelsB[_imgs], axis=0),  
                            self.domain_B_twin_key_point : np.expand_dims(_labelsB_twin[j], axis=0),  
                        }
                        fake_img_key_itself, fake_img_key_twin = self.sess.run([self.fake_A, self.fake_A_with_guild_pose], feed_dict =train_feed_dict)

                        if i ==0:
                            key_point = _labelsB_twin[j].copy()
                            key_point[key_point!=-1] = 255
                            key_point = np.expand_dims(key_point, axis = 2)
                            key_point = np.concatenate((key_point,key_point,key_point), axis = 2)
                            fake_imageB[(j + 1)*self.img_h:(j + 2)*self.img_h,  : self.img_w, :self.img_ch] = key_point.copy() 
                        if j ==0:
                            fake_imageB[:self.img_h,  (i + 1) * self.img_w:(i + 2) * self.img_w, :self.img_ch] = (fake_img_key_itself[0].copy() + 1.)*127.5 

                        begin = (j + 1)*self.img_h
                        end = (j + 2)*self.img_h
                        begin_ = (i+1) * self.img_w
                        end_ = (i+2) * self.img_w
                        fake_imageB[begin:end,  begin_:end_, :] = (fake_img_key_twin[0].copy() + 1.0)*127.5 


                if not os.path.exists('%s/A'%self.result_dir):
                    os.makedirs('%s/A'%self.result_dir)
                if not os.path.exists('%s/B'%self.result_dir):
                    os.makedirs('%s/B'%self.result_dir)
                IMSAVE('%s/A/A_%d.png'%(self.result_dir, _idx), fake_imageA)
                IMSAVE('%s/B/B_%d.png'%(self.result_dir, _idx), fake_imageB)
                print _idx
                _idx += 1

       # for sample_file  in test_B_files : # B -> A
       #     print('Processing B image: ' + sample_file)
       #     sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
       #     file_name = os.path.basename(sample_file).split(".")[0]
       #     file_extension = os.path.basename(sample_file).split(".")[1]

       #     for i in range(self.num_style):
       #         test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
       #         image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

       #         fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image, self.test_style: test_style})
       #         save_images(fake_img, [1, 1], image_path)

       #         index.write("<td>%s</td>" % os.path.basename(image_path))
       #         index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
       #                 '../..' + os.path.sep + sample_file), self.img_w, self.img_h))
       #         index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
       #                 '../..' + os.path.sep + image_path), self.img_w, self.img_h))
       #         index.write("</tr>")
        index.close()
    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style) :
                test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_image : sample_image, self.test_style : test_style})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                    '../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
            file_name = os.path.basename(sample_file).split(".")[0]
            file_extension = os.path.basename(sample_file).split(".")[1]

            for i in range(self.num_style):
                test_style = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, self.style_dim])
                image_path = os.path.join(self.result_dir, '{}_style{}.{}'.format(file_name, i, file_extension))

                fake_img = self.sess.run(self.test_fake_A, feed_dict={self.test_image: sample_image, self.test_style: test_style})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")
        index.close()

    def style_guide_test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        style_file = np.asarray(load_test_data(self.guide_img, size_h=self.img_h, size_w=self.img_w))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir, 'guide')
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        if self.direction == 'a2b' :
            for sample_file in test_A_files:  # A -> B
                print('Processing A image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_B, feed_dict={self.content_image: sample_image, self.style_image : style_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")

        else :
            for sample_file in test_B_files:  # B -> A
                print('Processing B image: ' + sample_file)
                sample_image = np.asarray(load_test_data(sample_file, size_h=self.img_h, size_w=self.img_w))
                image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

                fake_img = self.sess.run(self.guide_fake_A, feed_dict={self.content_image: sample_image, self.style_image : style_file})
                save_images(fake_img, [1, 1], image_path)

                index.write("<td>%s</td>" % os.path.basename(image_path))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_w, self.img_h))
                index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_w, self.img_h))
                index.write("</tr>")
        index.close()
    def style_guide_with_key(self):
        tf.global_variables_initializer().run()

        batch_test = 5
        self.num_key_point = batch_test
        domainA_train, domainA_test = lib.data.load(batch_test, self.dataset_name, image_size = self.img_h, domain = 'domainA')
        domainB_train, domainB_test = lib.data.load(batch_test, self.dataset_name, image_size = self.img_h, domain = 'domainB')
        style_file = np.asarray(load_test_data(self.guide_img, size_h=self.img_h, size_w=self.img_w))

        def domainA_test_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainA_train():
                    yield images,labels,images_twin, labels_twin 
        def domainB_test_export():
            while True:
                for (images,labels,images_twin, labels_twin) in domainB_train():
                    yield images,labels,images_twin, labels_twin 

        domainA_squeue = domainA_test_export()
        domainB_squeue = domainB_test_export()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        self.saver.restore(self.sess, 'result/face2cat_key_point/checkpoint/copy of MUNIT_face2cat_lsgan_2perl/MUNIT.model-105000') 
        # write html for visual comparison

        _idx = 0
        while True : # A -> B
            _imagesA,_labelsA,_imagesA_twin, _labelsA_twin = domainA_squeue.next() 
            _imagesB,_labelsB,_imagesB_twin, _labelsB_twin = domainB_squeue.next() 
            style_a = np.random.normal(loc=0.0, scale=1.0, size=[batch_test, 1, 1, self.style_dim])
            style_b = np.random.normal(loc=0.0, scale=1.0, size=[batch_test, 1, 1, self.style_dim])
            for _imgs in xrange(1):
                fake_imageA = np.zeros((self.img_h * (batch_test + 1), self.img_w* (batch_test + 1), self.img_ch), dtype = np.float32)
                fake_imageA[:self.img_h, :self.img_w, :self.img_ch] = (_imagesA[_imgs] + 1.)*127.5 
                for i in range(self.num_style) :
                    for j in range(self.num_key_point) :
                        train_feed_dict = {
                           # self.style_b : np.expand_dims(style_b[i], axis=0),
                           # self.domain_A : np.expand_dims(_imagesA[_imgs], axis=0),
                           # self.domain_A_twin : np.expand_dims(_imagesA_twin[j], axis=0), 
                           # self.domain_A_key_point : np.expand_dims(_labelsA[_imgs], axis=0),  
                           # self.domain_A_twin_key_point : np.expand_dims(_labelsA_twin[j], axis=0),  
                            self.content_image : np.expand_dims(_imagesA[_imgs], axis=0),
                            #self.style_image : np.expand_dims(_imagesB[_imgs], axis=0),
                            self.style_image : style_file,
                            self.key_point_image : np.expand_dims(_labelsA[j], axis=0),
                        }
                        fake_img_key_itself, fake_img_key_twin = self.sess.run([self.guide_fake_A, self.guide_fake_A], feed_dict =train_feed_dict)

                        if i ==0:
                            key_point = _labelsA_twin[j].copy()
                            key_point[key_point!=-1] = 255
                            key_point = np.expand_dims(key_point, axis = 2)
                            key_point = np.concatenate((key_point,key_point,key_point), axis = 2)
                            fake_imageA[(j + 1)*self.img_h:(j + 2)*self.img_h,  : self.img_w, :self.img_ch] = key_point.copy() 
                        if j ==0:
                            fake_imageA[:self.img_h,  (i + 1) * self.img_w:(i + 2) * self.img_w, :self.img_ch] = (fake_img_key_itself[0].copy() + 1.)*127.5 

                        begin = (j + 1)*self.img_h
                        end = (j + 2)*self.img_h
                        begin_ = (i+1) * self.img_w
                        end_ = (i+2) * self.img_w
                        fake_imageA[begin:end,  begin_:end_, :] = (fake_img_key_twin[0].copy() + 1.0)*127.5 

                if not os.path.exists('%s/guide'%self.result_dir):
                    os.makedirs('%s/guide'%self.result_dir)
                IMSAVE('%s/guide/A_%d.png'%(self.result_dir, _idx), fake_imageA)
                print _idx
                _idx += 1
        index.close()
