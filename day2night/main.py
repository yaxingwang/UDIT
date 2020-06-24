from UDIT import UDIT
import argparse
from utils import *
import os
import pdb

# TODO: Implement Visualizers like in https://github.com/azadis/MC-GAN

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of UNIT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', type=str, default='configs/UNIT_flat2texture.yaml', help='Path to the config file.')
    opts = parser.parse_args()

    config = get_config(opts.config)

    return check_args(config)

"""checking arguments"""
def check_args(config):
    # --checkpoint_dir
    check_folder(config['checkpoint_dir'])

    # --result_dir
    check_folder(config['result_dir'])

    # --result_dir
    check_folder(config['log_dir'])

    # --sample_dir
    check_folder(config['sample_dir'])

    # --epoch
    try:
        assert config['epoch'] >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert config['batch_size'] >= 1
    except:
        print('batch size must be larger than or equal to one')
    return config

"""main"""
def main():
    # parse arguments
    conf = parse_args()
    if conf is None:
      exit()

    os.environ[ "CUDA_VISIBLE_DEVICES" ] = str(conf['GPU'])
    
    # open session
    config = tf.ConfigProto( allow_soft_placement = False, log_device_placement = True )
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        gan = UDIT(sess, conf)

        # build graph
        gan.build_model()

        # show network architecture
        #show_all_variables()

        if conf['phase'] == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if conf['phase'] == 'test' :
            gan.test()
            print(" [*] Test finished!")

        if conf['phase'] == 'test_with_key_v2':
            gan.test_with_key_v2()
            print(" [*] Test finished!")
        if conf['phase'] == 'test_with_key_v4' :
            gan.test_with_key_v4()
            print(" [*] Guide finished!")

        if conf['phase'] == 'guide' :
            gan.style_guide_test()
            print(" [*] Guide finished!")


if __name__ == '__main__':
    main()
