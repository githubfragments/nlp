import os.path
from pathlib2 import Path
import configargparse
import pprint
import json

from nlp.util.utils import mkdirs, adict
from nlp.util import utils as U

## namespace->dict
def ns2dict(args):
    return vars(args)

## dict->namespace
def dict2ns(dict):
    return adict(dict)

def dump_flags_to_json(flags, file):
    with open(file, 'w') as f:
        json.dump(ns2dict(flags), f)#, ensure_ascii=False)

def restore_flags_from_json(file):
    with open(file, 'r') as f:
        str = f.read()#.encode('utf8')
    d = json.loads(str)
    return dict2ns(d)

def dump_config(flags, file):
    with open(file, 'w') as f:
        for k, v in sorted(flags.items()):
            if k != 'config':
                f.write('{} \t= {}\n'.format(k,v))
                
def save_local_config(flags, verbose=True):
    loc_file = os.path.abspath(os.path.join(flags.chkpt_dir, os.path.basename(flags.config)))
    abs_config = os.path.abspath(flags.config)
    if os.path.realpath(loc_file) != os.path.realpath(abs_config):
        if not os.path.exists(flags.chkpt_dir):
            mkdirs(FLAGS.chkpt_dir)
            if verbose:
                print('Created checkpoint directory: {}'.format(os.path.abspath(flags.chkpt_dir)))
        dump_config(flags, loc_file)
        if verbose:
            print('Saving FLAGS to: {}'.format(loc_file))

def get_config(config_file=None, argv=[], parser=None):
    if config_file:
        # if passed an override config file -->
        # --> override chkpt_dir to point to same dir as config
        argv.append('--config'); argv.append(config_file)
        config_dir = os.path.dirname(config_file)
        if not config_dir.endswith('config'):
            argv.append('--chkpt_dir'); argv.append(config_dir)
    return parse_argv(argv=argv, parser=parser)

def parse_argv(argv=None, parser=None):
    if not parser:
        parser = get_demo_parser()
    args, unparsed = parser.parse_known_args(argv)
    if len(unparsed)>0:
        print('WARNING -- UNKOWN ARGUMENTS...')
        pprint.pprint(unparsed)
        print('\n')
    args = dict2ns(ns2dict(args)) ## convert namespace to attribute_dictionary
    #pprint.pprint(args)
    return args

def get_ids(fn, default=None):
    try:
        ids = set(U.read_col(fn, col=0, type='unicode'))
    except IOError:
        ids = default
    return ids

def parse_config(config_file, parser):
    #parser = options.get_parser()
    argv=[]# override config file here
    FLAGS = get_config(parser=parser, config_file=config_file, argv=argv)
    FLAGS.chkpt_dir = U.make_abs(FLAGS.chkpt_dir)
    FLAGS.rand_seed = U.seed_random(FLAGS.rand_seed)
    
    if FLAGS.att_size>0:
        FLAGS.mean_pool = False
        if FLAGS.att_type<0:
            FLAGS.att_type=0
    if FLAGS.embed_type=='word':
        FLAGS.model_std = None
        FLAGS.attn_std = None
    
    valid_ids, valid_id_file = None, None
    if FLAGS.valid_pat is None:
        FLAGS.save_valid = None
        FLAGS.load_valid = None
    else:
        valid_id_file = os.path.join(FLAGS.id_dir, FLAGS.valid_pat).format(FLAGS.item_id)
        if FLAGS.load_valid:
            valid_ids = get_ids(valid_id_file)
        if FLAGS.save_valid and valid_ids is not None:
            FLAGS.save_valid = False
    FLAGS.valid_ids = valid_ids
    FLAGS.valid_id_file = valid_id_file
    
    ''' don't overwrite MLT test ids!!! '''
    if FLAGS.valid_id_file.endswith('test_ids.txt'):
        FLAGS.save_valid = False
    
    train_ids = []
    if FLAGS.train_pat:
        train_id_file = os.path.join(FLAGS.id_dir, FLAGS.train_pat).format(FLAGS.item_id)
        train_ids = get_ids(train_id_file, default=[])
    FLAGS.train_ids = train_ids
    
    return FLAGS
  

## demo parser...
def get_demo_parser():
    p = configargparse.ArgParser()#default_config_files=['config/model.conf'])
    
    p.add('-c', '--config', required=False, is_config_file=True, default='config/model.conf', help='config file path')
    
    p.add("-td", "--chkpt_dir", type=str, metavar='<str>', required=True, help="The path to the checkpoint dir")
    p.add("-dd", "--data_dir", type=str, metavar='<str>', required=True, help="The path to the data dir")
    p.add("-vf", "--vocab_file", type=str, metavar='<str>', required=True, help="The path to the vocab file")
    p.add("-rs", "--rnn_size", type=int, metavar='<int>', default=500, help='size of LSTM internal state')#500
    
    p.add("-bs", "--batch_size", type=int, metavar='<int>', default=128, help='number of sequences to train on in parallel')
    p.add("-rl", "--rnn_layers", type=int, metavar='<int>', default=2, help='number of layers in the LSTM')#2
    p.add("-hl", "--highway_layers", type=int, metavar='<int>', default=2, help='number of highway layers')#2
    p.add("-ces","--char_embed_size", type=int, metavar='<int>', default=15, help='dimensionality of character embeddings')
    p.add("-nus","--num_unroll_steps", type=int, metavar='<int>', default=35, help='number of timesteps to unroll for (word sequence length)')
    p.add("-me", "--max_epochs", type=int, metavar='<int>', default=50, help="")
    p.add("-mw", "--max_word_length", type=int, metavar='<int>', default=65, help="")

    p.add("--dropout", type=float, metavar='<float>', default=0.5, help="")
    
    p.add("-kw", "--kernel_widths", type=str, metavar='<str>', default='[1,2,3,4,5,6,7]', help="")
    p.add("-kf", "--kernel_features", type=str, metavar='<str>', default='[50,100,150,200,200,200,200]', help="")
    
    p.add("-lr", "--learning_rate", type=float, metavar='<float>', default=1.0, help='starting learning rate')
    p.add("-d",  "--learning_rate_decay", type=float, metavar='<float>', default=0.75, help='learning rate decay')
    p.add("-dw", "--decay_when", type=float, metavar='<float>', default=1.5, help='decay if validation perplexity does not improve by more than this much')
    p.add("-pi", "--param_init", type=float, metavar='<float>', default=0.05, help='initialize parameters at')
    p.add("-mn", "--max_grad_norm", type=float, metavar='<float>', default=5.0, help='normalize gradients at')
    
    p.add("-pe", "--print_every", type=int, metavar='<int>', default=50, help='how often to print current loss (batches)')
    p.add("-se", "--save_every", type=int, metavar='<int>', default=2, help='how often to validate AND save checkpoint (shards)')
    p.add("-vs", "--num_valid_steps", type=int, metavar='<int>', default=100, help='num validation steps (batches)')
    p.add("-vpe","--valid_print_every", type=int, metavar='<int>', default=10, help='how often to print current loss (batches)')
    p.add("-de", "--decay_every", type=int, metavar='<int>', default=50, help='how often to decay lr (shards)')
    
    p.add("--EOS", type=str, metavar='<str>', default='+', help='<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
    p.add("--seed", type=int, metavar='<int>', default=0, help="")
    
    return p


if __name__ == "__main__":

    config_file = None
    #config_file = 'chkpt/mod1_650-20/model.conf'
    FLAGS = get_config(config_file)
    pprint.pprint(FLAGS)
