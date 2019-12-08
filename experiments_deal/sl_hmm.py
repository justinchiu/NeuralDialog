import time
import os
import json
import torch as th
import logging
from latent_dialog.utils import Pack, prepare_dirs_loggers, set_seed
from latent_dialog.corpora import DealCorpus
from latent_dialog.dealta_loaders import DealDataLoaders
from latent_dialog.evaluators import BleuEvaluator
from latent_dialog.models_deal import CatHRED
from latent_dialog.models_hmm import Hmm
from latent_dialog.main import train, validate, generate
import latent_dialog.domain as domain


stats_path = 'config_log_model'
if not os.path.exists(stats_path):
    os.mkdir(stats_path)
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
print('[START]', start_time, '='*30)

domain_name = 'object_division'
domain_info = domain.get_domain(domain_name)

config = Pack(
    train_path = '../data/negotiate/train.txt',
    val_path = '../data/negotiate/val.txt',
    test_path = '../data/negotiate/test.txt',
    last_n_model = 4,
    max_utt_len = 20,
    #backward_size = 14, 
    backward_size = 8, 
    #backward_size = 1, 
    #batch_size = 32,
    batch_size = 4,
    grad_clip = 10.0,
    use_gpu = True,
    op = 'adam', 
    init_lr = 0.001,
    l2_norm=0.00001,
    momentum = 0.0, 
    dropout = 0.3,
    max_epoch = 100,
    embed_size = 256, 
    #num_layers = 1, 
    num_layers = 2, 
    utt_rnn_cell = 'gru', 
    utt_cell_size = 128, 
    bi_utt_cell = True, 
    enc_use_attn = False, 
    ctx_rnn_cell = 'gru',
    ctx_cell_size = 256,
    bi_ctx_cell = False,
    z_size = 128,
    #beta = 0.01,
    #simple_posterior = False,
    #use_pr = True,
    dec_use_attn = False,
    dec_rnn_cell = 'gru', # must be same as ctx_cell_size due to the passed initial state
    dec_cell_size = 256,  # must be same as ctx_cell_size due to the passed initial state
    dec_attn_mode = 'cat', 
    #
    fix_train_batch = False,
    fix_batch = False,
    beam_size = 20,
    avg_type = 'real_word',
    print_step = 100,
    ckpt_step = 400,
    #ckpt_step = 2523,
    improve_threshold = 0.996, 
    patient_increase = 2.0, 
    save_model = True, 
    early_stop = False, 
    gen_type = 'greedy', 
    preview_batch_num = 1,
    max_dec_len = 40, 
    k = domain_info.input_length(), 
    goal_embed_size = 64, 
    goal_nhid = 64, 
    init_range = 0.1,
    pretrain_folder = '2019-12-06-02-20-58-sl_hmm',
    forward_only = False,
    #forward_only = True,
    # options for sequence LVMs
    seq = True,
    noisy_proposal_labels = True,
    sup_proposal_labels = False,
    #sup_proposal_labels = True,
    #label_weight = 0.1,
    label_weight = 1,
)

set_seed(10)

if config.forward_only:
    saved_path = os.path.join(stats_path, config.pretrain_folder)
    config = Pack(json.load(open(os.path.join(saved_path, 'config.json'))))
    config['forward_only'] = True
else:
    saved_path = os.path.join(stats_path, start_time+'-'+os.path.basename(__file__).split('.')[0])
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

config.saved_path = saved_path

prepare_dirs_loggers(config)
logger = logging.getLogger()
logger.info('[START]\n{}\n{}'.format(start_time, '='*30))

# save configuration
with open(os.path.join(saved_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4) # sort_keys=True

corpus = DealCorpus(config)
train_dial, val_dial, test_dial = corpus.get_corpus()

train_data = DealDataLoaders('Train', train_dial, config)
val_data = DealDataLoaders('Val', val_dial, config)
test_data = DealDataLoaders('Test', test_dial, config)

evaluator = BleuEvaluator('Deal')

model = Hmm(corpus, config)

if config.use_gpu:
    model.cuda()

best_epoch = None
if not config.forward_only:
    try:
        #best_epoch = train(model, train_data, val_data, test_data, config, evaluator, gen=generate)
        best_epoch = train(model, train_data, val_data, test_data, config, evaluator, gen=None)
    except KeyboardInterrupt:
        print('Training stopped by keyboard.')

config.batch_size = 4
if best_epoch is None:
    model_ids = sorted([int(p.replace('-model', '')) for p in os.listdir(saved_path) if '-model' in p])
    best_epoch = model_ids[-1]

model.load_state_dict(th.load(os.path.join(saved_path, '{}-model'.format(best_epoch))))
logger.info("Load model {}".format(best_epoch))
logger.info("Forward Only Evaluation")
# run the model on the test dataset
validate(model, val_data, config)
validate(model, test_data, config) 

#with open(os.path.join(saved_path, '{}_test_file.txt'.format(start_time)), 'w') as f:
    #generate(model, test_data, config, evaluator, num_batch=None, dest_f=f)

end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[END]'+ end_time+ '='*30)
