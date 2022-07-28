import argparse
import logging
import os
import json
import pickle


def _set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True, help='use GPU')

    parser.add_argument('--do_train', default=True)
    parser.add_argument('--do_valid', default=True)
    parser.add_argument('--do_test', default=True)

    parser.add_argument('--data_path', type=str, default='data_concept/FB15k-237_concept/')
    parser.add_argument('--model', default='TransE', type=str)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=1000, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', default=True)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)

    parser.add_argument('--test_batch_size', default=64, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.00005, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=100, type=int)
    parser.add_argument('--uni_weight', default=False, help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default='models/TransE_FB15k-237_concept_domain', type=str)

    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=1000, type=int)
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=10000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=10000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args()


def _set_logger(args):
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    args = _set_args()

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    _set_logger(args)

    with open(os.path.join(args.save_path, 'entities.dict'), 'r') as f:
        entity2id = json.load(f)
    with open(os.path.join(args.save_path, 'relations.dict'), 'r') as f:
        relation2id = json.load(f)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    with open(os.path.join(args.save_path, 'train_triples.pkl'), 'rb') as f:
        train_triples = pickle.load(f)
    logging.info('#train: %d' % len(train_triples))
    with open(os.path.join(args.save_path, 'valid_triples.pkl'), 'rb') as f:
        valid_triples = pickle.load(f)
    logging.info('#valid: %d' % len(valid_triples))
    with open(os.path.join(args.save_path, 'test_triples.pkl'), 'rb') as f:
        test_triples = pickle.load(f)
    logging.info('#test: %d' % len(test_triples))

    all_true_triples = train_triples + valid_triples + test_triples
    
