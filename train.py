import argparse
import logging
import os
import json
import pickle
from data_utils import FB17K_237
from KGE import KGEModel
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


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


def save_model(model, optimizer, save_variable_list, args):

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def _test(model, test_loader, args, num):
    model.eval()
    logs = []
    step = 0

    with torch.no_grad():
        if num == 0:
            for batch in test_loader:
                positive_sample, negative_sample, subsampling_weight = batch
                if args.cuda:
                    if args.cuda:
                        negative_sample = negative_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        subsampling_weight = subsampling_weight.cuda()
                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), 'head-batch')
                    score += subsampling_weight
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    positive_arg = positive_sample[:, 0]
                    for i in range(batch_size):
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })
                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps/2))
                        step += 1



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

    train_set = FB17K_237(args=args, type='train')
    logging.info('#train: %d' % len(train_set))
    valid_set = FB17K_237(args=args, type='valid')
    logging.info('#valid: %d' % len(valid_set))
    test_set = FB17K_237(args=args, type='test')
    logging.info('#test: %d' % len(test_set))

    model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma
    )
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        model = model.cuda()

    if args.do_train:
        train_loader = DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=args.batch_size
        )
    if args.do_test:
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.test_batch_size
        )

    current_learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=current_learning_rate
    )
    if args.warm_up_steps:
        warm_up_steps = args.warm_up_steps
    else:
        warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))

    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)


    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)
        training_logs = []

        for eps in range(10):
            for batch in tqdm(train_loader):
                model.train()
                optimizer.zero_grad()

                positive_sample, negative_sample, subsampling_weight = batch

                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    subsampling_weight = subsampling_weight.cuda()

                negative_score = model((positive_sample, negative_sample), mode='head-batch')
                if args.negative_adversarial_sampling:
                    negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
                else:
                    negative_score = F.logsigmoid(-negative_score).mean(dim=1)
                positive_score = model(positive_sample)
                positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

                if args.uni_weight:
                    positive_sample_loss = - positive_score.mean()
                    negative_sample_loss = - negative_score.mean()
                else:
                    positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                    negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

                loss = (positive_sample_loss + negative_sample_loss)/2

                if args.regularization != 0.0:
                    #Use L3 regularization for ComplEx and DistMult
                    regularization = args.regularization * (
                        model.entity_embedding.norm(p = 3)**3 +
                        model.relation_embedding.norm(p = 3).norm(p = 3)**3
                    )
                    loss = loss + regularization
                    regularization_log = {'regularization': regularization.item()}
                else:
                    regularization_log = {}

                loss.backward()
                optimizer.step()

                log = {
                    **regularization_log,
                    'positive_sample_loss': positive_sample_loss.item(),
                    'negative_sample_loss': negative_sample_loss.item(),
                    'loss': loss.item()
                }
                training_logs.append(log)
                if step >= warm_up_steps:
                    current_learning_rate = current_learning_rate / 10
                    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=current_learning_rate
                    )
                    warm_up_steps = warm_up_steps * 3

                if step % args.save_checkpoint_steps == 0:
                    save_variable_list = {
                        'step': step,
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
                    save_model(model, optimizer, save_variable_list, args)

                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                    log_metrics('Training average', step, metrics)
                    training_logs = []

                if args.do_valid and step % args.valid_steps == 0:
                    logging.info('Evaluating on Valid Dataset...')
                    metrics = _test(model=model, test_loader=test_loader, args=args, num=0)
                    log_metrics('Valid', step, metrics)

                step += 1

