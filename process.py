import os
import json
import pickle


def count_frequency(triples, start=4):
    count = {}
    for head, relation, tail in triples:
        if (head, relation) not in count:
            count[(head, relation)] = start
        else:
            count[(head, relation)] += 1

        if (tail, -relation-1) not in count:
            count[(tail, -relation-1)] = start
        else:
            count[(tail, -relation-1)] += 1
    return count


def concept_filter_h(head, relation, rel_h, rel2nn):
    if str(relation) not in rel_h:
        return []
    rel_hc = rel_h[str(relation)]
    set_hc = set(rel_hc)
    h = []
    if rel2nn[str(relation)] == 0 or rel2nn[str(relation)] == 1:
        ...
        #if str(head)


# process to get negative samples
save_path = 'models/TransE_FB15k-237_concept_domain'
data_path = 'data_concept/FB15k-237_concept/'

with open(os.path.join(save_path, 'entities.dict'), 'r') as f:
    entity2id = json.load(f)
with open(os.path.join(save_path, 'relations.dict'), 'r') as f:
    relation2id = json.load(f)

with open(os.path.join(data_path, 'rel2dom_h.json')) as fin:
    rel2dom_h = json.load(fin)
with open(os.path.join(data_path, 'rel2nn.json')) as fin:
    rel2nn = json.load(fin)
rel_h = rel2dom_h
rel2nn = rel2nn

# with open(os.path.join(save_path, 'train_triples.pkl'), 'rb') as f:
#     train_triples = pickle.load(f)
# with open(os.path.join(save_path, 'valid_triples.pkl'), 'rb') as f:
#     valid_triples = pickle.load(f)
with open(os.path.join(save_path, 'test_triples.pkl'), 'rb') as f:
    test_triples = pickle.load(f)

count = count_frequency(test_triples)

for trip in test_triples:
    head, relation, tail = trip
    subsampling_weight = count[(head, relation)] + count[(tail, -relation-1)]

    negative_sample_list = []
    negative_sample_size = 0
