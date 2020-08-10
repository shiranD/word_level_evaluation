import torch
import argparse
import pdb
import sys
import json
import math
import re
from collections import defaultdict
from word_process import process
from bert_beam_search import word_path_topk

###############################################################################
# Parsing Arguments
###############################################################################

parser = argparse.ArgumentParser(description='PyTorch WGAN Language Model Game')
parser.add_argument('--model', type=str, help='path to save the final Discriminator model')
parser.add_argument('--output_dir', type=str, help='path to save the final Discriminator model')
parser.add_argument('--testdata', type=str, help='location of the data corpus')
parser.add_argument('--m_type', type=str, help='location of the data corpus')
args = parser.parse_args()
print(args.m_type)
#https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf

if args.m_type == 'roberta':
    from transformers import RobertaTokenizer, RobertaForMaskedLM 
    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    model = RobertaForMaskedLM.from_pretrained(args.model)
    model_keys = {'type': 'roberta', 'space': 'Ġ', 'eos': '</s>', 'mask': '<mask>', 'nline': 'Ċ'}

elif args.m_type == 'bert':
    from transformers import BertTokenizer, BertForMaskedLM 
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BertForMaskedLM.from_pretrained(args.model)
    model_keys = {'type': 'bert', 'space': '####','sfx': '##', 'eos': '[SEP]', 'mask': '[MASK]'}
else:
    sys.exit()

# freeze model weights
for _, param in model.named_parameters():
    param.requires_grad = False

mask_id = tokenizer._convert_token_to_id(model_keys['mask'])
eos_id = tokenizer._convert_token_to_id(model_keys['eos'])
model_keys['mask_eos'] = torch.tensor([mask_id,eos_id]).view(1,2)

b=1
total = 0
number=0
t_ppx = 0
types1 = defaultdict(int)
types10 = defaultdict(int)
soft_match = defaultdict(lambda: defaultdict(int))

# load sentence by sentece
for context in open(args.testdata, 'r').readlines():

    context = context.strip()
    words = context.split()
    if len(words) < 3:
        continue
    total+=len(words)-1
    # iterate over word prediction
    for i in range(1,len(words)):
        ctxt = " ".join(words[:i]+[model_keys['mask']])
        input_ids = tokenizer.encode(ctxt, return_tensors='pt')
        # spin until a complete word is spelled
        unfinished = 1
        pr = 1
        cnt=0
        seq = []
        #pdb.set_trace()
        while unfinished:# while loop for spinning 
            # mask one before [SEP] label

            logits = model(input_ids, labels=input_ids)[1] # final input rmed
            logits = logits[0,-2,:]
            if not cnt: topk = logits
            prs = torch.nn.functional.softmax(logits, dim=0)
            argmax = torch.argmax(prs) 
            argmax = argmax.view(-1)
            pr*=prs[argmax]
            # decode
            token = tokenizer._convert_id_to_token(int(argmax))

            # is there a space for word boundary?
            if model_keys['type'] == 'bert':
                if model_keys['sfx'] not in token and cnt > 0: break # leave before adding a new token (finish a word)
            seq.append(token)
            if model_keys['type'] == 'roberta': # leave after adding a new token (word boundary is there)
                if model_keys['space'] in token and cnt>0: break
                if model_keys['nline'] in token: break

            # concat to input_ids the new token
            argmax = argmax.view(1,1)
            input_ids = torch.cat((input_ids[:,:-2].view(1,-1),argmax), dim=1)
            input_ids = torch.cat((input_ids, model_keys['mask_eos']), dim=1)
            cnt+=1
            if cnt > 5: break

        # process word
        if model_keys['type'] == 'bert': pred1 = re.sub(model_keys['sfx'],"", "".join(seq))
        if model_keys['type'] == 'roberta': pred1 = process(seq, model_keys['space'])

        if pred1 == words[i]:
            types1[pred1]+=1
            types10[pred1]+=1
            t_ppx+=-math.log(pr)
        else: # BFS
            soft_match[words[i]][pred1]+=1
            pr = word_path_topk(model, tokenizer, topk, ctxt, words[i], model_keys)
            if isinstance(pr, torch.Tensor):
                types10[words[i]]+=1
                t_ppx+=-math.log(pr)
            else:
                t_ppx+=100


    # dump to json
    if total > 1000:
        d = {'total': total, 'type1': types1, 'type10': types10, 'ppx': float(t_ppx), 'soft_match': soft_match}
        with open(args.output_dir+'/results/results_'+str(number), 'w') as fp:
            json.dump(d, fp)
        number+=1
        total=0
        types1 = defaultdict(int)
        types10 = defaultdict(int)
        t_ppx = 0
        soft_match = defaultdict(lambda: defaultdict(int))

d = {'total': total, 'type1': types1, 'type10': types10, 'ppx': float(t_ppx), 'soft_match': soft_match}
with open(args.output_dir+'/results/results_'+str(number), 'w') as fp:
    json.dump(d, fp)
