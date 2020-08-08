import torch
import argparse
import pdb
import sys
import json
from collections import defaultdict
from beam_search import word_path_topk
from word_process import process
 
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
if args.m_type == 'gpt2':
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
#elif args.m_type == 'roberta':
#    from transformers import RobertaTokenizer, RobertaForMaskedLM 
#    tokenizer = RobertaTokenizer.from_pretrained(args.model)
#    model = RobertaForMaskedLM.from_pretrained(args.model)
#elif args.m_type == 'bert':
#    from transformers import BertTokenizer, BertForMaskedLM 
#    tokenizer = BertTokenizer.from_pretrained(args.model)
#    model = BertForMaskedLM.from_pretrained(args.model)
elif args.m_type == 'gpt':
    from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
    model = OpenAIGPTLMHeadModel.from_pretrained(args.model)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model)
    # load int2token instead
else:
    sys.exit()

if args.m_type == 'gpt2': space='Ġ'
if args.m_type == 'gpt': space='</w>'

# freeze model weights
for _, param in model.named_parameters():
    param.requires_grad = False

total = 0
number=0
t_ppx = 0
types1 = defaultdict(int)
types10 = defaultdict(int)
soft_match = defaultdict(lambda: defaultdict(int))

# load sentence by sentece
for context in open(args.testdata, 'r').readlines():

    # evaluate on longer than a single token sentence
    if args.m_type== 'gpt': context = context.lower()
    context = context.strip()
    words = context.split()
    if len(words) < 3:
        continue
    total+=len(words)-1
    # iterate over word prediction
    for i in range(1,len(words)):
        ctxt = " ".join(words[:i])
        #print(ctxt) 
        input_ids = tokenizer.encode(ctxt, return_tensors='pt')
        # spin until a complete word is spelled
        unfinished = 1
        pr = 1
        cnt=0
        seq = []
        while unfinished:# while loop for spinning 
            logits = model(input_ids)[0] # final input rmed
            logits = logits[0,-1,:]
            if not cnt: topk = logits # save them for top10
            prs = torch.nn.functional.softmax(logits, dim=0)
            argmax = torch.argmax(prs) 
            argmax = argmax.view(-1)
            pr*=prs[argmax]
            # decode
            token = tokenizer._convert_id_to_token(int(argmax))
            seq.append(token)
            # is there a space for word boundary?
            if (args.m_type == 'gpt' and space in token): break
            if (args.m_type == 'gpt2' and ((space in token and cnt>0) or token == 'Ċ')): break
            # concat to input_ids the new token
            argmax = argmax.view(1,1)
            input_ids = torch.cat((input_ids, argmax), 1)
            cnt+=1
            if cnt > 6: break # avoid infinite loop
        # process word
        pred1 = process(seq, space)

        if pred1 == words[i]:
            types1[pred1]+=1
            types10[pred1]+=1
            t_ppx+=pr
            continue
        else: # BFS
            soft_match[words[i]][pred1]+=1
            status = word_path_topk(model, tokenizer, topk, ctxt, words[i], args.m_type)
            if status == 'found':
                #pdb.set_trace()
                #print('UPDATE TOP 10\n')
                types10[words[i]]+=1

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
