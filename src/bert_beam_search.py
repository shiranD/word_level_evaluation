import torch
import re
import pdb

def to_string(word, m_type):
    if m_type == 'roberta':
        return ''.join(word)
    elif m_type == 'bert':
        return re.sub('##', '', ''.join(word))

def check(pfx, tgt, m_type, space):

    if m_type  == 'roberta' and space in pfx: pfx = pfx.strip(space) # rm first space

    if space in pfx: # space indicates completion of word
        pfx_c = pfx.split(space)[0]
        if pfx_c == tgt: return 'same'

    if pfx == tgt: return 'same' # no space after completion
    elif len(pfx) < len(tgt): # incompletion of word is containment 
        if pfx == tgt[:len(pfx)]: return 'contained'

def word_path_topk(model, tokenizer, topks, ctxt, target, model_keys, k=10):

    top_seeds = []
    ctxt_ids = tokenizer.encode(ctxt, return_tensors='pt') # ctxt ids
    input_ids = ctxt_ids[:,:-2].view(1,-1) # rm mask, eos
    _, topk_ind = torch.topk(topks, k, dim=0, sorted=True) # topk seeds
    #pdb.set_trace()
    for rank in range(k):

        ind = int(topk_ind[rank])
        prefix = tokenizer._convert_id_to_token(ind)
        #print(prefix, target)
        if model_keys['type'] =='roberta' and prefix == model_keys['space']: continue
        if model_keys['type'] == 'bert' and model_keys['sfx'] in prefix: continue

        relation = check(prefix, target, model_keys['type'], model_keys['space'])
        if relation == 'same': return 'found'
        if relation == 'contained': top_seeds.append(ind)

    if top_seeds: # an appropirate seed was not found
        #pdb.set_trace()
        for root in top_seeds:
            path = [str(int(root))]

            while path:
                node = path.pop(0)
                if len(node.split('_'))>5: continue # too long for current seqs, ignore node, focus on other paths

                prefix = [] # prefix is a seq at this point (and not just a token)
                for ind in node.split('_'): # prepare prefix as input (to generate top k)
                    ind = int(ind)
                    prefix.append(tokenizer._convert_id_to_token(ind))
                    ind = torch.tensor(ind).view(1,1)
                    input_ids = torch.cat((input_ids, ind), 1)
                input_ids = torch.cat((input_ids, model_keys['mask_eos']), dim=1) # add mask, eos

                prefix_s = to_string(prefix, model_keys['type'])

                logits = model(input_ids, labels=input_ids)[1] # final input rmed
                logits = logits[0,-2,:] # -2 due to masking one b4 last
                _, topk_ind = torch.topk(logits, k, dim=0, sorted=True) # topk
    
                for new_rank in range(k):
                    ind = int(topk_ind[new_rank])
                    suffix = tokenizer._convert_id_to_token(ind)
                    #print(suffix)
                    if model_keys['type'] == 'bert' and model_keys['sfx'] not in suffix: continue

                    word = to_string([prefix_s,suffix], model_keys['type'])
                    relation = check(word, target, model_keys['type'], model_keys['space'])
                    if relation == 'same': return 'found'
                    if relation == 'contained':
                        new_node = node+'_'+str(int(ind))
                        path.append(new_node) # root nodes to start a search
