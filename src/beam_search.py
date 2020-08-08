import torch
import pdb

def to_string(word):
    return ''.join(word)


def check(pfx, tgt, m_type, space):

    if m_type  == 'gpt2' and space in pfx: pfx = pfx.strip(space) # rm first space

    if space in pfx: # space indicates completion of word
        pfx_c = pfx.split(space)[0]
        if pfx_c == tgt: return 'same'
    if pfx == tgt: return 'same' # no space after completion
    elif len(pfx) < len(tgt): # incompletion of word is containment 
        if pfx == tgt[:len(pfx)]: return 'contained'

def word_path_topk(model, tokenizer, topks, ctxt, target, model_type, k=10):

    if model_type == 'gpt2': space='Ġ'
    if model_type == 'gpt': space='</w>'

    top_seeds = []
    ctxt_ids = tokenizer.encode(ctxt, return_tensors='pt') # ctxt ids
    _, topk_ind = torch.topk(topks, k, dim=0, sorted=True) # topk seeds
    for rank in range(k):

        ind = int(topk_ind[rank])
        prefix = tokenizer._convert_id_to_token(ind)
        if model_type =='gpt2' and prefix == space: continue

        relation = check(prefix, target, model_type, space)
        if relation == 'same': return 'found'
        if relation == 'contained': top_seeds.append(ind)
    if top_seeds: # an appropirate seed was not found
        same = 0 
        for root in top_seeds:
            path = [str(int(root))]
            while path:
                node = path.pop(0)
                if len(node.split('_'))>5: continue # too long for current seqs, ignore node, focus on other paths
                input_ids = ctxt_ids
                prefix = [] # prefix is a seq at this point (and not just a token)
                for ind in node.split('_'): # prepare prefix as input (to generate top k)
                    ind = int(ind)
                    prefix.append(tokenizer._convert_id_to_token(ind))
                    ind = torch.tensor(ind).view(1,1)
                    input_ids = torch.cat((input_ids, ind), 1)

                prefix_s = to_string(prefix)
                logits = model(input_ids)[0] # final input rmed
                logits = logits[0,-1,:]
                _, topk_ind = torch.topk(logits, k, dim=0, sorted=True) # topk
    
                for new_rank in range(k):
                    ind = int(topk_ind[new_rank])
                    suffix = tokenizer._convert_id_to_token(ind)
                    word = to_string([prefix_s,suffix])
                    relation = check(word, target, model_type, space)
                    if relation == 'same': return 'found'
                    if relation == 'contained':
                        new_node = node+'_'+str(int(ind))
                        path.append(new_node) # root nodes to start a search
