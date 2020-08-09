import os
import sys
import argparse
import json
import pdb
from collections import defaultdict

parser = argparse.ArgumentParser(description='Process results')
parser.add_argument('--folder', type=str, help='location of the results folder')
parser.add_argument('--number', type=int, help='file number to process')
parser.add_argument('--termdir', type=str, help='directory for old, new, and shared terms')
parser.add_argument('--sett', type=str, help='test set kwd')
args = parser.parse_args()

with open(args.termdir+'high_'+args.sett, 'r') as fp:
    high, highf = json.load(fp)

with open(args.termdir+'mid1_'+args.sett, 'r') as fp:
    mid1, mid1f = json.load(fp)
    
with open(args.termdir+'mid2_'+args.sett, 'r') as fp:
    mid2, mid2f = json.load(fp)

with open(args.termdir+'low_'+args.sett, 'r') as fp:
    low, lowf = json.load(fp)
#print(sharedf, newf, oldf)

total, ppx, top1, top10 = 0, 0, 0, 0
types1, types10 = defaultdict(int), defaultdict(int)

high_found, mid1_found, mid2_found, low_found = {}, {}, {}, {}
#mid2_found = defaultdict(int)

sum_high, sum_mid1, sum_mid2, sum_low = 0, 0, 0, 0

for i in range(args.number):
    i = str(i)

    exists = os.path.isfile(args.folder+'/results/results_'+i)
    
    if not exists:
        continue 
    fname = args.folder+'/results/results_'+i
    with open(fname, 'r') as fp:
        d = json.load(fp)
        type1 = d["type1"]
        type10 = d["type10"]
        print(total)
        print(fname)

    for key in type1.keys():
        if key in high:
            sum_high+=type1[key]
            high_found[key]=True
            types1[key]+=type1[key]
        elif key in mid1:
            sum_mid1+=type1[key]
            mid1_found[key]=True
            types1[key]+=type1[key]
        elif key in mid2:
            sum_mid2+=type1[key]
            mid2_found[key]=True
            types1[key]+=type1[key]
        elif key in low:
            sum_low+=type1[key]
            low_found[key]=True
            types1[key]+=type1[key]
        else:
           pass
           #print(key)

    for key in type10.keys():
        if key in high:
            types10[key]+=type10[key]
        elif key in mid1:
            types10[key]+=type10[key]
        elif key in mid2:
            types10[key]+=type10[key]
        elif key in low:
            types10[key]+=type10[key]
        else:
           pass
           #print(key)

    total+=d["total"]
    ppx+=d["ppx"]

f = open(args.folder+'/final', 'w')
my_types=len(high.keys())+len(mid1.keys())+len(mid2.keys())
f.write('total is {}\n'.format(total)) 
f.write('top10 {}\n'.format(sum(types10.values())/total*100)) 
f.write('top1 {}\n'.format(sum(types1.values())/total*100)) 
f.write('ppx {}\n'.format(ppx/total)) 
f.write('types10 {}\n'.format(len(types10)))
f.write('types1 {}\n'.format(len(types1)))
f.write('& {:2.2f}({:2.2f}) & {:2.2f}({:2.2f}) & {:1.3f} \\\\ \n'.format(sum(types1.values())/total*100, sum(types10.values())/total*100, len(types1.keys())/my_types*100, len(types10.keys())/my_types*100, ppx/total))
# types (partial)
f.write('high {}\n'.format(len(high_found)/len(high)*100))
f.write('mid1 {}\n'.format(len(mid1_found)/len(mid1)*100))
f.write('mid2 {}\n'.format(len(mid2_found)/len(mid2)*100))
f.write('low {}\n'.format(len(low_found)/len(low)*100))
#print(mid2_found)
# events (partial)
f.write('high {}\n'.format(sum_high/highf*100))
f.write('mid1 {}\n'.format(sum_mid1/mid1f*100))
f.write('mid2 {}\n'.format(sum_mid2/mid2f*100))
f.write('low {}\n'.format(sum_low/lowf*100))
#print(types1.keys())
print('\n')
print(total, top10, top1)
print(highf,mid1f,mid2f,lowf)
print(len(high), len(mid1), len(mid2), len(low))
#print(types10.keys())
