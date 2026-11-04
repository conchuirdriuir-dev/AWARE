"""Step 3, after its run (agent, reported, not judged): the pooled out-of-fold AUC against per-fold, within-conversation
and shuffled-label versions. python -m research.source_check runs/source-qwen8b"""
import numpy as np
from pathlib import Path
from substrate.record import read
from research import source_read as s
import sys
run=Path(sys.argv[1] if len(sys.argv)>1 else "runs/source-qwen8b")
rows=[{"id":r["conversation"],"samples":r["samples"]} for r in read(run/"record.jsonl") if r["kind"]=="read"]
st=s.load_states(run,[r["id"] for r in rows])
items=[(r["id"],i,x) for r in rows for i,x in enumerate(r["samples"]) if x["key"]]
g=np.array([a for a,_,_ in items]); y=np.array([x["key"]=="partner" for *_,x in items])
feats={"fit":(np.array([[x["F"]["logprob"]] for *_,x in items]),None),
 "text":(np.array([[x["F"]["hits"],x["F"]["length"],x["F"]["first_hit_token"]] for *_,x in items],float),None),
 "layer 18":(np.array([st[a]["18"][i] for a,i,_ in items]),20)}
f=s.folds(list(g))
def within(sc,y,g):
    num=den=0
    for c in set(g):
        m=g==c; a,b=sc[m&~y],sc[m&y]
        if len(a) and len(b): num+=s.auc(np.r_[a,b],np.r_[np.zeros(len(a),bool),np.ones(len(b),bool)])*len(a)*len(b); den+=len(a)*len(b)
    return num/den
print("own share per fold:",[round(float((~y[f==k]).mean()),3) for k in range(5)])
for name,(X,comp) in feats.items():
    sc=s.out_of_fold(X,y,list(g),comp)
    perfold=[s.auc(sc[f==k],y[f==k]) for k in range(5)]
    print(name,"pooled",round(s.auc(sc,y),3),"per fold",[round(v,3) for v in perfold],"within-conversation",round(within(sc,y,g),3))
rng=np.random.default_rng(0); null=[]
X,comp=feats["layer 18"]
for _ in range(20):
    yp=y.copy()
    for c in set(g):
        m=np.where(g==c)[0]; yp[m]=rng.permutation(y[m])
    null.append(s.auc(s.out_of_fold(X,yp,list(g),comp),yp))
print("null (labels shuffled within conversation), layer 18 pooled AUC: mean",round(np.mean(null),3),"range",round(min(null),3),round(max(null),3))
# raw, untrained: hits and first-hit position, within conversation
for k in ("hits","first_hit_token","length","logprob"):
    v=np.array([x["F"][k] for *_,x in items],float); print("raw",k,"within-conv AUC (partner high)",round(within(v,y,g),3))
