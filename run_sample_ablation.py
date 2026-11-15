import numpy as np, pandas as pd, sys
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

OUT = open(str(__import__("pathlib").Path(__file__).resolve().parent) + r'\results\predictor_v2\sample_size_ablation.log', 'w', encoding='utf-8')
def p(msg):
    print(msg)
    OUT.write(msg + '\n')
    OUT.flush()

p('Loading embeddings...')
all_var = pd.read_csv(str(__import__("pathlib").Path(__file__).resolve().parent) + r'\data\processed\all_variants.csv')
var2idx = {v:i for i,v in enumerate(all_var.variant)}
emb = np.load(str(__import__("pathlib").Path(__file__).resolve().parent) + r'\data\esm2_embeddings\esm2_pooled_L33.npy')
p(f'{len(all_var)} variants, emb {emb.shape}')

AA='ACDEFGHIKLMNPQRSTVWY'
def oh(vs):
    m=np.zeros((len(vs),80))
    for r,v in enumerate(vs):
        for p_,a in enumerate(v): m[r,20*p_+AA.index(a)]=1
    return m

results=[]
for ts in [2000, 10000, 50000]:
    sps={'rf':[], 'mlp':[], 'ridge':[]}
    for seed in [11,22,33]:
        rng=np.random.RandomState(seed)
        idx=rng.permutation(len(all_var))
        tr,va=idx[:ts],idx[ts:ts+2000]
        trv=all_var.variant.iloc[tr].values; vav=all_var.variant.iloc[va].values
        trf=all_var.fitness.iloc[tr].values; vaf=all_var.fitness.iloc[va].values
        tr_rows=[var2idx[v] for v in trv]; va_rows=[var2idx[v] for v in vav]
        Xe_tr=emb[tr_rows]; Xe_va=emb[va_rows]
        
        rf=RandomForestRegressor(n_estimators=100,min_samples_leaf=1,random_state=seed,n_jobs=4).fit(oh(trv),trf)
        sps['rf'].append(spearmanr(vaf,rf.predict(oh(vav))).statistic)
        
        mlp=Pipeline([('s',StandardScaler()),('m',MLPRegressor(hidden_layer_sizes=(256,64),max_iter=300,alpha=1e-4,early_stopping=True,random_state=seed))]).fit(Xe_tr,trf)
        sps['mlp'].append(spearmanr(vaf,mlp.predict(Xe_va)).statistic)
        
        rg=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=10))]).fit(Xe_tr,trf)
        sps['ridge'].append(spearmanr(vaf,rg.predict(Xe_va)).statistic)
        p(f'  seed {seed} done')
    
    p(f'train={ts:5d}  RF={np.mean(sps["rf"]):.4f}+-{np.std(sps["rf"]):.4f}  MLP={np.mean(sps["mlp"]):.4f}+-{np.std(sps["mlp"]):.4f}  Ridge={np.mean(sps["ridge"]):.4f}+-{np.std(sps["ridge"]):.4f}')
    
    for k in sps:
        for s,v in zip([11,22,33],sps[k]):
            results.append({'train_size':ts,'seed':s,'method':'rf_onehot' if k=='rf' else f'esm2_{k}','val_spearman':round(v,4)})

pd.DataFrame(results).to_csv(str(__import__("pathlib").Path(__file__).resolve().parent) + r'\results\predictor_v2\sample_size_ablation.csv',index=False)
p('DONE')
OUT.close()
