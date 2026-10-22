"""Independent audit of saved Christoffersen test CSV. No evaluator imports.
Usage: python validate_christoffersen_independent.py evaluation_results/christoffersen_tests.csv
"""
import sys, math
import pandas as pd
import numpy as np

def ll(k,n,p):
    if n == 0: return 0.
    if p == 0: return 0. if k == 0 else -math.inf
    if p == 1: return 0. if k == n else -math.inf
    return k*math.log(p)+(n-k)*math.log1p(-p)

def sf1(x): return math.erfc(math.sqrt(x/2))
def sf2(x): return math.exp(-x/2)

path = sys.argv[1] if len(sys.argv)>1 else 'evaluation_results/christoffersen_tests.csv'
df = pd.read_csv(path)
issues=[]; checked=0; missing=0
for i,r in df.iterrows():
    tag=f"row {i+2} {r.Model} {r.Period} {r.Interval_Level_Percent}%"
    n=int(r.Observations); k=int(r.Violations); p0=1-float(r.Nominal_Coverage_Percent)/100
    counts=[int(r[c]) for c in ['N00','N01','N10','N11']]
    n00,n01,n10,n11=counts
    if sum(counts)!=n-1: issues.append(f'{tag}: transition counts do not sum to n-1')
    if abs(100*(1-k/n)-float(r.Observed_Coverage_Percent))>1e-7: issues.append(f'{tag}: coverage mismatch')
    uc=max(0,2*(ll(k,n,k/n)-ll(k,n,p0)))
    if not np.isclose(uc,r.LRuc,atol=1e-7,rtol=1e-9): issues.append(f'{tag}: LRuc mismatch')
    if not np.isclose(sf1(uc),r.LRuc_p,atol=1e-12,rtol=1e-7): issues.append(f'{tag}: LRuc p mismatch')
    if pd.isna(r.LRind): missing+=1; continue
    if not (n00+n01 and n10+n11): issues.append(f'{tag}: LRind reported without both states'); continue
    ntr=n-1; m=n01+n11
    ind=max(0,2*(ll(n01,n00+n01,n01/(n00+n01))+ll(n11,n10+n11,n11/(n10+n11))-ll(m,ntr,m/ntr)))
    cc=uc+ind
    for col,value in [('LRind',ind),('LRcc',cc)]:
        if not np.isclose(value,r[col],atol=1e-7,rtol=1e-9): issues.append(f'{tag}: {col} mismatch')
    for col,value in [('LRind_p',sf1(ind)),('LRcc_p',sf2(cc))]:
        if not np.isclose(value,r[col],atol=1e-12,rtol=1e-7): issues.append(f'{tag}: {col} mismatch')
    for col,pval in [('LRuc_Reject',sf1(uc)),('LRind_Reject',sf1(ind)),('LRcc_Reject',sf2(cc))]:
        if str(r[col]).lower() != str(pval<0.05).lower(): issues.append(f'{tag}: {col} mismatch')
    checked+=1
print(f'Rows in CSV: {len(df)}; complete independently checked: {checked}; LRind missing: {missing}')
print(f'Discrepancies: {len(issues)}')
for item in issues[:30]: print('FAIL:',item)
print('PASS: saved Christoffersen statistics internally consistent' if not issues else 'FAIL: see above')
sys.exit(bool(issues))
