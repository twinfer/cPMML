# Stats after docs/design/typed-core/aab_stats.py for a 4-way run:
# A base seq, B head seq, C base OpenMP, D head OpenMP.
# Ratios of medians with 95% bootstrap CIs (4000 resamples). Gate (§6.4):
# fail when the CI lower bound > 1.03. Geomean with joint (by-round) bootstrap.
import re,sys,random,statistics,collections
random.seed(1)
d=collections.defaultdict(lambda: collections.defaultdict(dict))
for l in open(sys.argv[1]):
    m=re.match(r'(\d+) ([ABCD]) (\S+) .*Evaluate time: ([\d.]+)',l)
    if m: d[m[3]][m[2]][int(m[1])]=float(m[4])
med=statistics.median
def ci(a,b,n=4000):
    rs=sorted(med([random.choice(b) for _ in b])/med([random.choice(a) for _ in a]) for _ in range(n))
    return rs[int(.025*n)], rs[int(.975*n)]
pairs=[('B','A','seq head/base'),('D','C','omp head/base'),('D','B','head omp/seq'),('C','A','base omp/seq')]
print(f"{'model':26s} {'n':>2} {'A seq0':>7} {'B seq1':>7} {'C omp0':>7} {'D omp1':>7}  " + "  ".join(f"{t:>26s}" for _,_,t in pairs))
fails=collections.Counter(); geo=collections.defaultdict(list)
for k,v in d.items():
    row=f"{k:26s} {len(v['A']):>2} " + " ".join(f"{med(v[x].values()):7.2f}" for x in 'ABCD') + "  "
    for h,b,t in pairs:
        r=med(v[h].values())/med(v[b].values()); lo,hi=ci(list(v[b].values()),list(v[h].values()))
        flag='!' if lo>1.03 else ' '; fails[t]+= lo>1.03; geo[t].append((k,r))
        row+=f"  {r:6.3f} [{lo:.3f},{hi:.3f}]{flag}       "
    print(row)
def joint_geo(h,b,n=2000):
    models=list(d); rounds=sorted(set.intersection(*[set(d[m][h])&set(d[m][b]) for m in models]))
    def g(rs):
        p=1
        for m in models: p*=med([d[m][h][r] for r in rs])/med([d[m][b][r] for r in rs])
        return p**(1/len(models))
    bs=sorted(g([random.choice(rounds) for _ in rounds]) for _ in range(n))
    return g(rounds), bs[int(.025*n)], bs[int(.975*n)]
for h,b,t in pairs:
    gm,lo,hi=joint_geo(h,b)
    print(f"{t:14s}: per-model gate (CI low>1.03) fired on {fails[t]}/{len(d)}; geomean {gm:.4f} [{lo:.4f},{hi:.4f}]")
