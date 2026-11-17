# A/A noise study: A and B are the same binary. For each model:
#  - median eval of A, B, C (OpenMP build)
#  - B/A ratio of medians and a 95% bootstrap CI for it
#  - would the review's gate (fail if CI lower bound > +3%) or the old gate
#    (median delta > +3%) have fired on an A/A pair?
import re,sys,random,statistics,collections
random.seed(1)
d=collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(sys.argv[1]):
    m=re.match(r'(\d+) ([ABC]) (\S+) .*Evaluate time: ([\d.]+)',l)
    if m: d[m[3]][m[2]].append(float(m[4]))
def ci(a,b,n=4000):
    rs=[]
    for _ in range(n):
        ra=[random.choice(a) for _ in a]; rb=[random.choice(b) for _ in b]
        rs.append(statistics.median(rb)/statistics.median(ra))
    rs.sort(); return rs[int(.025*n)], rs[int(.975*n)]
old=new=0; ratios=[]; omp=[]
print(f"{'model':26s} {'n':>2} {'A med':>7} {'B/A':>6} {'95% CI':>15} {'spread%':>7} {'OMP med':>8} {'OMP/A':>6}")
for k,v in d.items():
    a,b,c=v['A'],v['B'],v['C']
    r=statistics.median(b)/statistics.median(a); lo,hi=ci(a,b); ratios.append(r)
    spread=100*(max(a)-min(a))/statistics.median(a)
    old+= r>1.03; new+= lo>1.03
    oc=statistics.median(c)/statistics.median(a); omp.append((k,oc))
    print(f"{k:26s} {len(a):>2} {statistics.median(a):7.2f} {r:6.3f} [{lo:.3f},{hi:.3f}] {spread:7.1f} {statistics.median(c):8.2f} {oc:6.2f}")
g=1
for r in ratios: g*=r
print(f"A/A: old gate (median delta>3%) fired on {old}/{len(ratios)}; CI-lower-bound gate fired on {new}/{len(ratios)}; geomean B/A = {g**(1/len(ratios)):.4f}")
print("max |B/A-1| = %.3f" % max(abs(r-1) for r in ratios))
