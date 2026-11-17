# For every model, find document text typed by a field's dataType that is not
# in that type's lexical space (integer/boolean/float/double fields).
import glob,re,sys,zipfile,collections
import xml.etree.ElementTree as ET
import subprocess as _sp; R=_sp.check_output(['git','rev-parse','--show-toplevel'],cwd=__import__('os').path.dirname(__import__('os').path.abspath(__file__)),text=True).strip()
INT=re.compile(r'^\s*[+-]?\d+\s*$')
DBL=re.compile(r'^\s*([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?|-?INF|NaN)\s*$')
BOOL={'true','false','1','0'}
def ln(t): return t.rsplit('}',1)[-1]
def docs():
    for f in sorted(glob.glob(R+'/test/data/model/*.zip')):
        with zipfile.ZipFile(f) as z:
            for n in z.namelist():
                if n.endswith(('.pmml','.xml')): yield f.split('/')[-1], z.read(n)
    for f in sorted(glob.glob(R+'/benchmark/data/model/*.xml')): yield 'bench/'+f.split('/')[-1], open(f,'rb').read()
def ok(t,v):
    if t=='integer': return bool(INT.match(v))
    if t in('float','double'): return bool(DBL.match(v))
    if t=='boolean': return v.strip() in BOOL
    return True
hits=collections.Counter(); ex={}
for name,data in docs():
    try: root=ET.fromstring(data)
    except Exception as e: continue
    types={}
    for el in root.iter():
        if ln(el.tag) in('DataField','DerivedField') and el.get('name'): types[el.get('name')]=el.get('dataType')
    def chk(kind,field,val):
        t=types.get(field)
        if t and val is not None and not ok(t,val):
            k=(kind,t, 'numeric' if DBL.match(val) else 'non-numeric'); hits[k]+=1; ex.setdefault(k,set()).add(f'{name}: {field}={val!r}')
    for el in root.iter():
        tag=ln(el.tag)
        if tag=='SimplePredicate': chk('SimplePredicate@value',el.get('field'),el.get('value'))
        if tag=='SimpleSetPredicate':
            for a in el:
                if ln(a.tag)=='Array' and a.text:
                    for m in re.findall(r'"((?:[^"\\]|\\.)*)"|(\S+)',a.text): chk('Array member',el.get('field'),m[0] or m[1])
        if tag=='CategoricalPredictor': chk('CategoricalPredictor@value',el.get('name'),el.get('value'))
        if tag=='NormDiscrete': chk('NormDiscrete@value',el.get('field'),el.get('value'))
        if tag=='MiningField':
            for a in('missingValueReplacement','invalidValueReplacement'): chk('MiningField@'+a,el.get('name'),el.get(a))
            for a in('lowValue','highValue'): chk('MiningField@'+a+' (outliers='+str(el.get('outliers'))+')',el.get('name'),el.get(a))
        if tag=='DataField':
            for v in el:
                if ln(v.tag)=='Value': chk('Value@value('+(v.get('property') or 'valid')+')',el.get('name'),v.get('value'))
        if tag=='PPCell': chk('PPCell@value',el.get('predictorName'),el.get('value'))
        if tag=='MapValues':
            cols={ln(c.tag) and c.get('column'):c.get('field') for c in el if ln(c.tag)=='FieldColumnPair'}
            for row in el.iter():
                if ln(row.tag)=='row':
                    for cell in row:
                        if ln(cell.tag) in cols: chk('MapValues key cell',cols[ln(cell.tag)],cell.text or '')
for k,v in sorted(hits.items()):
    print(v,k); [print('    ',e) for e in sorted(ex[k])[:4]]
