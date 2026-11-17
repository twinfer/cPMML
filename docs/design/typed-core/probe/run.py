import re,subprocess,sys,os
P=os.path.dirname(os.path.abspath(__file__))
src=open(P+'/mixed.pmml').read()
outs=re.findall(r'<OutputField .*?</OutputField>',src)
head,tail=src.split('<Output>')[0],src.split('</Output>')[1]
for o in outs:
    name=re.search(r'name="([^"]+)"',o)[1]
    pm=f'{P}/one_{name}.pmml'; open(pm,'w').write(head+'<Output>'+o+'</Output>'+tail)
    r=subprocess.run(['java','-cp',__import__('subprocess').check_output(['git','rev-parse','--show-toplevel'],text=True).strip()+'/tools/pmml-evaluator-example-executable-1.7.7.jar','org.jpmml.evaluator.example.EvaluationExample','--model',pm,'--input',P+'/mixed.csv','--output',f'{P}/one_{name}.out'],capture_output=True,text=True)
    if r.returncode==0:
        rows=[l.strip().split(',')[-1] for l in open(f'{P}/one_{name}.out').read().splitlines()[1:]]
        print(f'{name:12s} {rows}')
    else:
        m=[l for l in r.stderr.splitlines() if 'Exception' in l][:1]
        print(f'{name:12s} ERROR {m}')
