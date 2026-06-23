from __future__ import annotations
import argparse,csv,json,re,sys,time
from pathlib import Path

ARTS=['Clipping','Hiss','Buzz','Pops','Unnatural Prosody']
BASE={'sample_id','media_path','modality','track_id','label'}
TF_RE=re.compile(r'\b(true|false)\b',re.I); MC_RE=re.compile(r'\b([A-E])\b',re.I)
LINE_RE=re.compile(r'^\s*[\-*]?\s*`?\"?\s*([^:\n\r]+?)\s*\"?`?\s*:\s*\"?\s*(true|false|yes|no|0|1)\s*\"?\s*,?\s*$',re.I)
MAPPING_PROMPT = """
You are an AI evaluation engine. Your task is to process an analysis of a digital media sample (`Analysis Text`) and determine which artifacts from a predefined list (`Artifact Definitions`) are present.

Your evaluation must be based **strictly** on the definitions provided.

Your output must be a simple key-value checklist suitable for automated parsing. Use "True" or "False". Do not include any justifications, explanations, or any text other than the artifact name and its corresponding boolean value.

---

# **1. Analysis Text**

{RESPONSE}

---

# **2. Artifact Definitions**

You must check for the presence of the following artifacts. An artifact is "True" **only if** the `Analysis Text` provides evidence that matches its specific `Definition`.

* **Blurriness**
    * **Definition**: ["The loss of sharpness and fine detail, making the image appear out of focus."]
* **Blockiness**
    * **Definition**: ["Visible square or rectangular patterns on the screen."]
* **Noise**
    * **Definition**: ["Random, fine speckles or a sandy texture across the image."]
* **Banding**
    * **Definition**: ["Distinct, abrupt steps or bands in areas that should have a smooth color gradient, like a sunset or a clear sky."]
* **Color Inconsistency**
    * **Definition**: ["Colors appear unnatural, with excessive saturation or vibrancy that makes the sample look too intense or unrealistic."]
* **Blending Artifacts**
    * **Definition**: ["Visible boundaries where elements should merge smoothly."]
* **Lighting Inconsistency**
    * **Definition**: ["Illumination that does not agree across the scene."]
* **Unnatural Texture**
    * **Definition**: ["The surface is overly smooth, missing the natural irregularities and tactile cues of real materials."]
* **Temporal Artifacts**
    * **Definition**: ["Inconsistencies across frames that break motion continuity."]
* **Flicker**
    * **Definition**: ["Noticeable and often rapid variation in the overall brightness of the video."]
* **Clipping**
    * **Definition**: ["A harsh, fuzzy, or crackling sound that occurs when the audio is too loud for the system to handle."]
* **Hiss**
    * **Definition**: ["High-frequency static noise, often described as a shhhh sound."]
* **Buzz**
    * **Definition**: ["Low-frequency tone, typically caused by electrical interference."]
* **Pops**
    * **Definition**: ["Abrupt, short, and sharp sounds that interrupt the audio."]
* **Reflection Inconsistency**
    * **Definition**: ["Reflections do not match the subject, lighting, or scene geometry."]
* **Shadow Inconsistency**
    * **Definition**: ["Shadows do not match the subject, lighting, or scene geometry."]
* **Spatial & Contact Incoherence**
    * **Definition**: ["Objects or people fail to make contact with surfaces or each other."]
* **Unrealistic Background**
    * **Definition**: ["Background lacks plausible detail, perspective, or depth."]
* **Anatomical Inconsistency**
    * **Definition**: ["Human anatomy is implausible."]
* **Unnatural Expressions**
    * **Definition**: ["Facial expressions do not align with emotion or context or appears unrealistic."]
* **Unnatural Gaze or Blinking**
    * **Definition**: ["Eye direction or blink behavior appears robotic."]
* **Unnatural Body or Head Movement**
    * **Definition**: ["Motion lacks physical plausibility."]
* **Object Integrity Flaws**
    * **Definition**: ["The object is incomplete, broken, or internally inconsistent."]
* **Unrecognizable Text**
    * **Definition**: ["The text is unrecognizable, incomplete, broken, or distorted."]
* **Unnatural Prosody**
    * **Definition**: ["Speech often sounds robotic, monotonous, or flat, lacking natural intonation."]
* **Audio-Visual Desynchronization**
    * **Definition**: ["A mismatch between spoken audio and visible mouth movements or facial actions."]
* **Emotional Contradiction**
    * **Definition**: ["The face, voice, or body language conveys a different emotion than the content."]

---

# **Begin Evaluation**
"""
PROMPT = MAPPING_PROMPT

def b(x):
    if isinstance(x,bool): return x
    if isinstance(x,(int,float)): return True if x==1 else False if x==0 else None
    s=str(x or '').strip().strip('"\'').lower()
    if s in ('true','t','yes','y','1'): return True
    if s in ('false','f','no','n','0','none','null',''): return False
    return None

def norm(s): return re.sub(r'[^a-z0-9]+','',str(s or '').lower().replace('&','and'))
def recs(p):
    if p.suffix=='.jsonl':
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                try:
                    o=json.loads(line)
                    if isinstance(o,dict): yield o
                except Exception: pass
    elif p.suffix=='.json':
        try: o=json.loads(p.read_text(encoding='utf-8'))
        except Exception: return
        if isinstance(o,dict): yield o
        elif isinstance(o,list):
            for x in o:
                if isinstance(x,dict): yield x

def allrecs(root):
    for p in sorted(root.rglob('*.json'))+sorted(root.rglob('*.jsonl')): yield from recs(p)
def sid(r):
    if r.get('sample_id') or r.get('record_id'): return str(r.get('sample_id') or r.get('record_id')).strip()
    s=r.get('sample'); return str(s.get('sample_id') or '') if isinstance(s,dict) else ''
def qid(r):
    if r.get('question_id'): return str(r['question_id']).strip()
    s=r.get('sample'); return str(s.get('question_id') or s.get('sample_id') or '') if isinstance(s,dict) else ''
def tf(x):
    if isinstance(x,bool): return x
    if not isinstance(x,str): return None
    m=TF_RE.search(' '.join(x.splitlines()[:2])); return None if not m else m.group(1).lower()=='true'
def mc(x):
    if isinstance(x,list): return sorted({str(i).strip().upper() for i in x if str(i).strip().upper() in set('ABCDE')})
    if isinstance(x,str): return sorted(set(MC_RE.findall(' '.join(x.splitlines()[:3]).upper())))
    return None
def label(x):
    if not isinstance(x,str): return None
    for l in [z.strip().lower() for z in x.splitlines() if z.strip()]:
        if 'likely authentic' in l: return 'real'
        if 'likely manipulated' in l: return 'fake'
        break
    return None

def parse_map(text,arts=ARTS):
    ok={norm(a):a for a in arts}; out={}; text=str(text or '').strip()
    for cand in (text, text[text.find('{'):text.rfind('}')+1] if '{' in text and '}' in text else ''):
        if not cand: continue
        try: obj=json.loads(cand)
        except Exception: continue
        if isinstance(obj,dict):
            for k,v in obj.items():
                a=ok.get(norm(k)); bv=b(v)
                if a and bv is not None: out[a]=bv
            if out: return out
    for line in text.splitlines():
        m=LINE_RE.match(line.strip())
        if m:
            a=ok.get(norm(m.group(1))); bv=b(m.group(2))
            if a and bv is not None: out[a]=bv
    return out

def choice_answers(root,task,split):
    d=root/('MCQ' if task=='mcq' else 'TFQ')/split; ans={}
    if (d/'answers.jsonl').exists():
        for r in recs(d/'answers.jsonl'):
            if r.get('question_id'): ans[str(r['question_id'])]=r
    for p in d.glob('*.json'):
        try: rows=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        if isinstance(rows,list):
            for r in rows:
                q=str(r.get('question_id') or '')
                if q in ans:
                    ans[q].setdefault('question',r.get('question')); ans[q].setdefault('options',r.get('options'))
    return ans

def oeq_answers(root,split):
    p=root/'OEQ'/split/'answers_audio.csv'; ans={}
    if not p.exists(): return ans
    with p.open(newline='',encoding='utf-8') as f:
        rd=csv.DictReader(f); arts=[x for x in (rd.fieldnames or []) if x not in BASE] or ARTS
        for r in rd:
            if r.get('sample_id'):
                ans[r['sample_id']]={'label':str(r.get('label') or '').lower(),'arts':arts,'gt':{a:bool(b(r.get(a))) for a in arts}}
    return ans

def pdir(root,task,model,split):
    names={'typea_oeq':['typea_oeq','perception_oeq'],'typeb_oeq':['typeb_oeq','detection_oeq']}.get(task,[task])
    for n in names:
        p=root/n/model/split
        if p.is_dir(): return p
    raise FileNotFoundError(f'missing {task}/{model}/{split} under {root}')
class Mapper:
    def __init__(self,args): self.a=args; self.tok=None; self.mod=None
    def load(self):
        if self.mod: return
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        dt='auto' if self.a.torch_dtype=='auto' else getattr(torch,self.a.torch_dtype)
        self.tok=AutoTokenizer.from_pretrained(self.a.qwen_model,trust_remote_code=True,cache_dir=self.a.cache_dir)
        self.mod=AutoModelForCausalLM.from_pretrained(self.a.qwen_model,torch_dtype=dt,device_map=self.a.device_map,trust_remote_code=True,cache_dir=self.a.cache_dir).eval()
    def gen(self,text):
        self.load(); prompt=PROMPT.format(RESPONSE=text)
        msgs=[{'role':'system','content':'Output only the artifact checklist.'},{'role':'user','content':prompt}]
        try: q=self.tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        except Exception: q=prompt
        inp=self.tok(q,return_tensors='pt'); dev=getattr(self.mod,'device',None)
        if dev: inp={k:v.to(dev) for k,v in inp.items()}
        out=self.mod.generate(**inp,max_new_tokens=self.a.max_new_tokens,do_sample=False,pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inp['input_ids'].shape[-1]:],skip_special_tokens=True).strip()

def text_of(r,field):
    if field=='response' and r.get('response'): return str(r['response'])
    s=r.get('sample')
    if isinstance(s,dict):
        if s.get('analysis_text'): return str(s['analysis_text'])
        m=s.get('media_meta')
        if isinstance(m,dict) and m.get('analysis_text'): return str(m['analysis_text'])
    if r.get('analysis_text'): return str(r['analysis_text'])
    return str(r.get('response')) if field=='auto' and r.get('response') else None

def map_oeq(inp,out,args,task):
    out.mkdir(parents=True,exist_ok=True); mp=Mapper(args); rows=list(allrecs(inp))
    for i,r in enumerate(rows,1):
        k=sid(r) or qid(r) or f'sample-{i}'; op=out/f'{k}.json'
        if args.skip_existing_mapping and not args.overwrite_mapping and op.exists(): continue
        tx=text_of(r,args.analysis_field)
        if not tx: continue
        st=time.time(); err=None
        try: resp=mp.gen(tx)
        except Exception as e: err=str(e); resp=f'[ERROR] {e}'
        obj={'model_id':args.qwen_model,'sample':{'sample_id':k,'task':task,'modality':'audio','media_meta':{'analysis_text':tx}},'response':resp,'parsed_artifact_map':parse_map(resp),'latency_ms':(time.time()-st)*1000}
        if err: obj['error']=err
        op.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def score_choice(pred,ans,task):
    n=len(ans); m=0; bad=0; val=0.0; seen=set()
    for r in allrecs(pred):
        q=qid(r)
        if not q or q in seen or q not in ans: continue
        seen.add(q); m+=1; a=ans[q]
        if task=='tfq':
            p=tf(r.get('parsed_answer')); p=p if p is not None else tf(r.get('response')); g=tf(a.get('ground_truth'))
            if p is None or g is None: bad+=1
            elif p==g: val+=1
        else:
            p=mc(r.get('parsed_choices')); p=p if p is not None else mc(r.get('response')); g=mc(a.get('ground_truth'))
            if p is None or g is None: bad+=1
            else:
                total=5; kk=len(g); wrong=max(total-kk,1); val+=max(0,sum(1/kk for x in p if x in g)-sum(1/wrong for x in p if x not in g)) if kk else 0
    key='acc_tfq' if task=='tfq' else 'score_mcq'; return {'task_type':task,'expected_answers':n,'matched_answers':m,'invalid_predictions':bad,key:val/n if n else 0,'by_modality':[{'modality':'audio',key:val/n if n else 0}]}

def score_det(pred,ans):
    n=len(ans); m=0; ok=0; bad=0; seen=set()
    for r in allrecs(pred):
        k=sid(r)
        if not k or k in seen or k not in ans: continue
        seen.add(k); m+=1; p=r.get('parsed_label') if r.get('parsed_label') in ('real','fake') else label(r.get('response')); g=ans[k]['label']
        if p not in ('real','fake') or g not in ('real','fake'): bad+=1
        elif p==g: ok+=1
    return {'task_type':'typeb_oeq','expected_answers':n,'matched_answers':m,'correct':ok,'invalid_predictions':bad,'acc_det':ok/n if n else 0,'by_modality':[{'modality':'audio','acc_det':ok/n if n else 0}]}

def score_art(mp,ans,task):
    exp=sum(1 for x in ans.values() if any(x['gt'].values())); seen=set(); cover=chair=hal=f05=0.0; m=0
    for r in allrecs(mp):
        k=sid(r)
        if not k or k in seen or k not in ans: continue
        seen.add(k); a=ans[k]; arts=a['arts']; gt=a['gt']; pm=r.get('parsed_artifact_map') if isinstance(r.get('parsed_artifact_map'),dict) else parse_map(str(r.get('response') or ''),arts)
        gc=sum(1 for x in arts if gt.get(x)); pc=sum(1 for x in arts if pm.get(x)); mt=sum(1 for x in arts if gt.get(x) and pm.get(x))
        if gc<=0: continue
        m+=1; c=mt/gc; ch=1.0 if pc==0 else 1-mt/pc; pr=1-ch; den=.25*pr+c; f=1.25*pr*c/den if den>0 else 0
        cover+=c; chair+=ch; hal+=1 if ch>0 else 0; f05+=f
    miss=max(exp-m,0); res={'task_type':task,'expected_fake_samples':exp,'fake_samples_scored':m,'cover':cover/exp if exp else 0,'chair':(chair+miss)/exp if exp else 0,'hal_rate':(hal+miss)/exp if exp else 0,'f_0_5':f05/exp if exp else 0}
    res['by_modality']=[{'modality':'audio','cover':res['cover'],'chair':res['chair'],'hal_rate':res['hal_rate'],'f_0_5':res['f_0_5']}]; return res

def compact(x):
    t=x['task_type']; fs={'tfq':['acc_tfq'],'mcq':['score_mcq'],'typea_oeq':['cover','chair','hal_rate','f_0_5'],'typeb_oeq':['acc_det']}[t]
    d={'task_type':t,'scores_by_modality':[{'modality':'audio',**{f:x['by_modality'][0].get(f) for f in fs}}]}
    if t=='typeb_oeq' and 'artifact_by_modality' in x: d['artifact_scores_by_modality']=x['artifact_by_modality']
    return d

def calc_tcs(tasks):
    d={x['task_type']:x for x in tasks}; get=lambda t,k: d.get(t,{}).get('scores_by_modality',[{}])[0].get(k)
    acc,tfqv,mcqv=get('typeb_oeq','acc_det'),get('tfq','acc_tfq'),get('mcq','score_mcq')
    ta=get('typea_oeq','f_0_5'); tb=d.get('typeb_oeq',{}).get('artifact_scores_by_modality',[{}])[0].get('f_0_5')
    f=(ta+tb)/2 if ta is not None and tb is not None else None
    complete=all(v is not None for v in (acc,tfqv,mcqv,f)); val=None if not complete else 40*acc+30*f+30*(.5*tfqv+.5*mcqv)
    return [{'modality':'audio','acc_det':acc,'acc_tfq':tfqv,'score_mcq':mcqv,'typea_f_0_5':ta,'typeb_f_0_5':tb,'f_0_5':f,'tcs':val,'complete':complete}]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',default='all',choices=['all','tfq','mcq','typea_oeq','typeb_oeq']); ap.add_argument('--split',default='public_val'); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--predictions-root',type=Path,required=True); ap.add_argument('--model',required=True); ap.add_argument('--qwen-model',default='Qwen/Qwen3.5-4B'); ap.add_argument('--analysis-field',default='response',choices=['response','analysis_text','auto']); ap.add_argument('--device-map',default='auto'); ap.add_argument('--torch-dtype',default='auto'); ap.add_argument('--max-new-tokens',type=int,default=512); ap.add_argument('--cache-dir'); ap.add_argument('--mapping-root',type=Path); ap.add_argument('--summary-out',type=Path); ap.add_argument('--skip-existing-mapping',action='store_true'); ap.add_argument('--overwrite-mapping',action='store_true'); args=ap.parse_args()
    ts=['tfq','mcq','typea_oeq','typeb_oeq'] if args.task=='all' else [args.task]; raw={}; comps=[]
    for t in ts:
        pd=pdir(args.predictions_root,t,args.model,args.split)
        if t in ('tfq','mcq'): res=score_choice(pd,choice_answers(args.data_root,t,args.split),t)
        else:
            ans=oeq_answers(args.data_root,args.split); md=(args.mapping_root or args.predictions_root/'evaluation_results'/'oeq_mappings')/t/args.model/args.split/'qwen35-4b-response'; map_oeq(pd,md,args,t); art=score_art(md,ans,t)
            res=score_det(pd,ans) if t=='typeb_oeq' else art
            if t=='typeb_oeq': res['artifact_by_modality']=art['by_modality']
            res['oeq_mapping_dir']=str(md)
        raw[t]=res; comps.append(compact(res))
    out={'split':args.split,'model_dir':args.model,'qwen_model':args.qwen_model,'tasks':sorted(comps,key=lambda x:x['task_type']),'tcs_formula':'tcs = 0.4*SDet + 0.3*SHal + 0.3*SPerc','raw_tasks':raw}; out['tcs_by_modality']=calc_tcs(out['tasks'])
    op=args.summary_out or args.predictions_root/'evaluation_results'/f'audio__{args.split}__{args.model}.json'; op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({'summary_out':str(op),'tcs_by_modality':out['tcs_by_modality']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
