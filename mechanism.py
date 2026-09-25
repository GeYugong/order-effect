"""Frozen follow-up: length-repair factorial and one-rule ablation, campus API only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import random

from benchmark import ROOT, dataset
from experiment import digest, event_sink, merge, now, parse_object, run_task, save
from study import build_manifest, client_for
from pipeline import seal

OUT = ROOT/'reports/mechanism_v1'
DATA = ROOT/'results/mechanism_v1'
REPAIR_PROMPT = """Shorten only the supplied overlong rules to 60-85 English whitespace-separated words each (hard limit 100). Preserve each rule's meaning, scope, conditions, exceptions and tool constraints. Add no advice or evidence. Do not change IDs. Return only JSON: {"items":[{"index":0,"content":"..."}]}, one item for every supplied index. You have no task outcome or evaluation data."""


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def config():
    return read(OUT/'manifest.json')


def prepare():
    seal()
    m = build_manifest()
    pool = read(ROOT/'results/B/frozen_pool.json')
    memory = read(ROOT/'results/B/merge_shuffle2_r2/final_memory.json')['memory']
    assert 'call traveler verification with full details' in memory[8]['content']
    assert len(memory[8]['content'].split()) == 79
    assert len(memory[10]['content'].split()) == 77
    x = {'created': now(), 'design': 'Post-hoc mechanism screening; new generations; no modification of A/B',
         'model_config': m, 'pool_hash': pool['pool_hash'], 'policies': ['skip_baseline', 'repair_length_once'],
         'orders': ['shuffle1', 'shuffle2'], 'writer_repetitions': 3, 'evaluation_repetitions': 1,
         'expected_factorial_tasks': 240, 'expected_merge_steps': 360,
         'factorial_schedule_seed': 202609261, 'ablation_schedule_seed': 202609262,
         'ablation': {'task_id': 'multi_turn_base_192', 'source': 'results/B/merge_shuffle2_r2/final_memory.json',
                      'memory_hash': digest(memory), 'remove_index_zero_based': 8, 'removed_rule': memory[8],
                      'control_remove_index': 10, 'control_rule': memory[10],
                      'booking_request_turn_zero_based': 3,
                      'repetitions_each': 10, 'conditions': ['original', 'remove_rule', 'remove_control'], 'expected_tasks': 30},
         'primary': '(repair shuffle1 - repair shuffle2) - (baseline shuffle1 - baseline shuffle2), complete writer-run scores',
         'limits': '3 independent writers/cell, 20 reused test tasks and one fixed pool; not external confirmation. Extra repair call changes computation as well as update acceptance.',
         'repair_policy': 'Exactly one content-shortening call on a structurally valid, source-valid candidate failing the 100-word rule only; keep all source labels and all non-overlong rules unchanged. If repair fails, retain prior memory. Never retry on task outcomes.'}
    files = ['mechanism.py', 'gateway.py', 'experiment.py', 'benchmark.py', 'study.py', 'embedding.py', 'reports/mechanism_v1/PROTOCOL.md']
    x['code_hashes'] = {f: hashlib.sha256((ROOT/f).read_text(encoding='utf-8').encode()).hexdigest() for f in files}
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Follow-up already frozen; refusing to overwrite')
    save(OUT/'manifest.json', x)
    return x


def validate_config():
    seal()
    x = config()
    for f, h in x['code_hashes'].items():
        if hashlib.sha256((ROOT/f).read_text(encoding='utf-8').encode()).hexdigest() != h:
            raise RuntimeError('Follow-up frozen code changed: '+f)
    if digest(build_manifest()) != digest(x['model_config']):
        raise RuntimeError('Base model configuration changed')
    pool = read(ROOT/'results/B/frozen_pool.json')
    if digest(pool['experiences']) != x['pool_hash']:
        raise RuntimeError('Fixed pool changed')
    return x, pool


def repair_candidate(before, incoming, generation, client):
    """Return repaired rules or before; never allow a second semantic merge attempt."""
    record = {'attempted': False, 'accepted': False}
    if generation.get('error') != 'Playbook rule exceeds word budget':
        return before, record
    known = {s for r in before for s in r['sources']} | {r['source'] for r in incoming}
    try:
        candidate = parse_object(generation['message']['content'])['rules']
        if not isinstance(candidate, list) or len(candidate)>12:
            raise ValueError('Invalid candidate rule count')
        for r in candidate:
            if not isinstance(r.get('content'), str) or not r['content'].strip() or not isinstance(r.get('sources'), list) or not r['sources'] or not set(r['sources']) <= known:
                raise ValueError('Invalid candidate structure/provenance')
        overlong = {i for i,r in enumerate(candidate) if len(r['content'].split())>100}
        if not overlong:
            raise ValueError('No overlong candidate rules')
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        record['error'] = str(exc)
        return before, record
    record['attempted'] = True
    message, meta = client.chat([{'role':'system','content':REPAIR_PROMPT}, {'role':'user','content':json.dumps({'items':[{'index':i,'content':candidate[i]['content']} for i in sorted(overlong)]},ensure_ascii=False)}])
    record.update(message=message, meta=meta, indices=sorted(overlong))
    try:
        if meta['finish_reason']=='length':
            raise ValueError('Repair truncated')
        items = parse_object(message.get('content'))['items']
        if not isinstance(items,list) or len(items)!=len(overlong) or {r['index'] for r in items}!=overlong:
            raise ValueError('Repair indices changed')
        result = copy.deepcopy(candidate)
        for r in items:
            if type(r['index']) is not int or not isinstance(r.get('content'),str) or not 0<len(r['content'].split())<=100:
                raise ValueError('Repair still violates word budget')
            result[r['index']]['content'] = r['content']
        record['accepted'] = True
        return result, record
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        record['error'] = str(exc)
        return before, record


def start(folder, x, job):
    fingerprint = digest({'config':x,'job':job})
    if (folder/'started.json').exists():
        if read(folder/'started.json')['signature'] != fingerprint:
            raise RuntimeError('Follow-up checkpoint configuration mismatch')
    else:
        save(folder/'started.json', {'signature':fingerprint,'job':job,'started':now()})
    client = client_for(x['model_config'])
    client.set_ledger(folder/'usage.jsonl')
    return client


def factorial_job(job, x, pool):
    folder = DATA/'factorial'/f"{job['policy']}_{job['order']}_r{job['rep']}"
    client = start(folder,x,job)
    if (folder/'summary.json').exists(): return read(folder/'summary.json')
    try:
        rules = []
        for position, task_id in enumerate(x['model_config']['orders'][job['order']]):
            path = folder/f'merge_{position:03d}.json'
            if path.exists(): rules = read(path)['after']; continue
            proposal_path = folder/f'proposal_{position:03d}.json'
            if proposal_path.exists(): proposal = read(proposal_path)
            else:
                after, generation = merge(rules,pool['experiences'][task_id],client)
                proposal = {'before_hash':digest(rules),'after':after,'generation':generation}
                save(proposal_path,proposal)
            if proposal['before_hash']!=digest(rules): raise RuntimeError('Proposal memory mismatch')
            after = proposal['after']; repair = {'attempted':False,'accepted':False}
            if job['policy']=='repair_length_once' and not proposal['generation'].get('update_valid',True):
                after,repair = repair_candidate(rules,pool['experiences'][task_id],proposal['generation'],client)
            save(path, {'task_id':task_id,'before':rules,'after':after,'generation':proposal['generation'],'repair':repair})
            rules = after
        save(folder/'final_memory.json',{'memory':rules,'hash':digest(rules)})
        tasks,gold = dataset(); rows=[]
        context = '\n'.join(r['content'] for r in rules)
        for i,key in enumerate(x['model_config']['splits']['evaluation']):
            path = folder/f'eval_{i:03d}.json'
            if path.exists(): result=read(path)
            else:
                result=run_task(tasks[key],gold[key],client,context,x['model_config']['max_steps_per_turn'],event_sink(folder/'api_trace.jsonl',key),folder/f'checkpoint_{i:03d}.json')
                save(path,result)
            rows.append({'task_id':key,'success':result['success']})
        steps=[read(p) for p in folder.glob('merge_[0-9]*.json')]
        summary={**job,'rows':rows,'score':sum(r['success'] for r in rows)/len(rows),'usage':client.usage,'memory_hash':digest(rules),'proposal_invalid':sum(not s['generation'].get('update_valid',True) for s in steps),'repair_attempts':sum(s['repair']['attempted'] for s in steps),'repair_accepted':sum(s['repair']['accepted'] for s in steps),'updates_skipped':sum(not s['generation'].get('update_valid',True) and not s['repair']['accepted'] for s in steps),'finished':now()}
        save(folder/'summary.json',summary);return summary
    except Exception as exc:
        save(folder/'failure.json',{'error':str(exc),'time':now()});raise


def ablation_job(job,x,pool=None):
    folder=DATA/'ablation'/f"{job['condition']}_r{job['rep']}"
    client=start(folder,x,job)
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    memory=read(ROOT/x['ablation']['source'])['memory']
    if digest(memory)!=x['ablation']['memory_hash']:raise RuntimeError('Ablation source memory changed')
    remove_index = x['ablation']['remove_index_zero_based'] if job['condition']=='remove_rule' else x['ablation']['control_remove_index'] if job['condition']=='remove_control' else None
    if remove_index is not None:memory=[r for i,r in enumerate(memory) if i!=remove_index]
    tasks,gold=dataset();key=x['ablation']['task_id'];path=folder/'eval.json'
    try:
        if path.exists():result=read(path)
        else:
            result=run_task(tasks[key],gold[key],client,'\n'.join(r['content'] for r in memory),x['model_config']['max_steps_per_turn'],event_sink(folder/'api_trace.jsonl',key),folder/'checkpoint.json')
            save(path,result)
        calls=[call for turn in result['decoded'] for step in turn for call in step]
        request_turn=result['decoded'][x['ablation']['booking_request_turn_zero_based']]
        requested_calls=[call for step in request_turn for call in step]
        summary={**job,'score':int(result['success']),'booking_called':any(c.startswith('book_flight(') for c in calls),'booking_on_requested_turn':any(c.startswith('book_flight(') for c in requested_calls),'usage':client.usage,'memory_hash':digest(memory),'finished':now()}
        save(folder/'summary.json',summary);return summary
    except Exception as exc:
        save(folder/'failure.json',{'error':str(exc),'time':now()});raise


def run(phase,workers):
    if not 1 <= workers <= 2:
        raise ValueError('Follow-up concurrency must be 1 or 2')
    x,pool=validate_config()
    if phase=='ablation':
        jobs=[{'condition':c,'rep':r} for c in x['ablation']['conditions'] for r in range(x['ablation']['repetitions_each'])]
        fn=ablation_job
    else:
        jobs=[{'policy':p,'order':o,'rep':r} for p in x['policies'] for o in x['orders'] for r in range(x['writer_repetitions'])]
        fn=factorial_job
    random.Random(x[phase+'_schedule_seed']).shuffle(jobs)
    failures=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(fn,j,x,pool):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:result=future.result();print(json.dumps({'complete':job,'score':result['score']}),flush=True)
            except Exception as exc:failures.append({'job':job,'error':str(exc)});print(json.dumps({'failed':job,'error':str(exc)}),flush=True)
    save(DATA/phase/'execution.json',{'failures':failures,'finished':now()})
    if failures:raise RuntimeError(f'{len(failures)} incomplete follow-up jobs')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','ablation','factorial']);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:run(args.command,args.workers)
