"""Checkpointed follow-up supervisor; no modification to A/B status or summaries."""
import argparse
import os
import subprocess
import sys
import time
from filelock import FileLock

from experiment import now, save
from mechanism import DATA, OUT, ROOT, validate_config
from mechanism_report import main as report
from pipeline import sync


def stopped():
    return (DATA/'STOP').exists() or (ROOT/'results/STOP').exists()


def status(state,stage,error=None):
    x={'state':state,'stage':stage,'updated':now(),'pid':os.getpid(),
       'ablation_tasks':len(list((DATA/'ablation').glob('*/eval.json'))),'ablation_expected':30,
       'factorial_tasks':len(list((DATA/'factorial').glob('*/eval_[0-9]*.json'))),'factorial_expected':240,
       'factorial_runs':len(list((DATA/'factorial').glob('*/summary.json'))),'factorial_runs_expected':12,
       'merge_steps':len(list((DATA/'factorial').glob('*/merge_[0-9]*.json'))),'merge_steps_expected':360}
    if error:x['error']=error
    save(OUT/'progress.json',x)
    (OUT/'STATUS.md').write_text(f"# 机制实验状态\n\n状态：{state}；阶段：{stage}；UTC 更新时间：{x['updated']}。\n\n规则消融：{x['ablation_tasks']}/30 次任务。融合实验：{x['factorial_tasks']}/240 次任务、{x['factorial_runs']}/12 个完整运行、{x['merge_steps']}/360 次融合。\n\n校园 DeepSeek，两路并发；旧 A/B 已完成且保持不变。\n"+(f'\n错误：{error}\n' if error else ''),encoding='utf-8')
    return x


def stage_run(stage,workers):
    for attempt in range(1,4):
        validate_config()
        if stopped():raise InterruptedError('STOP requested')
        with (DATA/(stage+'.log')).open('a',encoding='utf-8') as log:
            log.write(f'\n{now()} attempt={attempt}\n');log.flush()
            child=subprocess.Popen([sys.executable,'-X','utf8','mechanism.py',stage,'--workers',str(workers)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                status('running',stage)
                if stopped():
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True);child.wait(timeout=20)
                    raise InterruptedError('STOP requested; checkpoints retained')
                time.sleep(15)
        if child.returncode==0:return
        if attempt<3:
            for left in range(300,0,-15):
                if stopped():raise InterruptedError('STOP requested during cooldown')
                status('waiting_retry',stage,f'Checkpoint resume in {left}s');time.sleep(15)
    raise RuntimeError(f'{stage} incomplete after three checkpoint resumes')


def main(workers):
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(ROOT/'results/pipeline.lock'),timeout=0):
        if os.name=='nt':
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        phase='validation'
        try:
            validate_config();status('running',phase)
            subprocess.run([sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
            for phase in ['ablation','factorial']:
                stage_run(phase,workers);report();status('running',phase+' finished');sync('docs(mechanism): sync completed '+phase+' study')
            summary=report()
            if summary['status']!='complete':raise RuntimeError('Expected follow-up results missing')
            status('complete','ablation+factorial');sync('docs(mechanism): finalize follow-up analysis and teacher brief')
        except InterruptedError as exc:status('stopped',phase,str(exc));report()
        except Exception as exc:
            status('needs_attention',phase,str(exc));report()
            try:sync('chore(mechanism): record follow-up interruption')
            except Exception:pass
            raise
        finally:
            if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);a=parser.parse_args();main(a.workers)
