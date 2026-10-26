"""CPU-only revised full-cohort analysis, fail closed on inspected review receipt."""
import os,sys,json,hashlib,subprocess,shutil
from pathlib import Path
EXPECTED_REVIEW_RECEIPT_SHA256='REPLACE_ONLY_AFTER_REVIEW_INSPECTION'
LAUNCH_REVISION='REPLACE_ONLY_AFTER_REVIEW_PRESERVATION'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipts=list(Path('/kaggle/input').rglob('analysis_release_receipt.json'));assert len(receipts)==1
rp=receipts[0];assert digest(rp)==EXPECTED_REVIEW_RECEIPT_SHA256
proof=json.loads(rp.read_text());assert proof['status']=='review_passed_no_combined_performance_computed'
assert proof['rows']==49929 and proof['strict_violations_retained']==13419
assert proof['guard_tests']==6 and proof['original_portable_suite_returncode']==0
assert proof['contract']['id']=='float32_conversion_envelope_20261026_v1'
base=rp.parent.parent;source=base/'code';archive=base/'archive'
for rel,h in proof['source_sha256'].items():assert digest(source/rel)==h,'review source mismatch '+rel
for rel,h in proof['original_json_sha256'].items():assert digest(archive/rel)==h,'original input mismatch '+rel
for rel,h in proof['clean_csv_sha256'].items():assert digest(archive/rel)==h,'clean CSV mismatch '+rel
code=Path('/kaggle/working/code');shutil.copytree(source,code)
os.environ['PYTHONPATH']=str(code/'src')+':'+str(code);sys.path[:0]=[str(code/'src'),str(code)]
from p3a.data.openbmi_release import execution_code_hashes
protocol=code/'splits/openbmi_independent_20261022/protocol.json'
oldreceipt=json.loads((protocol.parent/'receipt_s37.json').read_text());assert execution_code_hashes(code)==oldreceipt['execution_code_sha256']
assert digest(protocol)==proof['protocol_sha256']
out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
(out/'analysis_provenance.json').write_text(json.dumps(dict(launch_revision=LAUNCH_REVISION,review_receipt_sha256=digest(rp),runner_sha256=digest(__file__),contract=proof['contract'],original_protocol_sha256=digest(protocol),original_strict_contract_satisfied=False,original_strict_violations=13419,scope='all51originaltasks17conditionalusers;post-executionnumerical-envelopeanalysis;notuntouchedstrictprimary'),indent=2))
def run(args):subprocess.run([sys.executable]+args,cwd=code,check=True)
run(['reporting/aggregate_openbmi_numerical_envelope.py','--root',str(archive/'independent_results'),'--protocol',str(protocol),'--out-dir',str(out/'aggregate'),'--git-commit',LAUNCH_REVISION,'--seed','20261013'])
a=json.loads((out/'aggregate/reliability_aggregate.json').read_text());assert a['numerical_analysis_contract']==proof['contract'] and a['original_strict_contract_satisfied'] is False
run(['reporting/build_reliability_report.py','--aggregate',str(out/'aggregate/reliability_aggregate.json'),'--protocol',str(protocol),'--out-dir',str(out/'report'),'--git-commit',LAUNCH_REVISION])
run(['reporting/build_cost_report.py','--clean-root',str(archive/'independent_results'),'--timing-root',str(code/'timing'),'--protocol',str(protocol),'--out-dir',str(out/'cost_report'),'--git-commit',LAUNCH_REVISION])
(out/'analysis_complete.json').write_text(json.dumps(dict(status='revised_numerical_envelope_analysis_complete',tasks=51,eligible_users=17,evaluation_flow=18,unavailable_subject=52,review_receipt_sha256=digest(rp),aggregate_sha256=digest(out/'aggregate/reliability_aggregate.json'),protocol_sha256=digest(protocol),launch_revision=LAUNCH_REVISION,strict_original_compliance=False,figures_require_actual_visual_review=True,interpretation_requires_review=True),indent=2))
print('Complete separately labeled revised analysis; figures and scientific interpretation still require review.',flush=True)
