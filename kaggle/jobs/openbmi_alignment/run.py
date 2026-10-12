"""Cloud-only OpenBMI structure and clean-FAR feasibility audit."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path


def upper_zero(n, confidence=0.95):
    if n < 1:
        raise ValueError("positive independent evaluation count required")
    return 1 - (1 - confidence) ** (1 / n)


def event_audit(times, labels, samples):
    import numpy as np
    times, labels = np.asarray(times).ravel(), np.asarray(labels).ravel()
    return dict(n_events=len(times), aligned=len(times) == len(labels),
                increasing=bool(np.all(np.diff(times) > 0)),
                within_recording=bool(len(times) and times.min() >= 0 and times.max() <= samples),
                label_counts={str(v): int((labels == v).sum()) for v in np.unique(labels)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    out = Path('/kaggle/working/out'); out.mkdir(exist_ok=True)
    raw = Path('/kaggle/working/raw_openbmi'); raw.mkdir(exist_ok=True)
    # Independent sample design arithmetic only; not fitted EEG performance.
    design = [dict(n=n, zero_alarm_upper_95=upper_zero(n), rank_resolution=1/(n+1))
              for n in (5, 6, 20, 33, 36, 59, 69, 100, 200, 299, 500)]
    (out/'sample_size_design.json').write_text(json.dumps(dict(
        assumptions='IID Bernoulli evaluation; exchangeable rank calibration. Neither guaranteed for EEG.',
        confidence=0.95, nominal_far=0.05, rows=design,
        zero_alarm_min_n_95_for_5_percent=math.ceil(math.log(0.05)/math.log(0.95)),
        warning='Group pooling and flash repetitions do not imply per-user independent characters.'), indent=2))
    assert abs(upper_zero(1) - .95) < 1e-12
    assert upper_zero(59) <= .05 < upper_zero(58)
    assert event_audit([1,2], [1,2], 3)['aligned']
    assert not event_audit([2,1], [1,2], 3)['increasing']
    assert not event_audit([1], [1,2], 3)['aligned']
    (out/'focused_tests.json').write_text(json.dumps({'checks_passed':5,
        'scope':'five focused arithmetic/schema assertions; metadata-only follow-up, not full repository suite'}))
    import numpy as np
    import requests
    from scipy.io import loadmat, whosmat
    base='https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/'
    config=dict(seed=args.seed, subjects=[1], sessions=[1,2],
                selection='fixed first subject, both sessions; no outcome-based selection',
                source_commit='e2fddbc', runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=platform.python_version(),
                stage='event ordering and identity metadata only; no performance evaluation',
                reference='MOABB Lee2019_ERP documentation; nasion reference documented, not measured',
                no_raw_output=True, frozen_splits_untouched=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    schedule_url='https://raw.githubusercontent.com/PatternRecognition/OpenBMI/master/GigaScience/function_ERP/random_cell_order.mat'
    response=requests.get(schedule_url,timeout=60); response.raise_for_status()
    schedule_path=raw/'schedule.mat'; schedule_path.write_bytes(response.content)
    schedule_mat=loadmat(schedule_path,simplify_cells=True)
    keys=[k for k in schedule_mat if not k.startswith('__')]
    schedule_value=schedule_mat[keys[0]] if len(keys)==1 else None
    schedule_rows=[]
    if schedule_value is not None:
        cells=np.asarray(schedule_value,dtype=object)
        for item in cells.ravel():
            a=np.asarray(item)
            if a.ndim==2: schedule_rows.append(a.astype(int))
    schedule_available_shapes=[list(a.shape) for a in schedule_rows]
    schedule_rows=schedule_rows[:5]  # Original ERP_performance uses nSeq1..Nsequence=5.
    schedule_ok=bool(len(schedule_rows)==5 and all(a.shape==(12,6) and a.min()>=1 and a.max()<=36 for a in schedule_rows))
    if schedule_ok:
        schedule_ok=all(np.all(np.bincount(a.ravel(),minlength=37)[1:]==2) for a in schedule_rows)
    config['schedule']=dict(url=schedule_url,sha256=hashlib.sha256(response.content).hexdigest(),keys=keys,available_shapes=schedule_available_shapes,selection='first five entries as original decoder Nsequence=5',valid_structure=schedule_ok,shapes=[list(a.shape) for a in schedule_rows])
    (out/'config.json').write_text(json.dumps(config,indent=2))
    records=[]
    for session in (1,2):
        name=f'sess{session:02d}_subj01_EEG_ERP.mat'
        url=base+f'session{session}/s1/'+name
        path=raw/name
        print('Downloading cloud-only metadata source',name,flush=True)
        digest=hashlib.sha256()
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with path.open('wb') as stream:
                for block in response.iter_content(1024*1024):
                    if block: stream.write(block); digest.update(block)
        schema=whosmat(path)
        mat=loadmat(path,simplify_cells=True,
                    variable_names=['EEG_ERP_train','EEG_ERP_test'])
        for split in ('train','test'):
            data=mat[f'EEG_ERP_{split}']
            fields={k:dict(shape=list(np.shape(v)),dtype=str(np.asarray(v).dtype)) for k,v in data.items()}
            x=np.asarray(data['x']); times=np.asarray(data['t']).ravel()
            labels=np.asarray(data['y_dec']).ravel()
            events=event_audit(times,labels,x.shape[0])

            order=np.argsort(times,kind='stable')
            chronological=times[order]
            chronological_labels=labels[order]
            unique_times,counts=np.unique(times,return_counts=True)
            blocks=chronological_labels.reshape(-1,60) if len(times)%60==0 else None
            ordering=dict(duplicate_onsets=int(np.sum(counts-1)),
                nonincreasing_steps=int(np.sum(np.diff(times)<=0)),
                label_transition_count=int(np.sum(np.diff(labels)!=0)),
                original_label_runs=[dict(label=int(labels[i]),start=int(i)) for i in
                    np.r_[0,np.flatnonzero(np.diff(labels)!=0)+1]],
                sorted_gap_quantiles=np.quantile(np.diff(chronological),[0,.5,.9,.99,1]).tolist(),
                strict_zero_based_bounds=bool(times.min()>=0 and times.max()<x.shape[0]),
                strict_matlab_one_based_bounds=bool(times.min()>=1 and times.max()<=x.shape[0]),
                inferred60_block_target_counts=[] if blocks is None else (blocks==1).sum(axis=1).tolist(),
                sorted_order_status='diagnostic only, not validated character or symbol identity',
                label_vocab={k:np.unique(np.asarray(data[k]).astype(str)).tolist()[:40]
                    for k in ('y_dec','y_class')},
                class_mapping=str(data['class']),
                gap_interpretation='large gaps may locate decisions, cannot assign flashed symbols')
            symbols='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'
            phrase='NEURAL_NETWORKS_AND_DEEP_LEARNING' if split=='train' else 'PATTERN_RECOGNITION_MACHINE_LEARNING'
            membership=dict(schedule_valid_structure=schedule_ok,phrase_length=len(phrase),status='unverified')
            if schedule_ok and len(labels)==len(phrase)*60:
                groups=np.concatenate(schedule_rows,axis=0)
                expected_labels=np.concatenate([np.where(np.any(groups==symbols.index(c)+1,axis=1),1,2) for c in phrase])
                membership.update(original_order_mismatch_count=int(np.sum(expected_labels!=labels)),
                    chronological_order_mismatch_count=int(np.sum(expected_labels!=chronological_labels)),
                    original60_block_monotonic_counts=[int(np.sum(np.diff(b)<=0)) for b in times.reshape(-1,60)],
                    status='label consistency only; does not independently establish timestamp origin or physical schedule execution')

            # Fixed diagnostic samples, never selected by decoder outcome.
            reversals=np.flatnonzero(np.diff(times)<=0)
            reversal_details=[dict(position=int(i),position_in60=int(i%60),
                before=int(times[i]),after=int(times[i+1]),difference=int(times[i+1]-times[i])) for i in reversals]
            epoch=np.asarray(data['smt'])
            ival=np.asarray(data['ival']).ravel()
            selected=sorted(set([0,1,59,60,len(times)-1]+[int(i) for i in reversals[:3]]))
            matches=[]
            for shift in (-2,-1,0,1,2):
                squared=0.; reference=0.; count=0; correlations=[]
                for event in selected:
                    indices=times[event]+ival+shift
                    if np.all(indices>=0) and np.all(indices<x.shape[0]) and len(indices)==epoch.shape[0]:
                        a=x[indices.astype(int),:]; b=epoch[:,event,:]
                        squared+=float(np.sum((a-b)**2)); reference+=float(np.sum(b*b)); count+=a.size
                        correlations.append(float(np.corrcoef(a.ravel(),b.ravel())[0,1]))
                matches.append(dict(shift=shift,points=count,rmse=None if not count else float(np.sqrt(squared/count)),
                    relative_l2=None if not reference else float(np.sqrt(squared/reference)),
                    mean_correlation=None if not correlations else float(np.mean(correlations))))
            alignment=dict(reversals=reversal_details,selected_event_indices=selected,
                ival_min=float(ival.min()),ival_max=float(ival.max()),ival_length=len(ival),
                epoch_shape=list(epoch.shape),offset_comparisons=matches,
                formula='x[t + ival + shift]; MATLAB integer indexing predicts shift=-1 if ival are integer sample offsets',
                caveat='No shift automatically chosen. Filtered/baseline-adjusted epochs may differ; correlations do not prove exact alignment.')
            expected=33 if split=='train' else 36
            records.append(dict(subject=1,session=session,split=split,url=url,
                file_sha256=digest.hexdigest(),file_bytes=path.stat().st_size,
                top_level_schema=schema,fields=fields,fs=float(np.asarray(data['fs']).squeeze()),
                channel_names=[str(v) for v in np.asarray(data['chan']).ravel()],
                signal_shape=list(x.shape),events=events,ordering=ordering,schedule_membership=membership,alignment=alignment,
                documented_characters=expected,
                blocks_of_60_if_divisible=len(times)//60 if len(times)%60==0 else None,
                character_count_status='inferred block count only; independent decision IDs not yet validated',
                membership_status='unresolved; y_dec Target/NonTarget alone does not define flashed symbols',
                possible_membership_fields=[k for k in data if k not in ('x','t','fs','chan','y_dec','EMG','EMG_index')]))
            print('Metadata inspected',session,split,len(times),flush=True)
        del mat
        (out/'openbmi_alignment_audit.json').write_text(json.dumps(records,indent=2))
        path.unlink()  # Raw EEG never becomes an output artifact.
    (out/'README.md').write_text('Metadata-only audit completed. Character-block and stimulus membership reconstruction remains unresolved; no compatibility or performance guarantee. All calculations executed on Kaggle. Raw EEG removed from working storage after inspection. BNCI2015-003 was not accessed.\n')


if __name__ == '__main__':
    main()
