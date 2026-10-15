"""Source-only gradient-trained template adaptation, not an exact NSR replica."""
import numpy as np
import torch
from p3a.attacks.generators import place_template_at_onsets,linf_rescale


def fit_source_template(trials,logits_fn,eps_raw,seed,device='cpu',updates=200,batch_size=8,lr=.01,template_len=800):
    if not trials or updates<1 or batch_size<1 or template_len<1 or lr<=0 or not np.isfinite(eps_raw) or eps_raw<=0:
        raise ValueError('positive finite training settings and source trials required')
    allowed={f's{s:02d}_session1' for s in range(1,19)}
    if {t.subject_id for t in trials}!=allowed or any(t.source_split!='train' for t in trials):
        raise ValueError('exact source1..18 official train only')
    channels=trials[0].eeg.shape[0]
    if any(t.eeg.shape[0]!=channels for t in trials):raise ValueError('aligned source channels required')
    rng=np.random.default_rng(seed);generator=torch.Generator(device=device).manual_seed(seed)
    template=(.001*torch.randn(channels,template_len,device=device,generator=generator)).requires_grad_(True)
    optimizer=torch.optim.Adam([template],lr=lr);losses=[]
    for update in range(updates):
        # Accumulate individual variable-length trials; never pad raw EEG to batch.
        indices=rng.choice(len(trials),size=batch_size,replace=len(trials)<batch_size)
        optimizer.zero_grad();total=0.
        for index in indices:
            trial=trials[index];raw=torch.as_tensor(trial.eeg,dtype=torch.float32,device=device).unsqueeze(0)
            on=torch.as_tensor(trial.onsets,dtype=torch.long,device=device).unsqueeze(0)
            groups=torch.as_tensor(trial.flash_groups,dtype=torch.bool,device=device).unsqueeze(0)
            intended=trial.layout.index(trial.target_symbol);k=trial.flash_groups.shape[1]
            draw=int(rng.integers(k-1));target=torch.tensor([draw+(draw>=intended)],device=device)
            placed=place_template_at_onsets(template.unsqueeze(0),on,groups,target,raw.shape[-1])
            delta=linf_rescale(placed,eps_raw).clamp(-eps_raw,eps_raw)
            loss=torch.nn.functional.cross_entropy(logits_fn(raw+delta,on,groups),target)
            if not torch.isfinite(loss):raise ValueError('nonfinite source template objective')
            (loss/batch_size).backward();total+=float(loss.detach())/batch_size
        if template.grad is None or not torch.isfinite(template.grad).all():raise ValueError('nonfinite template gradient')
        optimizer.step()
        with torch.no_grad():template.clamp_(-1,1)
        losses.append(total)
    return template.detach(),dict(seed=seed,updates=updates,batch_size=batch_size,lr=lr,template_samples=template_len,
        eps_raw=eps_raw,losses=losses,source_trial_ids=[t.trial_id for t in trials],
        scope='gradient-trained source-only template adaptation; no target fitting or physical timing guarantee')
