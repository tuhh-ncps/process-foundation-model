"""Does a trained SuTraN+ checkpoint use its prefix encoder?  Loads the best checkpoint (their composite rule over
backup_results.csv, or --epoch), decodes the test set as-is and with the prefix destroyed while the decoder input is kept,
and reports first-token accuracy and how many first tokens change.  Usage:
  python scripts/sutran_ablate.py --repo external/SuTraN_Plus --data HELPDESK --run SUTRAN_DA_results_seed_1/CaLenDiR_training/Default_Equal_Weighting
"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--data", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--epoch", type=int, default=None); ap.add_argument("--max-rows", type=int, default=6000); ap.add_argument("--window", type=int, default=64)
a = ap.parse_args(); sys.path.insert(0, a.repo); from SuTraN.SuTraN import SuTraN  # noqa: E402
D = os.path.join(a.repo, a.data); R = os.path.join(D, a.run); W = a.window
if a.epoch is None:
    d = pd.read_csv(os.path.join(R, "backup_results.csv")); score = np.zeros(len(d))
    for c, m in (("Activity suffix: 1-DL (validation)", "max"), ("TTNE - minutes MAE validation", "min"), ("RRT - mintues MAE validation", "min")):
        x = d[c].values.astype(float); rng = x.max() - x.min(); n = (x - x.min()) / rng if rng > 0 else np.zeros_like(x); score += (1 - n) if m == "max" else n
    a.epoch = int(d.iloc[int(score.argmin())]["epoch"])
card = pickle.load(open(os.path.join(D, "%s_cardin_dict.pkl" % a.data), "rb"))["concept:name"]
means = pickle.load(open(os.path.join(D, "%s_train_means_dict.pkl" % a.data), "rb")); stds = pickle.load(open(os.path.join(D, "%s_train_std_dict.pkl" % a.data), "rb"))
ms = lambda k, i: [means[k][i], stds[k][i]]
m = SuTraN(num_activities=card + 2, d_model=32, cardinality_categoricals_pref=[card], num_numericals_pref=2, num_prefix_encoder_layers=4, num_decoder_layers=4, num_heads=8, d_ff=128, dropout=0.2, remaining_runtime_head=True, layernorm_embeds=True, outcome_bool=False, out_type=None, num_outclasses=None)
sd = torch.load(os.path.join(R, "model_epoch_%d.pt" % a.epoch), map_location="cpu", weights_only=False); m.load_state_dict(sd["model_state_dict"]); m.eval()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); m.to(dev)
import SuTraN.SuTraN as SM; SM.device = dev  # the model moves its decoding buffers to the module-level device
test = torch.load(os.path.join(D, "test_tensordataset.pt"), map_location="cpu", weights_only=False)
n = min(a.max_rows, test[0].shape[0]); test = tuple(x[:n] for x in test); lab = test[7]; END = int(lab.max()); has = lab[:, 0] != END
def run(inp):
    with torch.no_grad():
        return torch.cat([m(tuple(x[i:i + 512].to(dev) for x in inp), W, ms("timeLabel_df", 0), ms("suffix_df", 1), ms("suffix_df", 0))[0].cpu() for i in range(0, n, 512)])
acc = lambda d: float((d[:, 0][has] == lab[:, 0][has]).float().mean())
base = tuple(test[:5]); d0 = run(base)
print("%s epoch %d, %d test rows: first-token accuracy as-is %.3f" % (a.data, a.epoch, n, acc(d0)))
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
for name, inp in (("prefix shuffled across rows", (base[0][perm], base[1][perm], base[2][perm], base[3], base[4])),
                  ("prefix activities all id 1", (torch.full_like(base[0], 1), base[1], base[2], base[3], base[4])),
                  ("prefix masked to its first event", (base[0], base[1], torch.ones_like(base[2]).index_fill_(1, torch.tensor([0]), False), base[3], base[4])),
                  ("decoder input token set to id 1", (base[0], base[1], base[2], torch.full_like(base[3], 1), base[4]))):
    d1 = run(inp); print("  %-36s acc %.3f | first token changed on %.3f of rows" % (name, acc(d1), float((d1[:, 0] != d0[:, 0]).float().mean())))
