import os, subprocess
LOGS = {"helpdesk": "02:00:00", "bpi13_incidents": "02:00:00", "mimic_transfer": "02:00:00", "BPI20ID": "03:00:00", "BPI17": "08:00:00"}
for log, tl in LOGS.items():
    for mode, flag in (("names", ""), ("masked", " --mask-names")):
        for sd in (0, 1, 2):
            out = "outputs/fmv2/%s_%s_s%d.csv" % (log, mode, sd)
            cmd = ("cd /workspace && python scripts/fmv2_eval.py --repo external/events-transf --checkpoint-dir external/fmv2_ckpt/checkpoints "
                   "--splits exports/%s_splits.csv --log %s --seeds %d --k 5 10 20 --device cuda --out %s%s" % (log, log, sd, out, flag))
            name = "r9-fmv2-%s-%s-s%d" % (log, mode, sd)
            r = subprocess.run(["sbatch", "--job-name=" + name, "--time=" + tl, "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run_cmd.sbatch"],
                               env=dict(os.environ, USE_GPU="1", CMD=cmd), capture_output=True, text=True)
            print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
