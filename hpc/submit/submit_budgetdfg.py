import os, subprocess
base = ["task=evaluate", "evaluate=label_efficiency", "model=role_gin15", "~evaluate.backbones.ar", "~evaluate.backbones.random",
        "+evaluate.backbones.gin15allw1=backbone-20260825-125337-multi-none-frozen-gin15allw1-b1041f",
        "evaluate.eval_dataset=BPI20ID", "evaluate.eval_log.path=/workspace/data/raw/BPI20ID.xes",
        "evaluate.tasks=[next_activity,next_3_activities,next_5_activities,next_time,remaining_time,remaining_count,future_activity_set]",
        "evaluate.probe.max_epochs=100", "evaluate.probe.early_stop_patience=10",
        "evaluate.label_sizes=[0,10,30,100,300,1000,5000,null]", "evaluate.role_corpus=budget"]
for sd in (0, 1, 2):
    args = " ".join(base + ["evaluate.seeds=[%d]" % sd])
    name = "r7-budgetdfg-BPI20ID-s%d" % sd
    cmd = ["sbatch", "--job-name=" + name, "--time=08:00:00", "--cpus-per-task=4", "--gres=gpu:1", "--export=ALL", "slurm/ncps/run.sbatch"]
    r = subprocess.run(cmd, env=dict(os.environ, USE_GPU="1", ARGS=args), capture_output=True, text=True)
    print(("OK   " if r.returncode == 0 else "FAIL ") + name + "  " + (r.stdout or r.stderr).strip())
    if sd == 0: print("     ARGS: " + args)
