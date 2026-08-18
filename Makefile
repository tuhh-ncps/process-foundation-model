# Makefile — build the container once, submit to either cluster (NCPS / TUHH).
#
# Self-contained: the image, data and outputs all live INSIDE this project directory,
# so nothing depends on $SCRATCH or any path outside the repo. Partition / account /
# topology live in each cluster's sbatch script (slurm/<cluster>/pretrain.sbatch).
#
#     make build          # build the Apptainer image into the project (login node, needs internet)
#     make submit_ncps    # submit the NCPS job   (single-GPU, trainer=local)
#     make submit_tuhh    # submit the TUHH job   (2-node DDP)
#     make queue          # squeue for your jobs
# Override inline, e.g.:  make submit_ncps EXPERIMENT=time2vec DATASET=bpi12_bpi17
# ---------------------------------------------------------------------------

# Load secrets from a gitignored .env if present (e.g. WANDB_API_KEY=xxxxxxxx, no quotes).
# A value passed on the CLI (make sync WANDB_API_KEY=…) still overrides the file.
-include .env

APPTAINER   ?= apptainer                     # or: make build APPTAINER=singularity
DEF         := containers/pmfoundation.def
IMAGE       ?= $(CURDIR)/pmfoundation.sif     # container lives in the project (matches sbatch default $REPO/…)
EXPERIMENT  ?= baseline                       # architecture (configs/experiment/*): baseline|mlp|mlp_gate|fourier|time2vec|rope|tbias|rope_time2vec_tbias_gate|role
DATASET     ?= bpi12                          # training data (configs/data/*): bpi12 | bpi17 | bpi12_bpi17 | mimic_transfers
                                              #   comma list => ONE backbone on all of them, 70% train split each,
                                              #   e.g. DATASET=mimic_transfers,bpi12  (see configs/data/multi.yaml)
LOGGER      ?= wandb                            # wandb | none  (W&B runs offline; sync with `make sync`)
DATA        ?=                                # empty -> bundled data/raw; set to /shared/... for a shared FS
OUT         ?=                                # empty -> project outputs/;  set to override

NCPS_SBATCH := slurm/ncps/pretrain.sbatch
TUHH_SBATCH := slurm/tuhh/pretrain.sbatch

# Optional Hydra passthrough for any run target: EXTRA='k=v k2=v2', and (eval only) SEEDS=<N>.
EXTRA       ?=
SEEDS       ?=
# Per-head switches for backbone pretraining. Each takes: on (config weight, default) | off (0)
# | a NUMBER (custom weight, e.g. NEXT_TIME=0.5). Ablate/reweight any head, any combination.
# TAG labels the run (folded into the run id) so combinations are distinguishable.
# Current on-weights: ACT=1.0, NEXT_TIME/REMAINING/OUTCOME=0.3, JEPA=0.1 (configs/ar/default.yaml).
ACT         ?= on
NEXT_TIME   ?= on
REMAINING   ?= on
OUTCOME     ?= on
JEPA        ?= on
TAG         ?=
HEAD_EXTRA  := $(if $(filter on,$(ACT)),,ar.act_weight=$(patsubst off,0,$(ACT))) \
               $(if $(filter on,$(NEXT_TIME)),,ar.time_weight=$(patsubst off,0,$(NEXT_TIME))) \
               $(if $(filter on,$(REMAINING)),,ar.remaining_time_weight=$(patsubst off,0,$(REMAINING))) \
               $(if $(filter on,$(OUTCOME)),,ar.outcome_weight=$(patsubst off,0,$(OUTCOME))) \
               $(if $(filter on,$(JEPA)),,ar.jepa_weight=$(patsubst off,0,$(JEPA))) \
               $(if $(TAG),tag=-$(TAG),)

# Common env passthrough to the sbatch scripts (they read IMAGE/EXPERIMENT/LOGGER/DATA/OUT/EXTRA).
ENV = IMAGE=$(IMAGE) EXPERIMENT=$(EXPERIMENT) DATASET=$(DATASET) LOGGER=$(LOGGER) EXTRA='$(strip $(EXTRA) $(HEAD_EXTRA))' $(if $(DATA),DATA=$(DATA) )$(if $(OUT),OUT=$(OUT) )
EVAL ?= compare                               # eval mode (eval_ncps/eval_oland): label_efficiency|compare|zero_shot
EVAL_DATASET ?=                              # downstream eval dataset (BPI12|BPI17); empty -> config default
EVAL_ENV = IMAGE=$(IMAGE) EXTRA='$(EXTRA)' SEEDS='$(SEEDS)' EVAL_DATASET='$(EVAL_DATASET)'

.PHONY: help build submit_ncps submit_tuhh submit_oland eval_oland setup_native login_native sync_native eval_ncps backbones queue login sync

## build : build the Apptainer container into the project dir (removes any stale image first)
build:
	@img="$(strip $(IMAGE))"; \
	case "$$img" in \
	  "") echo "refusing to clean: IMAGE is empty"; exit 1 ;; \
	  *.sif) [ -f "$$img" ] && { echo "removing stale $$img"; rm -f -- "$$img"; } || true ;; \
	  *) echo "refusing to clean: IMAGE=$$img is not a .sif file"; exit 1 ;; \
	esac
	$(APPTAINER) build $(IMAGE) $(DEF)

## submit_ncps : submit the NCPS job (single-GPU, trainer=local)
submit_ncps:
	$(ENV) sbatch $(NCPS_SBATCH)

## submit_tuhh : submit the TUHH job (2-node DDP)
submit_tuhh:
	$(ENV) sbatch $(TUHH_SBATCH)

## backbones : list backbone run ids you can evaluate (from a finished pretrain)
backbones:
	@ls -1 outputs/backbones 2>/dev/null || echo "(none yet — run `make submit_ncps` first)"

## eval_ncps : containerized (Apptainer) downstream eval on ncps — EVAL picks the mode
##             (mirror of eval_oland). label_efficiency (single BACKBONE) | compare | zero_shot.
##   usage: make eval_ncps EVAL=compare          BACKBONES='{random:random,role:<id>}' SEEDS=3 EXTRA='...'
##          make eval_ncps EVAL=zero_shot        BACKBONES='{role:<id>}' EXTRA='...'
##          make eval_ncps EVAL=label_efficiency BACKBONE=<id>
eval_ncps:
	@[ -n "$(BACKBONES)$(BACKBONE)" ] || { echo "set BACKBONES='{...}' or BACKBONE=<id>"; exit 1; }
	$(EVAL_ENV) EVAL='$(EVAL)' BACKBONES='$(BACKBONES)' BACKBONE='$(BACKBONE)' sbatch slurm/ncps/eval.sbatch

## setup_native : ONE-TIME env setup on a CONTAINERLESS cluster (oland): install uv + build .venv.
##                 Run this ON the cluster from the repo root (the native analog of `make build`).
setup_native:
	bash scripts/setup_native.sh

## login_native : store your W&B API key on a native cluster (once) — enables ONLINE logging.
##                (native analog of `make login`; oland has internet so runs log live, no sync)
login_native:
	@PATH="$$HOME/.local/bin:$$PATH" uv run wandb login

## sync_native : push any OFFLINE W&B runs from a native cluster (fallback; oland logs online
##               by default, so this is rarely needed — only if you ran with logger.mode=offline).
sync_native:
	@PATH="$$HOME/.local/bin:$$PATH" uv run wandb sync --sync-all outputs/wandb 2>/dev/null || echo "(no offline runs to sync)"

## submit_oland : native (no-container) pretrain on oland — same knobs as submit_ncps.
##   usage: make submit_oland EXPERIMENT=role DATASET=mimic_transfers EXTRA='ar.outcome_labeler=mimic_mortality'
submit_oland:
	$(ENV) sbatch slurm/oland/pretrain.sbatch

## eval_oland : native (no-container) downstream eval on oland.
##   usage: make eval_oland EVAL=compare BACKBONES='{random:random,role:<id>}' SEEDS=3 EXTRA='...'
##          make eval_oland EVAL=zero_shot BACKBONES='{role:<id>}' EXTRA='...'
eval_oland:
	@[ -n "$(BACKBONES)$(BACKBONE)" ] || { echo "set BACKBONES='{...}' or BACKBONE=<id>"; exit 1; }
	$(EVAL_ENV) EVAL='$(EVAL)' BACKBONES='$(BACKBONES)' BACKBONE='$(BACKBONE)' sbatch slurm/oland/eval.sbatch

## queue : show your jobs
queue:
	squeue -u $$USER -o "%.10i %.14P %.18j %.2t %.10M %.6D %R"

## sync : push offline W&B runs to the cloud (LOGIN node; runs wandb INSIDE the container).
##        Key comes from .env (WANDB_API_KEY=…), or `make login` once, or CLI override.
sync:
	$(APPTAINER) exec --bind $(CURDIR):/workspace --pwd /workspace \
	    $(if $(WANDB_API_KEY),--env WANDB_API_KEY=$(WANDB_API_KEY) )$(IMAGE) \
	    wandb sync --sync-all /workspace/outputs/wandb

## help : show the procedure and list targets
help:
	@echo "PM-Foundation HPC — sbatch scripts organized per cluster under slurm/<cluster>/:"
	@echo "  CONTAINERIZED (Apptainer):  ncps (1-GPU), tuhh (2-node DDP)"
	@echo "  NATIVE uv (no container):   oland"
	@echo
	@echo "Containerized clusters (ncps/tuhh):        Native clusters (oland):"
	@echo "  1. make build       (build .sif once)      1. make setup_native  (uv + .venv, once)"
	@echo "  2. make submit_ncps / submit_tuhh          2. make submit_oland"
	@echo "  3. make eval_ncps EVAL=compare|zero_shot|label_efficiency          3. make eval_oland EVAL=compare|zero_shot|label_efficiency"
	@echo "  4. make sync        (push offline W&B)      (W&B logs online directly — no sync)"
	@echo "     make backbones / make queue    (both: list run ids / job status)"
	@echo
	@echo "Overrides: EXPERIMENT=<arch>  DATASET=bpi12|bpi12_bpi17  LOGGER=wandb|none  OUT=/..."
	@echo "  arch: baseline|mlp|mlp_gate|fourier|time2vec|rope|tbias|rope_time2vec_tbias_gate|role"
	@echo "           EXTRA='hydra.k=v ...' (any target)   SEEDS=<N> (eval: N seeds for error bands)"
	@echo "  pretrain heads (each on|off|<weight>): ACT NEXT_TIME REMAINING OUTCOME JEPA  (on-weights 1.0/0.3/0.3/0.3/0.1)"
	@echo "  TAG=<label> to name the run"
	@echo
	@echo "Targets:"
	@grep -hE '^## ' $(firstword $(MAKEFILE_LIST)) | sed 's/^## /  /'
