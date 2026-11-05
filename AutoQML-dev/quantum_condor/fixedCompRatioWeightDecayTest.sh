#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_name>"
    exit 1
fi

scan_name="$1"

bs_types=(
    # "trainable_alltoall"
    # "trainable_ring"
    # "trainable_chain"
    # "trainable_star"
    "trainable_int"
)

wires=(1 2 3 4 5 6)
wd_values=(0.0)
train_n=10000
n_seeds=5   # 10 repeats per config (seeds 0..9)

# Setup snapshot
date_str=$(date +%Y-%m-%d)
log_dir="/vols/cms/lgc124/QMLoutputs/quantum_scan_logs/${date_str}/${scan_name}"
code_dir="${log_dir}/code_snapshot"
mkdir -p "$code_dir"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/quantumCode "$code_dir/quantumCode"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/jointCode "$code_dir/jointCode"
echo "Code snapshot saved to $code_dir"

mkdir -p "${log_dir}/condor_logs"

# Write wrapper script
wrapper="${code_dir}/run_job.sh"
cat > "$wrapper" << 'WRAPEOF'
#!/bin/bash
scan_name=$1
idx=$2
w=$3
trash=$4
bs=$5
seed=$6
train_n=$7
code_dir=$8
wd=$9

source /home/hep/lgc124/miniforge3/bin/activate AutoQMLEnv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PYTHONPATH=$code_dir python $code_dir/quantumCode/q_main.py \
    --config=$code_dir/quantumCode/q_config.yaml \
    circuit.wires=$w \
    circuit.trash_modes="[$trash]" \
    circuit.bs_substrate_type=$bs \
    misc.random_seed=$seed \
    data.train_num_samples=$train_n \
    training.opt_weight_decay=$wd \
    misc.scan_id=$scan_name \
    misc.scan_index=$idx \
    misc.scan_keys=circuit.wires,circuit.bs_substrate_type,misc.random_seed,training.opt_weight_decay
WRAPEOF
chmod +x "$wrapper"

# Write submit file
submit_file="/tmp/${scan_name}_submit.sub"
cat > "$submit_file" << SUBMITEOF
executable = ${wrapper}
requirements = has_avx
initialdir = /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML
output = ${log_dir}/condor_logs/out.\$(Cluster).\$(Process).txt
error  = ${log_dir}/condor_logs/err.\$(Cluster).\$(Process).txt
log    = ${log_dir}/condor_logs/log.\$(Cluster).log
request_cpus = 1
request_memory = 4GB
+MaxRuntime = 10800

SUBMITEOF

i=0
for w in "${wires[@]}"; do
    # 2:1 kept:trash ratio -> trash = w/2 modes (integer division)
    n_trash=$((w / 2))
    if [ "$n_trash" -lt 1 ]; then
        n_trash=1
    fi
    trash=""
    for ((t=w-n_trash; t<w; t++)); do
        if [ -z "$trash" ]; then
            trash="$t"
        else
            trash="${trash},${t}"
        fi
    done

    for bs in "${bs_types[@]}"; do
        for wd in "${wd_values[@]}"; do
            for seed in $(seq 0 $((n_seeds - 1))); do
                echo "arguments = ${scan_name} ${i} ${w} ${trash} ${bs} ${seed} ${train_n} ${code_dir} ${wd}" >> "$submit_file"
                echo "queue" >> "$submit_file"
                i=$((i+1))
            done
        done
    done
done

echo "Submitting $i jobs..."
condor_submit "$submit_file"
echo "Done. Logs at ${log_dir}/condor_logs/"