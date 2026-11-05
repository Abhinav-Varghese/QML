#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_name>"
    exit 1
fi

scan_name="$1"

train_samples=(10 50 100 500 1000 5000 10000)

# Setup snapshot
date_str=$(date +%Y-%m-%d)
log_dir="/vols/cms/lgc124/QMLoutputs/joint_scan_logs/${date_str}/${scan_name}"
code_dir="${log_dir}/code_snapshot"
mkdir -p "$code_dir"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/quantumCode "$code_dir/quantumCode"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/classicalCode "$code_dir/classicalCode"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/jointCode "$code_dir/jointCode"
echo "Code snapshot saved to $code_dir"

mkdir -p "${log_dir}/condor_logs"

# ── Quantum wrapper ────────────────────────────────────────────────────────────
q_wrapper="${code_dir}/run_quantum.sh"
cat > "$q_wrapper" << 'WRAPEOF'
#!/bin/bash
scan_name=$1
idx=$2
n_samples=$3
seed=$4
code_dir=$5

source /home/hep/lgc124/miniforge3/bin/activate AutoQMLEnv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PYTHONPATH=$code_dir python $code_dir/quantumCode/q_main.py \
    --config=$code_dir/quantumCode/q_config.yaml \
    data.train_num_samples=$n_samples \
    misc.random_seed=$seed \
    misc.scan_id=${scan_name}_Q \
    misc.scan_index=$idx \
    misc.scan_keys=data.train_num_samples,misc.random_seed
WRAPEOF
chmod +x "$q_wrapper"

# ── Classical wrapper ──────────────────────────────────────────────────────────
c_wrapper="${code_dir}/run_classical.sh"
cat > "$c_wrapper" << 'WRAPEOF'
#!/bin/bash
scan_name=$1
idx=$2
n_samples=$3
seed=$4
code_dir=$5

source /home/hep/lgc124/miniforge3/bin/activate AutoQMLEnv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PYTHONPATH=$code_dir python $code_dir/classicalCode/c_main.py \
    --config=$code_dir/classicalCode/c_config.yaml \
    data.train_num_samples=$n_samples \
    misc.random_seed=$seed \
    misc.scan_id=${scan_name}_C \
    misc.scan_index=$idx \
    misc.scan_keys=data.train_num_samples,misc.random_seed
WRAPEOF
chmod +x "$c_wrapper"

# ── Submit file ────────────────────────────────────────────────────────────────
submit_file="/tmp/${scan_name}_submit.sub"

# Quantum jobs
cat > "$submit_file" << SUBMITEOF
executable = ${q_wrapper}
initialdir = /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML
requirements = has_avx
output = ${log_dir}/condor_logs/Q_out.\$(Cluster).\$(Process).txt
error  = ${log_dir}/condor_logs/Q_err.\$(Cluster).\$(Process).txt
log    = ${log_dir}/condor_logs/log.\$(Cluster).log
request_cpus = 1
request_memory = 4GB
+MaxRuntime = 10800

SUBMITEOF

i=0
for n_samples in "${train_samples[@]}"; do
    for seed in $(seq 0 99); do
        echo "arguments = ${scan_name} ${i} ${n_samples} ${seed} ${code_dir}" >> "$submit_file"
        echo "queue" >> "$submit_file"
        i=$((i+1))
    done
done

# Classical jobs — continue indexing from where quantum left off
cat >> "$submit_file" << SUBMITEOF2

executable = ${c_wrapper}
initialdir = /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML
requirements = has_avx
output = ${log_dir}/condor_logs/C_out.\$(Cluster).\$(Process).txt
error  = ${log_dir}/condor_logs/C_err.\$(Cluster).\$(Process).txt
log    = ${log_dir}/condor_logs/log.\$(Cluster).log
request_cpus = 1
request_memory = 4GB
+MaxRuntime = 4000

SUBMITEOF2

j=0
for n_samples in "${train_samples[@]}"; do
    for seed in $(seq 0 99); do
        echo "arguments = ${scan_name} ${j} ${n_samples} ${seed} ${code_dir}" >> "$submit_file"
        echo "queue" >> "$submit_file"
        j=$((j+1))
    done
done

total=$((i + j))
echo "Submitting $total jobs ($i quantum, $j classical)..."
condor_submit "$submit_file"
echo "Done. Logs at ${log_dir}/condor_logs/"