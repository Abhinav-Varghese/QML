#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_name>"
    exit 1
fi

scan_name="$1"

wires=(8 6 4 2)
train_num_samples_vals=(10 50 100 500 1000 5000)

# Setup snapshot
date_str=$(date +%Y-%m-%d)
log_dir="/vols/cms/lgc124/QMLoutputs/classical_scan_logs/${date_str}/${scan_name}"
code_dir="${log_dir}/code_snapshot"
mkdir -p "$code_dir"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/classicalCode "$code_dir/classicalCode"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/jointCode "$code_dir/jointCode"
echo "Code snapshot saved to $code_dir"

mkdir -p "${log_dir}/condor_logs"

# Write wrapper script
wrapper="${code_dir}/run_job.sh"
cat > "$wrapper" << 'WRAPEOF'
#!/bin/bash
scan_name=$1
idx=$2
encoder_shape=$3
seed=$4
train_n=$5
code_dir=$6

source /home/hep/lgc124/miniforge3/bin/activate AutoQMLEnv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PYTHONPATH=$code_dir python $code_dir/classicalCode/c_main.py \
    --config=$code_dir/classicalCode/c_config.yaml \
    circuit.encoder_shape="[$encoder_shape]" \
    misc.random_seed=$seed \
    data.train_num_samples=$train_n \
    misc.scan_id=$scan_name \
    misc.scan_index=$idx \
    misc.scan_keys=circuit.encoder_shape,misc.random_seed,data.train_num_samples
WRAPEOF
chmod +x "$wrapper"

# Write submit file
submit_file="/tmp/${scan_name}_submit.sub"
cat > "$submit_file" << SUBMITEOF
executable = ${wrapper}
initialdir = /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML
requirements = has_avx
output = ${log_dir}/condor_logs/out.\$(Cluster).\$(Process).txt
error  = ${log_dir}/condor_logs/err.\$(Cluster).\$(Process).txt
log    = ${log_dir}/condor_logs/log.\$(Cluster).log
request_cpus = 1
request_memory = 4GB
+MaxRuntime = 1000

SUBMITEOF

i=0
for w in "${wires[@]}"; do
    n_trash=$((w / 2))
    shape="${w},${n_trash}"

    for train_n in "${train_num_samples_vals[@]}"; do
        for seed in $(seq 0 99); do
            echo "arguments = ${scan_name} ${i} ${shape} ${seed} ${train_n} ${code_dir}" >> "$submit_file"
            echo "queue" >> "$submit_file"
            i=$((i+1))
        done
    done
done

echo "Submitting $i jobs..."
condor_submit "$submit_file"
echo "Done. Logs at ${log_dir}/condor_logs/"