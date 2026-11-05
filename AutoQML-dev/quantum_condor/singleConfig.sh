#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_name>"
    exit 1
fi

scan_name="$1"

# Setup snapshot
date_str=$(date +%Y-%m-%d)
log_dir="/vols/cms/lgc124/QMLoutputs/quantum_scan_logs/${date_str}/${scan_name}"
code_dir="${log_dir}/code_snapshot"

mkdir -p "$code_dir"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/quantumCode "$code_dir/quantumCode"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/jointCode "$code_dir/jointCode"

echo "Code snapshot saved to $code_dir"

mkdir -p "${log_dir}/condor_logs"

# Wrapper script
wrapper="${code_dir}/run_job.sh"

cat > "$wrapper" << 'WRAPEOF'
#!/bin/bash

scan_name=$1
idx=$2
seed=$3
code_dir=$4

source /home/hep/lgc124/miniforge3/bin/activate AutoQMLEnv

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PYTHONPATH=$code_dir python $code_dir/quantumCode/q_main.py \
    --config=$code_dir/quantumCode/q_config.yaml \
    misc.random_seed=$seed \
    misc.scan_id=$scan_name \
    misc.scan_index=$idx \
    misc.scan_keys=misc.random_seed
WRAPEOF

chmod +x "$wrapper"

# Submit file
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
+MaxRuntime = 8000

SUBMITEOF

i=0
for seed in $(seq 0 0); do
    echo "arguments = ${scan_name} ${i} ${seed} ${code_dir}" >> "$submit_file"
    echo "queue" >> "$submit_file"
    i=$((i+1))
done

echo "Submitting $i jobs..."
condor_submit "$submit_file"
echo "Done. Logs at ${log_dir}/condor_logs/"