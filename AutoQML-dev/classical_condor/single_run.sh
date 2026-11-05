#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_name>"
    exit 1
fi

scan_name="$1"
num_repeats=100

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
seed=$3
code_dir=$4

source /home/hep/lgc124/miniforge3/bin/activate AutoQMLEnv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PYTHONPATH=$code_dir python $code_dir/classicalCode/c_main.py \
    --config=$code_dir/classicalCode/c_config.yaml \
    misc.scan_id=$scan_name \
    misc.scan_index=$idx \
    misc.seed=$seed
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

# Submit num_repeats jobs, each using encoder_shape from config and sequential seed
for ((i=0; i<num_repeats; i++)); do
    echo "arguments = ${scan_name} ${i} ${i} ${code_dir}" >> "$submit_file"
    echo "queue" >> "$submit_file"
done

echo "Submitting ${num_repeats} jobs..."
condor_submit "$submit_file"
echo "Done. Logs at ${log_dir}/condor_logs/"