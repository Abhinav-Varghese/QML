#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_name>"
    exit 1
fi

scan_name="$1"
pt_mults=$(seq 1 15)

# Setup snapshot
date_str=$(date +%Y-%m-%d)
log_dir="/vols/cms/lgc124/QMLoutputs/qubit_scan_logs/${date_str}/${scan_name}"
code_dir="${log_dir}/code_snapshot"

mkdir -p "$code_dir"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/qubitCode "$code_dir/qubitCode"
cp -r /home/hep/lgc124/PhDCode/LXAutoQML/AutoQML/jointCode "$code_dir/jointCode"

echo "Code snapshot saved to $code_dir"
mkdir -p "${log_dir}/condor_logs"

# Wrapper script
wrapper="${code_dir}/run_job.sh"

cat > "$wrapper" << 'WRAPEOF'
#!/bin/bash

scan_name=$1
idx=$2
pt_mult=$3
seed=$4
code_dir=$5

source /home/hep/lgc124/miniforge3/bin/activate AutoQMLEnv

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1

PYTHONPATH=$code_dir python $code_dir/qubitCode/qubit_main.py \
    --config=$code_dir/qubitCode/qubit_config.yaml \
    data.pt_mult=$pt_mult \
    misc.random_seed=$seed \
    misc.scan_id=$scan_name \
    misc.scan_index=$idx \
    misc.scan_keys=data.pt_mult,misc.random_seed
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
+MaxRuntime = 10800

SUBMITEOF

i=0

for n in $pt_mults; do
    pt_mult=$(awk -v n="$n" 'BEGIN {printf "%.1f", n/10}')
    for seed in $(seq 0 4); do
        echo "arguments = ${scan_name} ${i} ${pt_mult} ${seed} ${code_dir}" >> "$submit_file"
        echo "queue" >> "$submit_file"
        i=$((i+1))
    done
done

echo "Submitting $i jobs..."
condor_submit "$submit_file"
echo "Done. Logs at ${log_dir}/condor_logs/"