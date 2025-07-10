## $1 == model-name ; $2 == device-cpus

## Only one test is needed, regardless of weights if noise == 0
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2

# ## lw = 1.0, ew = 0.0
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.25 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2

## lw = 0.75, ew = 0.25
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.75 --energy-weight 0.25 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.75 --energy-weight 0.25 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.75 --energy-weight 0.25 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.75 --energy-weight 0.25 --device-max-energy 0.0 --max-noises 0.25 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.75 --energy-weight 0.25 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2

## lw = 0.5, ew = 0.5
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.5 --energy-weight 0.5 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.5 --energy-weight 0.5 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.5 --energy-weight 0.5 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.5 --energy-weight 0.5 --device-max-energy 0.0 --max-noises 0.25 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.5 --energy-weight 0.5 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2

## lw = 0.25, ew = 0.75
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.25 --energy-weight 0.75 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.25 --energy-weight 0.75 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.25 --energy-weight 0.75 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.25 --energy-weight 0.75 --device-max-energy 0.0 --max-noises 0.25 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.25 --energy-weight 0.75 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2

## lw = 0.0, ew = 1.0
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.0 --energy-weight 1.0 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.0 --energy-weight 1.0 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.0 --energy-weight 1.0 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.0 --energy-weight 1.0 --device-max-energy 0.0 --max-noises 0.25 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
./start_test.sh DeviceOnlyPlan.py --model $1 --latency-weight 0.0 --energy-weight 1.0 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 10 --plan-deploy-runs 1 --plan-use-runs 100 --device-cpus $2
