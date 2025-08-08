## $1 == device-cpus

## As the energy consumption is proportional to time and the execution is only on device, it does not matter which weights are used
## In this case we are mostly evaluating the impact of model quantization on execution time

## Only one test is needed, regardless of weights if noise == 0
## As there is linear relation between latency and energy, it is enough to run the test with a single couple when using only one device lw = 1.0, ew = 0.0

############################## YOLO11n-cls ##############################
# ./start_test.sh DeviceOnlyPlan.py --model yolo11n-cls --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11n-cls --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1

# ############################## YOLO11m-det ##############################
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.025 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.125 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.15 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.175 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.2 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
# ./start_test.sh DeviceOnlyPlan.py --model yolo11m --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1

############################## YOLO11x-seg ##############################
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.025 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.05 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.075 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.1 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.125 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.15 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.175 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.2 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
./start_test.sh DeviceOnlyPlan.py --model yolo11x-seg --latency-weight 1.0 --energy-weight 0.0 --device-max-energy 0.0 --max-noises 0.5 --plan-gen-runs 1 --plan-deploy-runs 1 --plan-use-runs 25 --device-cpus $1
