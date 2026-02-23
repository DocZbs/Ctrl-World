CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
gg python nexus/run_nexus.py --config nexus/configs/droid_data_expert1_pick_place_into_iter3.yaml

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
gg python nexus/run_nexus.py --config nexus/configs/droid_data_expert2_pick_place_onto_iter3.yaml