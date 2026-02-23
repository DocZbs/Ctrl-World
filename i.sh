CTRLWORLD_POLICY_WARMUP=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.20 \
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python nexus/run_nexus.py --config nexus/configs/droid_data_main_chunk000_009_iter3.yaml
