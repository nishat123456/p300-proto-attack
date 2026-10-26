"""Post-execution numerical envelope, separate from the original strict protocol."""
import math,struct
CONTRACT={"id":"float32_conversion_envelope_20261026_v1","status":"post_execution_revised_analysis","bound":"max(original_epsilon,nearest_float32(original_epsilon))","original_strict_failure_retained":True,"causal_rounding_equivalence_proven":False,"paired_validation_tasks":[10,15],"analysis_seed":20261013}
def check_budget(applied,epsilon):
 if not math.isfinite(applied) or not math.isfinite(epsilon) or applied<0 or epsilon<0:raise ValueError('nonfinite/negative budget')
 upper=max(epsilon,struct.unpack('f',struct.pack('f',epsilon))[0])
 if not math.isfinite(upper) or applied>upper:raise ValueError('outside exact numerical envelope')
 return applied>epsilon
