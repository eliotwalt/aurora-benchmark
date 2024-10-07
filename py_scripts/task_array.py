import yaml
import sys

def count_jobs(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    resampling_frequencies = config['resampling_frequencies']
    num_resampling_frequencies = len(resampling_frequencies)
    
    min_variables = min(
        len(config.get('static_variables', [])),
        len(config.get('surface_variables', [])),
        len(config.get('atmospheric_variables', []))
    )
    
    return num_resampling_frequencies * min_variables  # +1 for the last sub-config with all remaining variables

def get_job_config(config, task_id):
    resampling_frequencies = config['resampling_frequencies']
    num_resampling_frequencies = len(resampling_frequencies)
    
    static_vars = config.get('static_variables', [])
    surface_vars = config.get('surface_variables', [])
    atmospheric_vars = config.get('atmospheric_variables', [])
    
    min_variables = min(len(static_vars), len(surface_vars), len(atmospheric_vars))
    total_jobs = num_resampling_frequencies * min_variables + 1
    
    if task_id >= total_jobs:
        raise ValueError("task_id is out of range")
    
    if task_id == total_jobs - 1:
        # Last sub-config with all remaining variables
        sub_config = {
            **config,
            'resampling_frequencies': resampling_frequencies,
            'static_variables': static_vars[min_variables-1:],
            'surface_variables': surface_vars[min_variables-1:],
            'atmospheric_variables': atmospheric_vars[min_variables-1:]
        }
    else:
        resampling_index = task_id % num_resampling_frequencies
        variable_index = task_id // num_resampling_frequencies
        
        resampling_frequency = resampling_frequencies[resampling_index]
        
        sub_config = {
            **config,
            'resampling_frequencies': [resampling_frequency],
            'static_variables': [static_vars[variable_index]],
            'surface_variables': [surface_vars[variable_index]],
            'atmospheric_variables': [atmospheric_vars[variable_index]]
        }
        
    # ensure that quantile variables are present
    for qvar in sub_config.get('quantile_variables', []):
        if qvar not in sub_config['surface_variables'] + sub_config['atmospheric_variables']:
            sub_config["quantile_variables"].remove(qvar)
    if sub_config["compute_quantile"] and len(sub_config["quantile_variables"]) == 0:
        sub_config["compute_quantile"] = False
    return sub_config
    

# Example usage
config_path = sys.argv[1]
print(count_jobs(config_path))
