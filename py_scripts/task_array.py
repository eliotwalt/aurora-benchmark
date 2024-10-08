import yaml
import sys

def count_jobs(config: str|dict):
    """
    Count the number of jobs as defined by the configuration file.
    
    There will be 1 job for each climatology frequency and variable combination.
    """
    if isinstance(config, str):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
        
    climatology_frequencies = config['climatology_frequencies']
    num_climatology_frequencies = len(climatology_frequencies)
    
    num_variables = sum([
        len(config.get('static_variables', [])),
        len(config.get('surface_variables', [])),
        len(config.get('atmospheric_variables', []))
    ])
    
    return num_climatology_frequencies * num_variables

def get_job_config(config: str|dict, task_id):
    """
    Creates a sub-config for a specific job.
    
    Each subconfig contains a single climatology frequency and variable combination.
    """
    if isinstance(config, str):
        with open(config, 'r') as file:
            config = yaml.safe_load(file)
            
    total_jobs = count_jobs(config)
    num_climatology_frequencies = len(config['climatology_frequencies'])
        
    if task_id >= total_jobs:
        raise ValueError("task_id is out of range")
    
    # compute indexes
    climatology_index = task_id % num_climatology_frequencies
    variable_index = task_id // num_climatology_frequencies
    
    # get climatology frequency
    climatology_frequency = config['climatology_frequencies'][climatology_index]
    
    # get the variable
    all_vars = config.get('static_variables', []) + config.get('surface_variables', []) + config.get('atmospheric_variables', [])
    all_var_types = ["static_variables"] * len(config.get('static_variables', [])) + \
                    ["surface_variables"] * len(config.get('surface_variables', [])) + \
                    ["atmospheric_variables"] * len(config.get('atmospheric_variables', []))
        
    sub_config = {
        **config,
        'climatology_frequencies': [climatology_frequency],
        'static_variables': [all_vars[variable_index]] if all_var_types[variable_index] == "static_variables" else [],
        'surface_variables': [all_vars[variable_index]] if all_var_types[variable_index] == "surface_variables" else [],
        'atmospheric_variables': [all_vars[variable_index]] if all_var_types[variable_index] == "atmospheric_variables" else []
    }

    return sub_config
    

if __name__ == "__main__":
    # Example usage
    config_path = sys.argv[1]
    # print(count_jobs(config_path))
    # with open(config_path, 'r') as file:
    #     config = yaml.safe_load(file)

    n = count_jobs(config_path)
    # for i in range(n):
    #     print(get_job_config(config, i))
    #     print()

    print(n)