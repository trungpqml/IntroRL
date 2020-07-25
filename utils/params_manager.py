import json


class ParamsManager:
    def __init__(self, params_file):
        """A class to manage the Parameters. Parameters include configuration parameters and Hyper-parameters
        :param params_file: Path to the parameters JSON file
        """
        self.params = json.load(open(params_file, 'r'))

    def get_params(self):
        """
        return all the parameters
        :return: The whole parameter dictionary
        """
        return self.params

    def get_env_params(self):
        """
        Returns the environment configuration parameters
        :return: A dictionary of configuration parameters used for the environment
        """
        return self.params['env']

    def get_agent_params(self):
        """
        Returns the hyper-parameters and configuration parameters used by the agent
        :return: A dictionary of parameters used by the agent
        """
        return self.params['agent']

    def update_agent_params(self, **kwargs):
        """
        Update the hyper-parameters (and configuration parameters) used by the agent
        :params kwargs: Comma-separated, hyper-parameter-key=value pairs. Eg.: ls=0.005, gamma=0.98 
        :return: None
        """
        for key, value in kwargs.items():
            if key in self.params['agent'].keys():
                self.params['agent'][key] = value

    def export_env_params(self, file_name):
        """
        Export the environment parameters to the specified file. Useful for logging experiment specific parameters
        :param file_name: Name of the file to write the agent parameters to
        :return:
        """
        with open(file_name, 'w') as f:
            json.dump(self.params['env'], f, indent=4,
                      separators=(',', ': '), sort_keys=True)
            f.write('\n')  # for POSIX compatibility

    def export_agent_params(self, file_name):
        """
        Export the agent parameters to the specified file.Useful for logging experiment specific parameters.
        :param file_name: Name of the file to write the agent parameters to
        :return:
        """
        with open(file_name, 'w') as f:
            json.dump(self.params['agent'], f, indent=4,
                      separators=(',', ': '), sort_keys=True)
            f.write('\n')  # for POSIX compatibility


if __name__ == "__main__":
    print("Testing ParamsManager")
    param_file = "parameters.json"
    params_manager = ParamsManager(param_file)

    agent_params = params_manager.get_agent_params()
    print("Agent params:")
    for k, v in agent_params.items():
        print(k, ":", v)

    env_params = params_manager.get_env_params()
    print("Environment params:")
    for k, v in env_params.items():
        print(k, ":", v)

    params_manager.update_agent_params(lr=0.01, gamma=0.95)
    updated_agent_params = params_manager.get_agent_params()
    print("Updated agent params:")
    for k, v in updated_agent_params.items():
        print(k, ":", v)
    print("ParamsManager test completed.")
