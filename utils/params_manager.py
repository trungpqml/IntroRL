import json


class ParamsManager:
    def __init__(self, params_file):
        """A class to manage the Parameters. Parameters include configuration parameters and Hyper-parameters
        :param params_file: Path to the parameters JSON file
        """
        self.params = json.load(params_file, 'r')

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
