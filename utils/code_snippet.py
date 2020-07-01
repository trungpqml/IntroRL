def replay_experience(self, batch_size=REPLAY_BATCH_SIZE):
    """Replays a mini-batch of experiecen sampled from Experience Memory
    :param batch_size: mini-batch size to sample from the Experience Memroy
    :return: None
    """
    experience_batch = self.memory.sample(batch_size)
    self.learn_from_batch_experience(experience_batch)


def learn_from_batch_experience(self, experiences):
    """ Updated the DQN based on the learning from a mini-batch of experience
    :param experiences: A mini-batch of experience
    :return: None
    """
    batch_xp = Experience(*zip(*experiences))
    obs_batch = np.array(batch_xp.obs)
    action_batch = np.array(batch_xp.action)
    reward_batch = np.array(batch_xp.reward)
    next_obs_batch = np.array(batch_xp.next_obs)
    done_batch = np.array(batch_xp.done)
    td_target = reward_batch + ~done_batch * \
        np.tile(self.gamma, len(next_obs_batch)) * \
        self.Q(next_obs_batch).detach().max(1)[0].data
    td_target = td_target.to(device)
    action_idx = torch.from_numpy(action_batch).to(device)
    td_error = torch.nn.functional.mse_loss(
        self.Q(obs_batch).gather(1, action_idx.view(-1, 1)), td_target.float().unsqueeze(1))
    self.Q_optimizer.zero_grad()
    self.Q_optimizer.step()
