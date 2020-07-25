# IntroRL
A simple program to play Atari games.

#### Clean up the output folder by running:
```
make clean
```

#### Pick any environment from the list of Atari environments and train the agent by using the following command. For example:
```
python main.py --env "PongNoFrameskip-v4"
```

#### To see how the agent learns, pass --render when running script. For example:
```
python main.py --env "PongNoFrameskip-v4" --render
```

#### Test agent after training, run the script as follow, replace the environment ID with yours:
```
python main.py --env "Seaquest-v0" --test --render
```
