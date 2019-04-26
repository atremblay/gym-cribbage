# gym-cribbage
An environment for playing Cribbage in OpenAI gym (2-4 players).

# setup

The gym environment can be installed using `python setup.py install`.

# usage

The environment can be imported after install via:

```
from gym_cribbage.envs.cribbage_env import CribbageEnv
env = CribbageEnv(n_players=2, verbose=False)
```

Where `n_players` can be between 2-4 and `verbose` can be set to `True` for
(copius) debug information.

The environment can be initalized via:

```
state, reward, done, debug = env.reset()
```

Which starts a new game. The environment can be interacted with as a standard
openAI gym environment, I.e., `env.step()`, `env.render()` and `env.reset()`.

The state returned by the environment contains a State() object with the
following fields:

+ `State.hand` -- the current player's playable cards. Non-playable cards are
                  never returned in the hand.
+ `State.hand_id` -- the ID of the current player.
+ `State.reward_id` -- the ID of the player who recieves the reward returned
                       by the environment.
+ `State.phase` -- the current phase of the game. `0` == 'the Deal',
                   `1` == 'the Play', `2` = 'the Show'.
+ `State.player_score` -- the current score (out of 121) of the current player.
+ `State.opponent_score` -- a list of all opponent scores (out of 121).

The cribbage environment cycles through hands an accumulates scores until one
player reaches 121, whereby the environment immediately returns `done==True`
and the game is over. A new game can be started via
`state, reward, done, debug = env.reset()`.

Actions taken during steps have different meaning during each phase of the game.

+ During 'the Deal' (`State.phase == 0`), cards submitted via `env.step(Card)`
  are discarded to the crib. Cards must be taken from the player's hand
  `State.hand[i]`.
+ During 'the Play' (`State.phase == 1`), cards submitted via `env.step(Card)`
  are played to the table to earn points. Again, the must be taken from the
  player's hand `State.hand[i]`.
+ During 'the Show' (`State.phase == 2`), nothing submitted is used by the
  environment. Therefore it is convienient to pass a 'dummy card' to the
  environment `state, reward, done, debug = env.step(Card(99, SUITS[0]))`,
  which can be easily done if one imports some utility classes from
  `cribbage_env`: `from gym_cribbage.envs.cribbage_env import Card, SUITS`.

## Rules
https://en.wikipedia.org/wiki/Cribbage

